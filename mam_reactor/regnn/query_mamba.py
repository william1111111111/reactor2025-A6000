from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from regnn.conditional_model import (
    ConditionalREGNN,
    masked_ccc,
)

try:
    from mamba_ssm import Mamba
except ImportError as error:  # pragma: no cover - exercised in deployment
    Mamba = None
    _MAMBA_IMPORT_ERROR = error
else:
    _MAMBA_IMPORT_ERROR = None


STYLE_DESCRIPTOR_VERSION = 1
STYLE_DESCRIPTOR_DIM = 75


@dataclass(frozen=True)
class QueryMambaConfig:
    num_queries: int = 10
    query_dim: int = 64
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    layers: int = 2
    dropout: float = 0.0
    max_residual_logit: float = 1.5
    bidirectional: bool = True


def reverse_valid_prefix(
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Reverse each valid prefix and keep padded positions at zero."""
    batch, frames, channels = values.shape
    frame_ids = torch.arange(frames, device=values.device)
    valid = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)
    reverse_ids = (
        lengths.unsqueeze(1) - 1 - frame_ids.unsqueeze(0)
    ).clamp(min=0, max=frames - 1)
    reversed_values = values.gather(
        1,
        reverse_ids.unsqueeze(-1).expand(batch, frames, channels),
    )
    return reversed_values * valid.unsqueeze(-1).to(reversed_values)


def apply_residual_channel_multipliers(
    model: nn.Module,
    output: Dict[str, torch.Tensor],
    multipliers: Sequence[float],
) -> torch.Tensor:
    """Rebuild anchor-plus-query predictions with per-channel residual scales.

    The anchor candidate is returned unchanged.  Only the learned dynamic and
    static query residuals are scaled, after their ordinary query gates and
    dynamic/style scales have already been applied.
    """

    required = (
        "base_raw_prediction",
        "base_prediction",
        "dynamic_residual_raw",
        "style_residual_raw",
    )
    missing = [name for name in required if name not in output]
    if missing:
        raise KeyError(
            "Channel-scaled prediction requires residual outputs; missing "
            f"{missing}"
        )
    scale = torch.as_tensor(
        multipliers,
        device=output["dynamic_residual_raw"].device,
        dtype=output["dynamic_residual_raw"].dtype,
    )
    output_dim = output["dynamic_residual_raw"].shape[-1]
    if scale.shape != (output_dim,):
        raise ValueError(
            "residual_channel_multipliers must have one value per output "
            f"channel, expected {output_dim}, got {tuple(scale.shape)}"
        )
    if torch.any(scale < 0.0):
        raise ValueError("Residual channel multipliers must be non-negative")
    combined_residual = (
        output["dynamic_residual_raw"] + output["style_residual_raw"]
    ) * scale
    query_raw = output["base_raw_prediction"].unsqueeze(1) + combined_residual
    query_prediction = model.anchor.constrain_reaction(query_raw)
    return torch.cat(
        (output["base_prediction"].unsqueeze(1), query_prediction),
        dim=1,
    )


class QueryConditionedMambaBlock(nn.Module):
    def __init__(
        self,
        config: QueryMambaConfig,
    ):
        super().__init__()
        if Mamba is None:
            raise RuntimeError(
                "mamba_ssm is required for Query-Residual Mamba"
            ) from _MAMBA_IMPORT_ERROR
        self.bidirectional = config.bidirectional
        self.norm = nn.LayerNorm(config.d_model)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.query_dim, 2 * config.d_model),
        )
        self.forward_mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        if self.bidirectional:
            self.backward_mamba = Mamba(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )
            self.direction_projection = nn.Linear(
                2 * config.d_model,
                config.d_model,
            )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        query_condition: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(inputs)
        scale, shift = self.film(query_condition).chunk(2, dim=-1)
        normalized = (
            normalized
            * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
            + shift.unsqueeze(1)
        )
        forward_output = self.forward_mamba(normalized)
        if self.bidirectional:
            reversed_inputs = reverse_valid_prefix(normalized, lengths)
            backward_output = self.backward_mamba(reversed_inputs)
            backward_output = reverse_valid_prefix(
                backward_output,
                lengths,
            )
            update = self.direction_projection(
                torch.cat((forward_output, backward_output), dim=-1)
            )
        else:
            update = forward_output
        return inputs + self.dropout(update)


class QueryConditionedFrameMLPBlock(nn.Module):
    """Parameter-matched frame-wise control for temporal Mamba blocks.

    Query FiLM, normalization, residual structure, depth, and output width are
    preserved.  The only removed capability is cross-frame state propagation.
    With d_model=128 and hidden_dim=1034 this block has 282,762 parameters,
    versus 282,752 for the bidirectional Mamba block.
    """

    def __init__(
        self,
        config: QueryMambaConfig,
        hidden_dim: int = 1034,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("Frame-MLP hidden_dim must be positive")
        self.norm = nn.LayerNorm(config.d_model)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.query_dim, 2 * config.d_model),
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        query_condition: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(inputs)
        scale, shift = self.film(query_condition).chunk(2, dim=-1)
        normalized = (
            normalized
            * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
            + shift.unsqueeze(1)
        )
        update = self.frame_mlp(normalized)
        frame_ids = torch.arange(
            inputs.shape[1],
            device=inputs.device,
        )
        valid = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)
        update = update * valid.unsqueeze(-1).to(update)
        return inputs + self.dropout(update)


class QueryResidualMamba(nn.Module):
    """Frozen conditional-mean anchor plus K-1 query Mamba residuals."""

    def __init__(
        self,
        anchor: ConditionalREGNN,
        config: QueryMambaConfig,
    ):
        super().__init__()
        if config.num_queries < 2:
            raise ValueError("num_queries must be at least two")
        self.anchor = anchor
        self.config = config
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()

        residual_queries = config.num_queries - 1
        self.query_embeddings = nn.Parameter(
            torch.empty(residual_queries, config.query_dim)
        )
        nn.init.orthogonal_(self.query_embeddings)
        self.speaker_style_projection = nn.Sequential(
            nn.LayerNorm(anchor.config.hidden_dim),
            nn.Linear(anchor.config.hidden_dim, config.query_dim),
            nn.Tanh(),
        )
        self.query_interaction = nn.Sequential(
            nn.Linear(3 * config.query_dim, 2 * config.query_dim),
            nn.SiLU(),
            nn.Linear(2 * config.query_dim, config.query_dim),
            nn.LayerNorm(config.query_dim),
        )
        self.context_projection = nn.Linear(
            anchor.config.hidden_dim,
            config.d_model,
        )
        self.query_input_projection = nn.Linear(
            config.query_dim,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            [
                QueryConditionedMambaBlock(config)
                for _ in range(config.layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.to_residual = nn.Linear(
            config.d_model,
            anchor.config.output_dim,
        )
        nn.init.normal_(self.to_residual.weight, std=1e-3)
        nn.init.zeros_(self.to_residual.bias)

    def train(self, mode: bool = True) -> "QueryResidualMamba":
        super().train(mode)
        self.anchor.eval()
        return self

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
        residual_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        with torch.no_grad():
            anchor_output = self.anchor(
                speaker_audio,
                speaker_emotion,
                speaker_3dmm,
                lengths,
            )
        context = anchor_output["context"].detach()
        base_raw = anchor_output["raw_prediction"].detach()
        base_prediction = anchor_output["prediction"].detach()
        valid_mask = anchor_output["valid_mask"]
        batch, frames, _ = context.shape
        lengths = torch.as_tensor(
            lengths,
            device=context.device,
            dtype=torch.long,
        ).clamp(min=1, max=frames)

        weights = valid_mask.unsqueeze(-1).to(context)
        pooled_context = (
            (context * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )
        speaker_style = self.speaker_style_projection(pooled_context)
        query = self.query_embeddings.unsqueeze(0).expand(batch, -1, -1)
        speaker = speaker_style.unsqueeze(1).expand_as(query)
        query_condition = self.query_interaction(
            torch.cat((query, speaker, query * speaker), dim=-1)
        )

        residual_queries = self.config.num_queries - 1
        hidden = (
            self.context_projection(context).unsqueeze(1)
            + self.query_input_projection(query_condition).unsqueeze(2)
        )
        hidden = hidden.reshape(
            batch * residual_queries,
            frames,
            self.config.d_model,
        )
        flat_query_condition = query_condition.reshape(
            batch * residual_queries,
            self.config.query_dim,
        )
        flat_lengths = (
            lengths.unsqueeze(1)
            .expand(batch, residual_queries)
            .reshape(-1)
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                flat_query_condition,
                flat_lengths,
            )
        residual = residual_scale * self.config.max_residual_logit * torch.tanh(
            self.to_residual(self.output_norm(hidden))
        )
        residual = residual.view(
            batch,
            residual_queries,
            frames,
            self.anchor.config.output_dim,
        )
        residual = residual * valid_mask[:, None, :, None].to(residual)

        query_raw = base_raw.unsqueeze(1) + residual
        all_raw = torch.cat((base_raw.unsqueeze(1), query_raw), dim=1)
        predictions = self.anchor.constrain_reaction(all_raw)
        predictions = (
            predictions
            * valid_mask[:, None, :, None].to(predictions)
        )
        return {
            "prediction": predictions,
            "base_prediction": base_prediction,
            "raw_prediction": all_raw,
            "residual_raw": residual,
            "context": context,
            "valid_mask": valid_mask,
            "query_condition": query_condition,
        }


def reaction_style_descriptor(
    reactions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable length-invariant [mean, std, |velocity|] descriptor."""
    if reactions.ndim == 3:
        reactions = reactions.unsqueeze(1)
    if reactions.ndim != 4 or reactions.shape[-1] != 25:
        raise ValueError("reactions must have shape [B,K,T,25]")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [B,T]")

    weights = valid_mask[:, None, :, None].to(reactions)
    count = weights.sum(dim=2).clamp_min(1.0)
    mean = (reactions * weights).sum(dim=2) / count
    variance = (
        (reactions - mean.unsqueeze(2)).square() * weights
    ).sum(dim=2) / count
    standard_deviation = torch.sqrt(variance + 1e-8)

    if reactions.shape[2] > 1:
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        pair_weights = pair_mask[:, None, :, None].to(reactions)
        pair_count = pair_weights.sum(dim=2).clamp_min(1.0)
        mean_absolute_velocity = (
            (reactions[:, :, 1:] - reactions[:, :, :-1]).abs()
            * pair_weights
        ).sum(dim=2) / pair_count
    else:
        mean_absolute_velocity = torch.zeros_like(mean)
    return torch.cat(
        (mean, standard_deviation, mean_absolute_velocity),
        dim=-1,
    )


def balanced_sinkhorn_cost(
    cost: torch.Tensor,
    temperature: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.ndim != 3 or cost.shape[1] != cost.shape[2]:
        raise ValueError("cost must be square [B,K,K]")
    if temperature <= 0 or iterations <= 0:
        raise ValueError("temperature and iterations must be positive")
    log_transport = -cost / temperature
    for _ in range(iterations):
        log_transport = (
            log_transport
            - torch.logsumexp(log_transport, dim=2, keepdim=True)
        )
        log_transport = (
            log_transport
            - torch.logsumexp(log_transport, dim=1, keepdim=True)
        )
    transport = torch.exp(log_transport)
    transport = transport / transport.sum(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1e-8)
    return (transport * cost).sum(dim=(1, 2)).mean(), transport


def centered_pairwise_smse(
    predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = valid_mask[:, None, :, None].to(predictions)
    count = weights.sum(dim=2).clamp_min(1.0)
    temporal_mean = (predictions * weights).sum(dim=2) / count
    centered = (
        predictions - temporal_mean.unsqueeze(2)
    ) * weights
    flat = centered.flatten(start_dim=2)
    squared_norm = flat.square().sum(dim=-1)
    squared_distance = (
        squared_norm.unsqueeze(2)
        + squared_norm.unsqueeze(1)
        - 2.0 * torch.bmm(flat, flat.transpose(1, 2))
    ).clamp_min(0.0)
    denominator = (
        valid_mask.sum(dim=1).to(predictions)
        * predictions.shape[-1]
    ).clamp_min(1.0)
    squared_distance = squared_distance / denominator[:, None, None]
    queries = predictions.shape[1]
    off_diagonal = ~torch.eye(
        queries,
        dtype=torch.bool,
        device=predictions.device,
    )
    values = squared_distance[:, off_diagonal]
    return values.mean(), values


class QueryResidualMambaLoss(nn.Module):
    def __init__(
        self,
        style_mean: torch.Tensor,
        style_std: torch.Tensor,
        style_weight: float = 1.0,
        paired_ccc_weight: float = 0.5,
        paired_mse_weight: float = 0.25,
        velocity_weight: float = 0.05,
        diversity_weight: float = 1.0,
        diversity_margin: float = 0.002,
        residual_weight: float = 0.02,
        sinkhorn_temperature: float = 0.1,
        sinkhorn_iterations: int = 8,
    ):
        super().__init__()
        if style_mean.numel() != STYLE_DESCRIPTOR_DIM:
            raise ValueError("Unexpected style descriptor mean dimension")
        if style_std.numel() != STYLE_DESCRIPTOR_DIM:
            raise ValueError("Unexpected style descriptor std dimension")
        self.register_buffer(
            "style_mean",
            style_mean.float().reshape(1, 1, -1),
        )
        self.register_buffer(
            "style_std",
            style_std.float().clamp_min(1e-4).reshape(1, 1, -1),
        )
        self.style_weight = float(style_weight)
        self.paired_ccc_weight = float(paired_ccc_weight)
        self.paired_mse_weight = float(paired_mse_weight)
        self.velocity_weight = float(velocity_weight)
        self.diversity_weight = float(diversity_weight)
        self.diversity_margin = float(diversity_margin)
        self.residual_weight = float(residual_weight)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        paired_target: torch.Tensor,
        target_styles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        predictions = output["prediction"]
        valid_mask = output["valid_mask"]
        batch, queries, frames, channels = predictions.shape
        if target_styles.shape != (
            batch,
            queries - 1,
            STYLE_DESCRIPTOR_DIM,
        ):
            raise ValueError(
                "target_styles must match the K-1 trainable queries"
            )

        query_predictions = predictions[:, 1:]
        predicted_styles = reaction_style_descriptor(
            query_predictions,
            valid_mask,
        )
        normalized_predicted_styles = (
            predicted_styles - self.style_mean
        ) / self.style_std
        normalized_target_styles = (
            target_styles.to(predicted_styles) - self.style_mean
        ) / self.style_std
        style_cost_matrix = (
            normalized_predicted_styles.unsqueeze(2)
            - normalized_target_styles.unsqueeze(1)
        ).square().mean(dim=-1)
        style_ot, transport = balanced_sinkhorn_cost(
            style_cost_matrix,
            temperature=self.sinkhorn_temperature,
            iterations=self.sinkhorn_iterations,
        )

        flat_predictions = predictions.reshape(
            batch * queries,
            frames,
            channels,
        )
        flat_target = (
            paired_target.unsqueeze(1)
            .expand(batch, queries, frames, channels)
            .reshape(batch * queries, frames, channels)
        )
        flat_mask = (
            valid_mask.unsqueeze(1)
            .expand(batch, queries, frames)
            .reshape(batch * queries, frames)
        )
        ccc = masked_ccc(
            flat_predictions,
            flat_target,
            flat_mask,
        )
        paired_ccc = ccc.mean()
        weights = valid_mask[:, None, :, None].to(predictions)
        denominator = (
            weights.sum() * queries * channels
        ).clamp_min(1.0)
        paired_mse = (
            (predictions - paired_target[:, None]).square() * weights
        ).sum() / denominator

        if frames > 1:
            pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            pair_weights = pair_mask[:, None, :, None].to(predictions)
            velocity_denominator = (
                pair_weights.sum() * queries * channels
            ).clamp_min(1.0)
            prediction_velocity = predictions[:, :, 1:] - predictions[:, :, :-1]
            target_velocity = paired_target[:, 1:] - paired_target[:, :-1]
            velocity = (
                (prediction_velocity - target_velocity[:, None]).square()
                * pair_weights
            ).sum() / velocity_denominator
        else:
            velocity = predictions.new_zeros(())

        centered_smse, pairwise_values = centered_pairwise_smse(
            predictions,
            valid_mask,
        )
        diversity_loss = F.relu(
            self.diversity_margin - pairwise_values
        ).mean()
        residual_l2 = output["residual_raw"].square().mean()
        total = (
            self.style_weight * style_ot
            + self.paired_ccc_weight * (1.0 - paired_ccc)
            + self.paired_mse_weight * paired_mse
            + self.velocity_weight * velocity
            + self.diversity_weight * diversity_loss
            + self.residual_weight * residual_l2
        )

        rounded_au = (predictions[..., :15] >= 0.5)
        au_flip_rate = (
            rounded_au[:, 1:]
            != rounded_au[:, :1]
        ).float().mean()
        return {
            "loss": total,
            "style_ot": style_ot,
            "paired_ccc": paired_ccc,
            "paired_mse": paired_mse,
            "velocity": velocity,
            "centered_smse": centered_smse,
            "diversity_loss": diversity_loss,
            "residual_l2": residual_l2,
            "au_flip_rate": au_flip_rate,
            "transport_entropy": -(
                transport
                * torch.log(transport.clamp_min(1e-8))
            ).sum(dim=(1, 2)).mean(),
        }
