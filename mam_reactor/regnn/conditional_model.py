import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConditionalREGNNConfig:
    audio_dim: int = 768
    speaker_emotion_dim: int = 25
    speaker_3dmm_dim: int = 58
    output_dim: int = 25
    hidden_dim: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 8
    graph_dim: int = 64
    graph_layers: int = 2
    graph_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 750
    listener_3dmm: bool = False


def lengths_to_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Return a Transformer padding mask with True on padded frames."""
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    frame_ids = torch.arange(max_len, device=lengths.device)
    return frame_ids.unsqueeze(0) >= lengths.unsqueeze(1)


def restrict_valid_mask_to_tail(
    valid_mask: torch.Tensor,
    tail_frames: int,
) -> torch.Tensor:
    """Restrict loss supervision to a fixed output tail of each window.

    A value of zero preserves the complete valid prefix.  For the official
    REACT2025 online geometry, a 60-frame speaker window uses ``tail_frames=30``
    so the first 30 frames remain context and only the final 30 are outputs.
    """
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [B,T]")
    if tail_frames < 0 or tail_frames > valid_mask.shape[1]:
        raise ValueError("tail_frames must be in [0,T]")
    if tail_frames == 0:
        return valid_mask
    tail_start = valid_mask.shape[1] - tail_frames
    frame_ids = torch.arange(valid_mask.shape[1], device=valid_mask.device)
    return valid_mask & (frame_ids.unsqueeze(0) >= tail_start)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_seq_len: int):
        super().__init__()
        if hidden_dim <= 0 or max_seq_len <= 0:
            raise ValueError("hidden_dim and max_seq_len must be positive")
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_seq_len, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if hidden_dim > 1:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[: encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] > self.encoding.shape[0]:
            raise ValueError(
                f"Sequence length {inputs.shape[1]} exceeds configured "
                f"maximum {self.encoding.shape[0]}"
            )
        return inputs + self.encoding[: inputs.shape[1]].to(inputs)


class ConditionFusion(nn.Module):
    """Encode the same three full-sequence conditions used by diffusion."""

    def __init__(self, config: ConditionalREGNNConfig):
        super().__init__()
        hidden_dim = config.hidden_dim

        def projection(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )

        self.audio_projection = projection(config.audio_dim)
        self.emotion_projection = projection(config.speaker_emotion_dim)
        self.face_projection = projection(config.speaker_3dmm_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
    ) -> torch.Tensor:
        expected_prefix = speaker_audio.shape[:2]
        if (
            speaker_emotion.shape[:2] != expected_prefix
            or speaker_3dmm.shape[:2] != expected_prefix
        ):
            raise ValueError("All conditional streams must share [B,T]")
        encoded = torch.cat(
            (
                self.audio_projection(speaker_audio),
                self.emotion_projection(speaker_emotion),
                self.face_projection(speaker_3dmm),
            ),
            dim=-1,
        )
        return self.fusion(encoded)


class RelationGraphBlock(nn.Module):
    """Learned relation reasoning among 25 facial attributes.

    The block uses full self-attention and does not encode a fixed FACS or
    anatomy-constrained adjacency.
    """

    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        graph_dim: int,
        graph_heads: int,
        dropout: float,
    ):
        super().__init__()
        if graph_dim % graph_heads != 0:
            raise ValueError("graph_dim must be divisible by graph_heads")
        self.output_dim = output_dim
        self.node_embedding = nn.Parameter(
            torch.empty(output_dim, graph_dim)
        )
        nn.init.normal_(self.node_embedding, std=0.02)
        self.context_projection = nn.Linear(context_dim, graph_dim)
        self.value_projection = nn.Linear(1, graph_dim)
        self.attention_norm = nn.LayerNorm(graph_dim)
        self.attention = nn.MultiheadAttention(
            graph_dim,
            graph_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(graph_dim)
        self.ffn = nn.Sequential(
            nn.Linear(graph_dim, 4 * graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * graph_dim, graph_dim),
            nn.Dropout(dropout),
        )
        self.to_residual = nn.Linear(graph_dim, 1)
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def forward(
        self,
        context: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, output_dim = values.shape
        if output_dim != self.output_dim:
            raise ValueError(
                f"Expected {self.output_dim} graph nodes, got {output_dim}"
            )

        node_states = (
            self.context_projection(context).unsqueeze(2)
            + self.value_projection(values.unsqueeze(-1))
            + self.node_embedding.view(1, 1, output_dim, -1)
        )
        flat_nodes = node_states.view(batch * frames, output_dim, -1)
        normalized = self.attention_norm(flat_nodes)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        flat_nodes = flat_nodes + attended
        flat_nodes = flat_nodes + self.ffn(self.ffn_norm(flat_nodes))
        residual = self.to_residual(flat_nodes).view(
            batch, frames, output_dim
        )
        residual = residual * valid_mask.unsqueeze(-1).to(residual)
        return values + residual


class ConditionalREGNN(nn.Module):
    """High-appropriateness deterministic listener-reaction anchor.

    This first-stage model predicts a conditional reaction mean and performs
    learned relation reasoning among facial attributes. Semantic residual
    queries are added only after this anchor is frozen.
    """

    def __init__(self, config: ConditionalREGNNConfig):
        super().__init__()
        if config.output_dim != 25:
            raise ValueError(
                "Official REACT FRC requires all 25 facial attributes"
            )
        if config.hidden_dim % config.temporal_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by temporal_heads"
            )
        self.config = config
        self.condition_fusion = ConditionFusion(config)
        self.position_encoding = SinusoidalPositionEncoding(
            config.hidden_dim,
            config.max_seq_len,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.temporal_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.temporal_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.to_initial_reaction = nn.Linear(
            config.hidden_dim,
            config.output_dim,
        )
        self.to_listener_3dmm = (
            nn.Linear(config.hidden_dim, 58) if config.listener_3dmm else None
        )
        self.graph_blocks = nn.ModuleList(
            [
                RelationGraphBlock(
                    context_dim=config.hidden_dim,
                    output_dim=config.output_dim,
                    graph_dim=config.graph_dim,
                    graph_heads=config.graph_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.graph_layers)
            ]
        )

    @staticmethod
    def constrain_reaction(raw_reaction: torch.Tensor) -> torch.Tensor:
        au = torch.sigmoid(raw_reaction[..., :15])
        valence_arousal = torch.tanh(raw_reaction[..., 15:17])
        expression = torch.softmax(raw_reaction[..., 17:25], dim=-1)
        return torch.cat((au, valence_arousal, expression), dim=-1)

    def encode_conditions(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if speaker_audio.ndim != 3:
            raise ValueError("speaker_audio must have shape [B,T,768]")
        frames = speaker_audio.shape[1]
        lengths = torch.as_tensor(
            lengths,
            device=speaker_audio.device,
            dtype=torch.long,
        ).clamp(min=1, max=frames)
        padding_mask = lengths_to_padding_mask(lengths, frames)
        valid_mask = ~padding_mask

        context = self.condition_fusion(
            speaker_audio,
            speaker_emotion,
            speaker_3dmm,
        )
        context = self.position_encoding(context)
        context = self.temporal_encoder(
            context,
            src_key_padding_mask=padding_mask,
        )
        context = context * valid_mask.unsqueeze(-1).to(context)
        return context, valid_mask

    def decode_context(
        self,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if context.ndim != 3 or valid_mask.shape != context.shape[:2]:
            raise ValueError("context/valid_mask shapes must be [B,T,C]/[B,T]")
        raw_reaction = self.to_initial_reaction(context)
        for graph_block in self.graph_blocks:
            raw_reaction = graph_block(
                context,
                raw_reaction,
                valid_mask,
            )
        prediction = self.constrain_reaction(raw_reaction)
        prediction = prediction * valid_mask.unsqueeze(-1).to(prediction)
        geometry = {}
        if self.to_listener_3dmm is not None:
            geometry["prediction_3dmm"] = self.to_listener_3dmm(context) * valid_mask.unsqueeze(-1).to(context)
        return {
            **geometry,
            "prediction": prediction,
            "raw_prediction": raw_reaction,
            "context": context,
            "valid_mask": valid_mask,
        }

    def forward(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context, valid_mask = self.encode_conditions(
            speaker_audio, speaker_emotion, speaker_3dmm, lengths,
        )
        return self.decode_context(context, valid_mask)


def masked_ccc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """CCC per sample and channel, matching the official 25-D definition."""
    weights = valid_mask.unsqueeze(-1).to(prediction)
    count = weights.sum(dim=1).clamp_min(1.0)
    prediction_mean = (prediction * weights).sum(dim=1) / count
    target_mean = (target * weights).sum(dim=1) / count
    prediction_centered = prediction - prediction_mean.unsqueeze(1)
    target_centered = target - target_mean.unsqueeze(1)
    prediction_variance = (
        prediction_centered.square() * weights
    ).sum(dim=1) / count
    target_variance = (
        target_centered.square() * weights
    ).sum(dim=1) / count
    covariance = (
        prediction_centered * target_centered * weights
    ).sum(dim=1) / count
    return (
        2.0
        * covariance
        / (
            prediction_variance
            + target_variance
            + (prediction_mean - target_mean).square()
            + epsilon
        )
    )


class ConditionalREGNNLoss(nn.Module):
    def __init__(
        self,
        ccc_weight: float = 1.0,
        velocity_weight: float = 0.05,
    ):
        super().__init__()
        self.ccc_weight = float(ccc_weight)
        self.velocity_weight = float(velocity_weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        velocity_supervision: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        weights = valid_mask.unsqueeze(-1).to(prediction)
        denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
        mse = ((prediction - target).square() * weights).sum() / denominator

        ccc = masked_ccc(prediction, target, valid_mask)
        mean_ccc = ccc.mean()
        ccc_loss = 1.0 - mean_ccc

        if prediction.shape[1] > 1:
            pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            if velocity_supervision is not None:
                velocity_supervision = torch.as_tensor(
                    velocity_supervision, device=pair_mask.device,
                    dtype=torch.bool,
                )
                if velocity_supervision.shape != (prediction.shape[0],):
                    raise ValueError("velocity_supervision must have shape [B]")
                pair_mask = pair_mask & velocity_supervision.unsqueeze(1)
            pair_weights = pair_mask.unsqueeze(-1).to(prediction)
            prediction_velocity = prediction[:, 1:] - prediction[:, :-1]
            target_velocity = target[:, 1:] - target[:, :-1]
            velocity_denominator = (
                pair_weights.sum() * prediction.shape[-1]
            ).clamp_min(1.0)
            velocity = (
                (prediction_velocity - target_velocity).square()
                * pair_weights
            ).sum() / velocity_denominator
        else:
            velocity = prediction.new_zeros(())

        total = (
            mse
            + self.ccc_weight * ccc_loss
            + self.velocity_weight * velocity
        )
        return {
            "loss": total,
            "mse": mse,
            "ccc_loss": ccc_loss,
            "ccc": mean_ccc,
            "velocity": velocity,
        }
