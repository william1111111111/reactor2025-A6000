from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from regnn.conditional_model import ConditionalREGNN, masked_ccc
from regnn.frd_surrogate import BandedSoftDTWFRDSurrogate
from regnn.query_mamba import (
    QueryResidualMambaLoss,
    QueryConditionedFrameMLPBlock,
    QueryConditionedMambaBlock,
    balanced_sinkhorn_cost,
    reaction_style_descriptor,
)


NUM_EMOTIONS = 8
NUM_OFFICIAL_QUERIES = 10
OFFICIAL_FRD_GROUPS = (
    (0, 15, 1.0 / 15.0),
    (15, 17, 1.0),
    (17, 25, 1.0 / 8.0),
)
OFFICIAL_FRDIV_GROUPS = (
    (0, 15),
    (15, 17),
    (17, 25),
)
AUVA_DESCRIPTOR_INDICES = tuple(
    index
    for offset in (0, 25, 50)
    for index in range(offset, offset + 17)
)


@dataclass(frozen=True)
class EmotionQueryMambaConfig:
    """Configuration for anchor + 8 semantic queries + one mixture query."""

    num_queries: int = NUM_OFFICIAL_QUERIES
    num_emotions: int = NUM_EMOTIONS
    query_dim: int = 64
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    layers: int = 2
    dropout: float = 0.0
    max_residual_logit: float = 1.5
    bidirectional: bool = True
    temporal_backbone: str = "mamba"
    mlp_hidden_dim: int = 1034
    compatibility_mode: str = "target_support"
    # Retained so old serialized configs remain parseable.
    compatibility_temperature: float = 1.0
    gate_temperature: float = 1.0
    gate_floor: float = 0.05
    gate_warmup_steps: int = 0
    gate_top_m: int = 0
    mixture_mode: str = "predicted_support"
    static_style_adapter: bool = False
    style_adapter_hidden: int = 128
    max_style_logit: float = 0.75
    listener_3dmm: bool = False

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Dict[str, object],
    ) -> "EmotionQueryMambaConfig":
        """Load old checkpoints without silently changing q9 semantics."""
        values = dict(payload)
        if "compatibility_mode" not in values:
            values["compatibility_mode"] = "legacy_relative_softmax"
        if "gate_temperature" not in values:
            values["gate_temperature"] = values.get(
                "compatibility_temperature",
                1.0,
            )
        values.setdefault("gate_warmup_steps", 0)
        values.setdefault("gate_top_m", 0)
        values.setdefault("mixture_mode", "predicted_support")
        values.setdefault("temporal_backbone", "mamba")
        values.setdefault("mlp_hidden_dim", 1034)
        return cls(**values)


def masked_emotion_profile(
    reactions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a normalized clip-level profile from expression channels 17:25."""
    if reactions.ndim != 3 or reactions.shape[-1] != 25:
        raise ValueError("reactions must have shape [B,T,25]")
    if valid_mask.shape != reactions.shape[:2]:
        raise ValueError("valid_mask must match reactions [B,T]")
    weights = valid_mask.unsqueeze(-1).to(reactions)
    profile = (reactions[..., 17:25] * weights).sum(dim=1)
    profile = profile / weights.sum(dim=1).clamp_min(1.0)
    return profile.clamp_min(0.0) / profile.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)


def target_set_emotion_support(
    targets: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-emotion max support across available target clips."""
    if targets.ndim != 4 or targets.shape[-1] != 25:
        raise ValueError("targets must have shape [B,K,T,25]")
    if target_valid_mask.shape != targets.shape[:3]:
        raise ValueError("target_valid_mask must have shape [B,K,T]")
    batch, target_count, frames, channels = targets.shape
    profiles = masked_emotion_profile(
        targets.reshape(batch * target_count, frames, channels),
        target_valid_mask.reshape(batch * target_count, frames),
    ).reshape(batch, target_count, NUM_EMOTIONS)
    available = target_valid_mask.any(dim=-1)
    profiles = profiles.masked_fill(~available[..., None], 0.0)
    return profiles.amax(dim=1).clamp(0.0, 1.0)


def semantic_identity_from_support(
    emotion_profiles: torch.Tensor,
    target_support: torch.Tensor,
    mode: str = "support_weighted",
    support_power: float = 0.5,
    min_support: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Supervise only emotion-query identities supported by the target set."""
    if emotion_profiles.ndim != 3 or emotion_profiles.shape[1:] != (
        NUM_EMOTIONS,
        NUM_EMOTIONS,
    ):
        raise ValueError("emotion_profiles must have shape [B,8,8]")
    if target_support.shape != emotion_profiles.shape[:2]:
        raise ValueError("target_support must have shape [B,8]")
    if mode not in {"support_weighted", "uniform"}:
        raise ValueError("Unknown semantic identity mode")
    if support_power <= 0.0:
        raise ValueError("support_power must be positive")
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support must be in [0,1]")

    semantic_targets = torch.arange(
        NUM_EMOTIONS,
        device=emotion_profiles.device,
    ).unsqueeze(0).expand(emotion_profiles.shape[0], -1)
    target_probability = emotion_profiles.gather(
        -1,
        semantic_targets.unsqueeze(-1),
    ).squeeze(-1)
    per_query_nll = -torch.log(target_probability.clamp_min(1e-8))
    if mode == "uniform":
        support_weights = torch.ones_like(target_support)
    else:
        detached_support = target_support.detach().clamp(0.0, 1.0)
        support_weights = detached_support.pow(float(support_power))
        if min_support > 0.0:
            support_weights = support_weights * (
                detached_support >= float(min_support)
            ).to(support_weights)
    denominator = support_weights.sum().clamp_min(1e-8)
    identity = (per_query_nll * support_weights).sum() / denominator
    correct = (
        emotion_profiles.argmax(dim=-1) == semantic_targets
    ).to(emotion_profiles)
    return {
        "loss": identity,
        "accuracy": correct.mean(),
        "weighted_accuracy": (
            correct * support_weights
        ).sum() / denominator,
        "mean_weight": support_weights.mean(),
        "weights": support_weights,
    }


def independent_support_probabilities(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Independent Bernoulli support probabilities for eight emotions."""
    if logits.ndim != 2 or logits.shape[-1] != NUM_EMOTIONS:
        raise ValueError("logits must have shape [B,8]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(logits / temperature)


def _support_top_m_mask(
    support_probs: torch.Tensor,
    top_m: int,
) -> torch.Tensor:
    if not 0 <= top_m <= support_probs.shape[-1]:
        raise ValueError("top_m must be in [0,num_emotions]")
    if top_m == 0 or top_m == support_probs.shape[-1]:
        return torch.ones_like(support_probs, dtype=torch.bool)
    indices = torch.topk(
        support_probs,
        k=top_m,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    mask = torch.zeros_like(support_probs, dtype=torch.bool)
    return mask.scatter(-1, indices, True)


def normalize_support_mixture(
    support_probs: torch.Tensor,
    top_m: int = 0,
) -> torch.Tensor:
    """Normalize non-exclusive support probabilities for q9 mixing."""
    mask = _support_top_m_mask(support_probs, top_m)
    selected = support_probs * mask.to(support_probs)
    denominator = selected.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(selected, 1.0 / selected.shape[-1])
    return torch.where(
        denominator > 1e-8,
        selected / denominator.clamp_min(1e-8),
        uniform,
    )


def apply_support_gates(
    support_probs: torch.Tensor,
    gate_floor: float,
    warmup_alpha: float = 1.0,
    top_m: int = 0,
) -> torch.Tensor:
    """Map support probabilities to residual gates with optional warmup."""
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0,1]")
    if not 0.0 <= warmup_alpha <= 1.0:
        raise ValueError("warmup_alpha must be in [0,1]")
    mask = _support_top_m_mask(support_probs, top_m)
    active_support = support_probs * mask.to(support_probs)
    predicted_gates = gate_floor + (1.0 - gate_floor) * active_support
    return 1.0 + float(warmup_alpha) * (predicted_gates - 1.0)


GATE_CONTROL_MODES = (
    "predicted",
    "constant_mean",
    "emotion_prior",
    "shuffled_predicted",
    "ungated",
)


def deterministic_shuffle_permutation(
    num_examples: int,
    seed: int,
    max_identity_fraction: float = 0.1,
) -> torch.Tensor:
    """Build a full-evaluation permutation independent of batch size."""
    if num_examples < 2:
        raise ValueError("Shuffled gate control requires at least 2 examples")
    if not 0.0 <= max_identity_fraction < 1.0:
        raise ValueError("max_identity_fraction must be in [0,1)")
    identity = torch.arange(num_examples)
    maximum_fixed = int(math.floor(max_identity_fraction * num_examples))
    for attempt in range(32):
        generator = torch.Generator().manual_seed(
            int(seed) + attempt * 1_000_003
        )
        permutation = torch.randperm(num_examples, generator=generator)
        if int((permutation == identity).sum()) <= maximum_fixed:
            return permutation
    # Deterministic derangement fallback. The non-zero cyclic shift has no
    # fixed points and still preserves the complete set of support vectors.
    shift = abs(int(seed)) % (num_examples - 1) + 1
    return torch.roll(identity, shifts=shift)


def resolve_gate_control_support(
    predicted_support: torch.Tensor,
    mode: str,
    *,
    support_prior: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    constant_support: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Resolve the support source while leaving q9 policy to the caller."""
    if predicted_support.ndim != 2 or predicted_support.shape[-1] != 8:
        raise ValueError("predicted_support must have shape [B,8]")
    if mode not in GATE_CONTROL_MODES:
        raise ValueError(f"Unknown gate control mode: {mode}")
    batch = predicted_support.shape[0]
    if mode == "predicted":
        support = predicted_support
    elif mode == "constant_mean":
        if constant_support is None:
            raise ValueError(
                "constant_mean requires a precomputed constant_support"
            )
        constant = torch.as_tensor(
            constant_support,
            dtype=predicted_support.dtype,
            device=predicted_support.device,
        )
        if constant.numel() != 1:
            raise ValueError("constant_support must be scalar")
        support = constant.reshape(1, 1).expand_as(predicted_support)
    elif mode == "emotion_prior":
        if support_prior is None:
            raise ValueError("emotion_prior requires support_prior")
        prior = torch.as_tensor(
            support_prior,
            dtype=predicted_support.dtype,
            device=predicted_support.device,
        )
        if prior.shape != (8,):
            raise ValueError("support_prior must have shape [8]")
        support = prior.unsqueeze(0).expand(batch, -1)
    elif mode == "shuffled_predicted":
        if permutation is None:
            raise ValueError("shuffled_predicted requires permutation")
        permutation = torch.as_tensor(
            permutation,
            dtype=torch.long,
            device=predicted_support.device,
        )
        if permutation.shape != (batch,):
            raise ValueError("permutation must have shape [B]")
        if not torch.equal(
            permutation.sort().values,
            torch.arange(batch, device=permutation.device),
        ):
            raise ValueError("permutation must contain each batch index once")
        support = predicted_support[permutation]
    else:
        support = torch.ones_like(predicted_support)
    return support.clamp(0.0, 1.0)


def resolve_emotion_query_gates(
    predicted_support: torch.Tensor,
    mode: str,
    gate_strength: float,
    gate_floor: float,
    support_prior: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    constant_support: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Apply one matched-control support source with shared alpha semantics.

    ``g(alpha) = 1 - alpha * (1 - (f + (1-f)*support))``.
    """
    if not 0.0 <= gate_strength <= 1.0:
        raise ValueError("gate_strength must be in [0,1]")
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0,1]")
    if gate_strength == 0.0 or mode == "ungated":
        return torch.ones_like(predicted_support)
    support = resolve_gate_control_support(
        predicted_support,
        mode,
        support_prior=support_prior,
        permutation=permutation,
        constant_support=constant_support,
    )
    base_gate = gate_floor + (1.0 - gate_floor) * support
    return 1.0 - float(gate_strength) * (1.0 - base_gate)


def resolve_control_mixture_weights(
    predicted_support: torch.Tensor,
    mode: str,
    *,
    control_q9_mixture: bool,
    support_prior: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    constant_support: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Keep q9 predicted by default; optionally apply the control support."""
    if not control_q9_mixture:
        return normalize_support_mixture(predicted_support)
    control_support = resolve_gate_control_support(
        predicted_support,
        mode,
        support_prior=support_prior,
        permutation=permutation,
        constant_support=constant_support,
    )
    return normalize_support_mixture(control_support)


def masked_pairwise_smse(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-example raw S-MSE aligned with the official FRDiv geometry."""
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("Pairwise predictions must have identical shapes")
    if prediction_a.ndim != 3 or prediction_a.shape[-1] != 25:
        raise ValueError("Pairwise predictions must have shape [B,T,25]")
    if valid_mask.shape != prediction_a.shape[:2]:
        raise ValueError("valid_mask must match predictions [B,T]")
    weights = valid_mask.unsqueeze(-1).to(prediction_a)
    denominator = (
        weights.sum(dim=(1, 2)) * prediction_a.shape[-1]
    ).clamp_min(1.0)
    return (
        (prediction_a - prediction_b).square() * weights
    ).sum(dim=(1, 2)) / denominator


def masked_style_pairwise_smse(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-example MSE between temporal channel means (static style)."""
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("Pairwise predictions must have identical shapes")
    if prediction_a.ndim != 3 or prediction_a.shape[-1] != 25:
        raise ValueError("Pairwise predictions must have shape [B,T,25]")
    if valid_mask.shape != prediction_a.shape[:2]:
        raise ValueError("valid_mask must match predictions [B,T]")
    weights = valid_mask.unsqueeze(-1).to(prediction_a)
    frame_count = weights.sum(dim=1).clamp_min(1.0)
    mean_a = (prediction_a * weights).sum(dim=1) / frame_count
    mean_b = (prediction_b * weights).sum(dim=1) / frame_count
    return (mean_a - mean_b).square().mean(dim=-1)


def masked_centered_pairwise_smse(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-example MSE after removing each channel's temporal mean."""
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("Pairwise predictions must have identical shapes")
    if prediction_a.ndim != 3 or prediction_a.shape[-1] != 25:
        raise ValueError("Pairwise predictions must have shape [B,T,25]")
    if valid_mask.shape != prediction_a.shape[:2]:
        raise ValueError("valid_mask must match predictions [B,T]")
    weights = valid_mask.unsqueeze(-1).to(prediction_a)
    frame_count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_a = (prediction_a * weights).sum(dim=1, keepdim=True) / frame_count
    mean_b = (prediction_b * weights).sum(dim=1, keepdim=True) / frame_count
    centered_difference = (prediction_a - mean_a) - (
        prediction_b - mean_b
    )
    denominator = (
        weights.sum(dim=(1, 2)) * prediction_a.shape[-1]
    ).clamp_min(1.0)
    return (
        centered_difference.square() * weights
    ).sum(dim=(1, 2)) / denominator


def masked_all_query_style_pairwise_smse(
    predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-style S-MSE across every ordered pair of query candidates."""
    if predictions.ndim != 4 or predictions.shape[-1] != 25:
        raise ValueError("predictions must have shape [B,Q,T,25]")
    if valid_mask.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("valid_mask must match predictions [B,T]")
    weights = valid_mask[:, None, :, None].to(predictions)
    frame_count = weights.sum(dim=2).clamp_min(1.0)
    temporal_mean = (predictions * weights).sum(dim=2) / frame_count
    squared_norm = temporal_mean.square().sum(dim=-1)
    squared_distance = (
        squared_norm.unsqueeze(2)
        + squared_norm.unsqueeze(1)
        - 2.0
        * torch.bmm(temporal_mean, temporal_mean.transpose(1, 2))
    ).clamp_min(0.0) / predictions.shape[-1]
    queries = predictions.shape[1]
    off_diagonal = ~torch.eye(
        queries,
        dtype=torch.bool,
        device=predictions.device,
    )
    values = squared_distance[:, off_diagonal]
    return values.mean(), values


def masked_all_query_official_frdiv(
    predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable official FRDiv geometry with straight-through AU rounding.

    The forward value matches ``compute_s_mse`` after the official offline
    path rounds the first 15 AU channels.  The straight-through estimator
    keeps useful gradients for AU values that must cross the 0.5 boundary.
    """
    if predictions.ndim != 4 or predictions.shape[-1] != 25:
        raise ValueError("predictions must have shape [B,Q,T,25]")
    if valid_mask.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("valid_mask must match predictions [B,T]")
    rounded_au = predictions[..., :15] + (
        torch.round(predictions[..., :15]) - predictions[..., :15]
    ).detach()
    official_predictions = torch.cat(
        (rounded_au, predictions[..., 15:]),
        dim=-1,
    )
    weights = valid_mask[:, None, :, None].to(predictions)
    flattened = (official_predictions * weights).flatten(start_dim=2)
    squared_norm = flattened.square().sum(dim=-1)
    squared_distance = (
        squared_norm.unsqueeze(2)
        + squared_norm.unsqueeze(1)
        - 2.0 * torch.bmm(flattened, flattened.transpose(1, 2))
    ).clamp_min(0.0)
    denominator = (
        valid_mask.sum(dim=1).to(predictions) * predictions.shape[-1]
    ).clamp_min(1.0)
    squared_distance = squared_distance / denominator[:, None, None]
    queries = predictions.shape[1]
    off_diagonal = ~torch.eye(
        queries,
        dtype=torch.bool,
        device=predictions.device,
    )
    per_example = squared_distance[:, off_diagonal].mean(dim=-1)
    return per_example.mean(), per_example


def masked_all_query_official_frdiv_groups(
    predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additively split differentiable official FRDiv over three groups.

    All groups retain the official denominator ``T*25`` so their three
    contributions sum to the total S-MSE.  AU uses the same straight-through
    rounding as :func:`masked_all_query_official_frdiv`.

    Returns ``(mean_groups, per_example_groups)`` with shapes ``[3]`` and
    ``[B,3]`` ordered as AU, VA, expression.
    """

    if predictions.ndim != 4 or predictions.shape[-1] != 25:
        raise ValueError("predictions must have shape [B,Q,T,25]")
    if valid_mask.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("valid_mask must match predictions [B,T]")
    rounded_au = predictions[..., :15] + (
        torch.round(predictions[..., :15]) - predictions[..., :15]
    ).detach()
    official_predictions = torch.cat(
        (rounded_au, predictions[..., 15:]),
        dim=-1,
    )
    weights = valid_mask[:, None, :, None].to(predictions)
    denominator = (
        valid_mask.sum(dim=1).to(predictions) * predictions.shape[-1]
    ).clamp_min(1.0)
    queries = predictions.shape[1]
    off_diagonal = ~torch.eye(
        queries,
        dtype=torch.bool,
        device=predictions.device,
    )
    per_group = []
    for start, end in OFFICIAL_FRDIV_GROUPS:
        flattened = (
            official_predictions[..., start:end] * weights
        ).flatten(start_dim=2)
        squared_norm = flattened.square().sum(dim=-1)
        squared_distance = (
            squared_norm.unsqueeze(2)
            + squared_norm.unsqueeze(1)
            - 2.0 * torch.bmm(flattened, flattened.transpose(1, 2))
        ).clamp_min(0.0)
        squared_distance = squared_distance / denominator[:, None, None]
        per_group.append(squared_distance[:, off_diagonal].mean(dim=-1))
    per_example_groups = torch.stack(per_group, dim=-1)
    return per_example_groups.mean(dim=0), per_example_groups


def masked_valid_channel_soft_tlcc(
    predictions: torch.Tensor,
    speaker_emotion: torch.Tensor,
    valid_mask: torch.Tensor,
    max_lag: int = 48,
    lag_step: int = 12,
    temporal_stride: int = 4,
    temperature: float = 0.05,
    variance_epsilon: float = 1e-6,
    min_overlap: int = 3,
) -> Dict[str, torch.Tensor]:
    """Differentiable robust-TLCC surrogate for the nine learned queries.

    The forward geometry follows the robust diagnostic metric: each lag is
    scored by the mean Pearson correlation over channels that vary in both
    the prediction and speaker sequence.  Constant channels are detached from
    the objective instead of being artificially animated.  AU channels use
    straight-through rounding so the forward pass matches offline scoring.

    A softmax over coarse lag scores makes the expected absolute lag
    differentiable.  Query zero is deliberately excluded because it is the
    frozen Conditional-REGNN anchor.
    """
    if predictions.ndim != 4 or predictions.shape[-1] != 25:
        raise ValueError("predictions must have shape [B,Q,T,25]")
    if predictions.shape[1] < 2:
        raise ValueError("predictions must include anchor + learned queries")
    if speaker_emotion.shape != (
        predictions.shape[0],
        predictions.shape[2],
        predictions.shape[3],
    ):
        raise ValueError("speaker_emotion must have shape [B,T,25]")
    if valid_mask.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("valid_mask must have shape [B,T]")
    if max_lag <= 0 or max_lag >= predictions.shape[2]:
        raise ValueError("max_lag must be positive and smaller than T")
    if lag_step <= 0 or max_lag % lag_step != 0:
        raise ValueError("lag_step must be positive and divide max_lag")
    if temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if variance_epsilon < 0.0:
        raise ValueError("variance_epsilon must be non-negative")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least two")

    learned_predictions = predictions[:, 1:]
    rounded_prediction_au = learned_predictions[..., :15] + (
        torch.round(learned_predictions[..., :15])
        - learned_predictions[..., :15]
    ).detach()
    official_predictions = torch.cat(
        (rounded_prediction_au, learned_predictions[..., 15:]),
        dim=-1,
    ).float()
    official_speaker = torch.cat(
        (
            torch.round(speaker_emotion[..., :15]),
            speaker_emotion[..., 15:],
        ),
        dim=-1,
    ).float()

    lags = torch.arange(
        -max_lag,
        max_lag + 1,
        lag_step,
        device=predictions.device,
        dtype=torch.long,
    )
    lag_scores = []
    lag_valid = []
    lag_valid_channel_counts = []
    for lag_value in lags.tolist():
        if lag_value > 0:
            shifted_prediction = official_predictions[:, :, lag_value:]
            shifted_speaker = official_speaker[:, :-lag_value]
            shifted_mask = (
                valid_mask[:, lag_value:] & valid_mask[:, :-lag_value]
            )
        elif lag_value < 0:
            shift = -lag_value
            shifted_prediction = official_predictions[:, :, :-shift]
            shifted_speaker = official_speaker[:, shift:]
            shifted_mask = valid_mask[:, :-shift] & valid_mask[:, shift:]
        else:
            shifted_prediction = official_predictions
            shifted_speaker = official_speaker
            shifted_mask = valid_mask

        shifted_prediction = shifted_prediction[:, :, ::temporal_stride]
        shifted_speaker = shifted_speaker[:, ::temporal_stride]
        shifted_mask = shifted_mask[:, ::temporal_stride]

        weights = shifted_mask[:, None, :, None].float()
        count = weights.sum(dim=2)
        safe_count = count.clamp_min(1.0)
        prediction_mean = (
            shifted_prediction * weights
        ).sum(dim=2) / safe_count
        speaker_mean = (
            shifted_speaker[:, None] * weights
        ).sum(dim=2) / safe_count
        prediction_centered = (
            shifted_prediction - prediction_mean[:, :, None]
        ) * weights
        speaker_centered = (
            shifted_speaker[:, None] - speaker_mean[:, :, None]
        ) * weights
        prediction_ss = prediction_centered.square().sum(dim=2)
        speaker_ss = speaker_centered.square().sum(dim=2)
        covariance = (
            prediction_centered * speaker_centered
        ).sum(dim=2)
        prediction_variance = prediction_ss / safe_count
        speaker_variance = speaker_ss / safe_count
        valid_channels = (
            (prediction_variance > variance_epsilon)
            & (speaker_variance > variance_epsilon)
            & (count >= min_overlap)
        ).detach()
        correlations = covariance / torch.sqrt(
            (prediction_ss * speaker_ss).clamp_min(1e-12)
        )
        correlations = correlations.clamp(-1.0, 1.0)
        valid_channel_count = valid_channels.sum(dim=-1)
        score = (
            correlations * valid_channels.to(correlations)
        ).sum(dim=-1) / valid_channel_count.clamp_min(1).to(correlations)
        lag_scores.append(score)
        lag_valid.append(valid_channel_count > 0)
        lag_valid_channel_counts.append(valid_channel_count)

    scores = torch.stack(lag_scores, dim=-1)
    valid_lags = torch.stack(lag_valid, dim=-1)
    valid_channel_counts = torch.stack(
        lag_valid_channel_counts,
        dim=-1,
    )
    safe_scores = scores.masked_fill(~valid_lags, -1e4)
    pair_valid = valid_lags.any(dim=-1)
    probabilities = torch.softmax(safe_scores / temperature, dim=-1)
    absolute_lags = lags.abs().to(scores)
    expected_absolute_lag = (
        probabilities * absolute_lags
    ).sum(dim=-1)
    peak_indices = safe_scores.argmax(dim=-1)
    hard_absolute_lag = absolute_lags[peak_indices]
    peak_correlation = safe_scores.gather(
        -1,
        peak_indices.unsqueeze(-1),
    ).squeeze(-1)
    valid_channels_at_peak = valid_channel_counts.gather(
        -1,
        peak_indices.unsqueeze(-1),
    ).squeeze(-1).to(scores)
    zero_index = max_lag // lag_step
    zero_lag_valid = valid_lags[..., zero_index]
    zero_lag_correlation = scores[..., zero_index]

    pair_weights = pair_valid.to(scores)
    pair_denominator = pair_weights.sum().clamp_min(1.0)
    zero_weights = zero_lag_valid.to(scores)
    zero_denominator = zero_weights.sum().clamp_min(1.0)
    normalized_loss = (
        expected_absolute_lag * pair_weights
    ).sum() / pair_denominator / float(max_lag)
    return {
        "sync_loss": normalized_loss,
        "surrogate_sync_expected_abs_lag": (
            expected_absolute_lag * pair_weights
        ).sum() / pair_denominator,
        "surrogate_sync_hard_abs_lag": (
            hard_absolute_lag * pair_weights
        ).sum() / pair_denominator,
        "surrogate_sync_zero_lag_correlation": (
            zero_lag_correlation * zero_weights
        ).sum() / zero_denominator,
        "surrogate_sync_peak_correlation": (
            peak_correlation * pair_weights
        ).sum() / pair_denominator,
        "sync_valid_pair_fraction": pair_weights.mean(),
        "sync_valid_channels_at_peak": (
            valid_channels_at_peak * pair_weights
        ).sum() / pair_denominator,
    }


def masked_official_frd_surrogate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    temporal_stride: int = 12,
    calibration: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable, scale-calibratable surrogate for official FRD.

    Official FRD sums, over the ten predictions, the minimum weighted DTW
    distance to any of ten real targets.  Full differentiable DTW over
    ``[B,10,10,750,750]`` is impractical inside every training step.  This
    surrogate keeps the official AU/VA/emotion groups, weights, target-wise
    minimum, and prediction-wise sum, while using a uniformly sampled
    diagonal path.  The squared path cost is corrected by the ratio of full
    to sampled valid frames, so constant per-frame errors are invariant to
    ``temporal_stride``.

    The first 15 prediction channels use straight-through AU rounding, which
    matches the official forward geometry without blocking gradients.
    ``calibration`` can be fitted on a fixed validation split to express the
    returned value in approximate official-FRD units.

    Returns ``(batch_mean, per_example, pairwise_cost)`` where
    ``pairwise_cost`` has shape ``[B,Q,K]``.
    """
    if predictions.ndim != 4 or predictions.shape[-1] != 25:
        raise ValueError("predictions must have shape [B,Q,T,25]")
    if targets.ndim != 4 or targets.shape[-1] != 25:
        raise ValueError("targets must have shape [B,K,T,25]")
    if (
        predictions.shape[0] != targets.shape[0]
        or predictions.shape[2:] != targets.shape[2:]
    ):
        raise ValueError("predictions and targets must share B,T,25")
    if prediction_valid_mask.shape != (
        predictions.shape[0],
        predictions.shape[2],
    ):
        raise ValueError(
            "prediction_valid_mask must have shape [B,T]"
        )
    if target_valid_mask.shape != targets.shape[:3]:
        raise ValueError("target_valid_mask must have shape [B,K,T]")
    if temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive")
    if calibration <= 0.0:
        raise ValueError("calibration must be positive")

    rounded_au = predictions[..., :15] + (
        torch.round(predictions[..., :15]) - predictions[..., :15]
    ).detach()
    official_predictions = torch.cat(
        (rounded_au, predictions[..., 15:]),
        dim=-1,
    ).float()
    official_targets = torch.cat(
        (torch.round(targets[..., :15]), targets[..., 15:]),
        dim=-1,
    ).float()

    sampled_predictions = official_predictions[:, :, ::temporal_stride]
    sampled_targets = official_targets[:, :, ::temporal_stride]
    sampled_prediction_mask = prediction_valid_mask[:, ::temporal_stride]
    sampled_target_mask = target_valid_mask[:, :, ::temporal_stride]

    full_pair_mask = (
        prediction_valid_mask[:, None, :] & target_valid_mask
    )
    sampled_pair_mask = (
        sampled_prediction_mask[:, None, :]
        & sampled_target_mask
    )
    full_count = full_pair_mask.sum(dim=-1).float()
    sampled_count = sampled_pair_mask.sum(dim=-1).float()
    path_scale = full_count / sampled_count.clamp_min(1.0)
    pair_mask_weights = sampled_pair_mask[:, None, :, :, None].float()

    batch, queries = predictions.shape[:2]
    target_count = targets.shape[1]
    pairwise_cost = torch.zeros(
        batch,
        queries,
        target_count,
        dtype=torch.float32,
        device=predictions.device,
    )
    for start, end, weight in OFFICIAL_FRD_GROUPS:
        difference = (
            sampled_predictions[:, :, None, :, start:end]
            - sampled_targets[:, None, :, :, start:end]
        )
        sampled_squared_path_cost = (
            difference.square() * pair_mask_weights
        ).sum(dim=(-1, -2))
        full_squared_path_cost = (
            sampled_squared_path_cost * path_scale[:, None, :]
        )
        group_distance = torch.sqrt(
            full_squared_path_cost.clamp_min(0.0) + 1e-12
        )
        pairwise_cost = pairwise_cost + weight * group_distance

    invalid_pairs = full_count <= 0
    pairwise_cost = pairwise_cost.masked_fill(
        invalid_pairs[:, None, :],
        torch.finfo(pairwise_cost.dtype).max,
    )
    pairwise_cost = pairwise_cost * float(calibration)
    valid_examples = (~invalid_pairs).any(dim=-1)
    best_target_cost = pairwise_cost.min(dim=-1).values
    best_target_cost = torch.where(
        valid_examples[:, None],
        best_target_cost,
        torch.zeros_like(best_target_cost),
    )
    per_example = best_target_cost.sum(dim=-1)
    valid_weights = valid_examples.to(per_example)
    mean = (
        per_example * valid_weights
    ).sum() / valid_weights.sum().clamp_min(1.0)
    return mean, per_example, pairwise_cost


def masked_sinkhorn_transport(
    cost: torch.Tensor,
    valid_targets: torch.Tensor,
    temperature: float,
    iterations: int,
) -> torch.Tensor:
    """Balanced query transport over only the available target columns.

    Rows always carry uniform mass.  Valid target columns share uniform mass;
    unavailable target columns carry zero mass.  This supports training clips
    with fewer than ten usable same-session reactions without assigning a
    query to an invalid, maximum-cost target.
    """
    if cost.ndim != 3:
        raise ValueError("cost must have shape [B,Q,K]")
    if valid_targets.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError("valid_targets must have shape [B,K]")
    if temperature <= 0.0 or iterations <= 0:
        raise ValueError("temperature and iterations must be positive")
    valid_columns = valid_targets[:, None, :]
    safe_cost = cost.masked_fill(~valid_columns, 0.0)
    log_zero = torch.full_like(safe_cost, -1e4)
    log_transport = torch.where(
        valid_columns,
        -safe_cost / temperature,
        log_zero,
    )
    log_row_mass = -torch.log(
        cost.new_full(
            (cost.shape[0], 1, 1),
            float(cost.shape[1]),
        )
    )
    valid_count = (
        valid_targets.sum(dim=-1, keepdim=True).to(cost).clamp_min(1.0)
    )
    log_column_mass = -torch.log(valid_count).unsqueeze(1)
    for _ in range(iterations):
        log_transport = (
            log_transport
            - torch.logsumexp(log_transport, dim=2, keepdim=True)
            + log_row_mass
        )
        column_logsumexp = torch.logsumexp(
            log_transport,
            dim=1,
            keepdim=True,
        )
        normalized_columns = (
            log_transport - column_logsumexp + log_column_mass
        )
        log_transport = torch.where(
            valid_columns,
            normalized_columns,
            log_zero,
        )
    return torch.exp(log_transport)


def masked_sinkhorn_cost(
    cost: torch.Tensor,
    valid_targets: torch.Tensor,
    temperature: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sinkhorn cost while assigning zero mass to invalid columns."""
    transport = masked_sinkhorn_transport(
        cost,
        valid_targets,
        temperature=temperature,
        iterations=iterations,
    )
    safe_cost = cost.masked_fill(~valid_targets[:, None], 0.0)
    per_example = (transport * safe_cost).sum(dim=(1, 2))
    valid_examples = valid_targets.any(dim=-1).to(cost)
    mean_cost = (
        per_example * valid_examples
    ).sum() / valid_examples.sum().clamp_min(1.0)
    return mean_cost, transport


def masked_nearest_cost(
    cost: torch.Tensor,
    valid_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independently assign every query to its cheapest valid target.

    The returned assignment has total mass one, like the Sinkhorn transport,
    but it does not constrain target marginals. Multiple queries may therefore
    collapse onto the same easy target.
    """
    if cost.ndim != 3:
        raise ValueError("cost must have shape [B,Q,K]")
    if valid_targets.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError("valid_targets must have shape [B,K]")
    valid_columns = valid_targets[:, None, :]
    safe_cost = cost.masked_fill(~valid_columns, float("inf"))
    nearest_cost, nearest_index = safe_cost.min(dim=-1)
    valid_examples = valid_targets.any(dim=-1)
    nearest_cost = torch.where(
        valid_examples[:, None],
        nearest_cost,
        torch.zeros_like(nearest_cost),
    )
    per_example = nearest_cost.mean(dim=-1)
    weights = valid_examples.to(cost)
    mean_cost = (
        per_example * weights
    ).sum() / weights.sum().clamp_min(1.0)
    assignment = torch.zeros_like(cost)
    assignment.scatter_(
        dim=-1,
        index=nearest_index.unsqueeze(-1),
        value=1.0 / cost.shape[1],
    )
    assignment = assignment * valid_examples[:, None, None].to(cost)
    return mean_cost, assignment


def masked_row_softmax_cost(
    cost: torch.Tensor,
    valid_targets: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent soft target selection for each prediction row."""
    if cost.ndim != 3:
        raise ValueError("cost must have shape [B,Q,K]")
    valid_columns = valid_targets[:, None, :]
    logits = (-cost / temperature).masked_fill(~valid_columns, -1e4)
    row_weights = torch.softmax(logits, dim=-1)
    valid_examples = valid_targets.any(dim=-1)
    assignment = row_weights / float(cost.shape[1])
    assignment = assignment * valid_examples[:, None, None].to(cost)
    safe_cost = cost.masked_fill(~valid_columns, 0.0)
    per_example = (assignment * safe_cost).sum(dim=(1, 2))
    weights = valid_examples.to(cost)
    mean_cost = (per_example * weights).sum() / weights.sum().clamp_min(1.0)
    return mean_cost, assignment


def masked_hungarian_cost(
    cost: torch.Tensor,
    valid_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard one-to-one assignment with gradients through selected costs."""
    assignment = torch.zeros_like(cost)
    detached = cost.detach().float().cpu().numpy()
    valid_cpu = valid_targets.detach().cpu().numpy()
    for batch_index in range(cost.shape[0]):
        columns = valid_cpu[batch_index].nonzero()[0]
        if columns.size == 0:
            continue
        rows, local_columns = linear_sum_assignment(
            detached[batch_index][:, columns]
        )
        mass = 1.0 / float(len(rows))
        assignment[batch_index, rows, columns[local_columns]] = mass
    safe_cost = cost.masked_fill(~valid_targets[:, None, :], 0.0)
    per_example = (assignment * safe_cost).sum(dim=(1, 2))
    valid_examples = valid_targets.any(dim=-1).to(cost)
    mean_cost = (
        per_example * valid_examples
    ).sum() / valid_examples.sum().clamp_min(1.0)
    return mean_cost, assignment


def masked_target_set_official_frdiv_groups(
    targets: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FRDiv group components for a target set with per-target masks.

    Only target pairs with at least one shared valid frame contribute. This
    avoids treating zero-length or duplicate placeholder targets as genuine
    diversity evidence.
    """
    if targets.ndim != 4 or targets.shape[-1] != 25:
        raise ValueError("targets must have shape [B,K,T,25]")
    if target_valid_mask.shape != targets.shape[:3]:
        raise ValueError("target_valid_mask must have shape [B,K,T]")
    rounded_au = targets[..., :15] + (
        torch.round(targets[..., :15]) - targets[..., :15]
    ).detach()
    official_targets = torch.cat((rounded_au, targets[..., 15:]), dim=-1)
    batch, target_count, _, _ = targets.shape
    pair_mask = (
        target_valid_mask[:, :, None]
        & target_valid_mask[:, None, :]
    )
    pair_frame_count = pair_mask.sum(dim=-1)
    off_diagonal = ~torch.eye(
        target_count,
        dtype=torch.bool,
        device=targets.device,
    )
    valid_pairs = (pair_frame_count > 0) & off_diagonal[None]
    per_example_groups = []
    for start, end in OFFICIAL_FRDIV_GROUPS:
        difference = (
            official_targets[:, :, None, :, start:end]
            - official_targets[:, None, :, :, start:end]
        )
        squared = (
            difference.square()
            * pair_mask[..., None].to(difference)
        ).sum(dim=(-1, -2))
        # Use the same T*25 denominator as prediction-side FRDiv. This keeps
        # AU, VA, and expression contributions additive and on one scale for
        # channel-allocation supervision.
        denominator = (
            pair_frame_count.to(difference) * targets.shape[-1]
        ).clamp_min(1.0)
        pair_values = squared / denominator
        valid_weights = valid_pairs.to(pair_values)
        per_example_groups.append(
            (pair_values * valid_weights).sum(dim=(1, 2))
            / valid_weights.sum(dim=(1, 2)).clamp_min(1.0)
        )
    per_example = torch.stack(per_example_groups, dim=-1)
    valid_examples = valid_pairs.flatten(start_dim=1).any(dim=1)
    valid_weights = valid_examples.to(targets)
    mean_groups = (
        per_example * valid_weights[:, None]
    ).sum(dim=0) / valid_weights.sum().clamp_min(1.0)
    return mean_groups, per_example, valid_examples


def build_emotion_prototypes(
    descriptors: torch.Tensor,
    num_emotions: int = NUM_EMOTIONS,
    shrinkage: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build smoothed 75-D real-reaction prototypes from the style cache."""
    if descriptors.ndim != 2 or descriptors.shape[1] != 75:
        raise ValueError("descriptors must have shape [N,75]")
    if num_emotions != NUM_EMOTIONS:
        raise ValueError("REACT facial attributes contain exactly 8 emotions")
    if shrinkage < 0.0:
        raise ValueError("shrinkage must be non-negative")

    profiles = descriptors[:, 17:25].clamp_min(0.0)
    labels = profiles.argmax(dim=-1)
    counts = torch.bincount(labels, minlength=num_emotions)
    global_mean = descriptors.mean(dim=0)
    prototypes = []
    for emotion_index in range(num_emotions):
        selected = descriptors[labels == emotion_index]
        class_mean = (
            selected.mean(dim=0)
            if selected.shape[0] > 0
            else global_mean
        )
        count = float(selected.shape[0])
        class_weight = count / (count + shrinkage)
        prototypes.append(
            class_weight * class_mean
            + (1.0 - class_weight) * global_mean
        )
    priors = counts.float() / counts.sum().clamp_min(1)
    return torch.stack(prototypes), counts, priors


class EmotionQueryResidualMamba(nn.Module):
    """Frozen Conditional-REGNN plus identifiable emotion residual queries."""

    def __init__(
        self,
        anchor: ConditionalREGNN,
        config: EmotionQueryMambaConfig,
        freeze_anchor: bool = True,
    ):
        super().__init__()
        if config.num_emotions != NUM_EMOTIONS:
            raise ValueError("Exactly eight emotion queries are required")
        if config.num_queries != config.num_emotions + 2:
            raise ValueError(
                "Queries must be anchor + 8 emotions + one mixture"
            )
        if config.compatibility_mode not in {
            "target_support",
            "legacy_relative_softmax",
        }:
            raise ValueError("Unknown compatibility_mode")
        if config.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if not 0.0 <= config.gate_floor <= 1.0:
            raise ValueError("gate_floor must be in [0,1]")
        if config.gate_warmup_steps < 0:
            raise ValueError("gate_warmup_steps must be non-negative")
        if not 0 <= config.gate_top_m <= config.num_emotions:
            raise ValueError("gate_top_m must be in [0,num_emotions]")
        if config.mixture_mode not in {"predicted_support", "uniform"}:
            raise ValueError("Unknown mixture_mode")
        if config.style_adapter_hidden <= 0:
            raise ValueError("style_adapter_hidden must be positive")
        if config.max_style_logit < 0.0:
            raise ValueError("max_style_logit must be non-negative")
        if config.temporal_backbone not in {"mamba", "mlp"}:
            raise ValueError("temporal_backbone must be 'mamba' or 'mlp'")
        if config.mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive")

        self.anchor = anchor
        self.config = config
        self.freeze_anchor = bool(freeze_anchor)
        self._gate_step = 0
        if self.freeze_anchor:
            for parameter in self.anchor.parameters():
                parameter.requires_grad_(False)
            self.anchor.eval()

        self.emotion_embeddings = nn.Parameter(
            torch.empty(config.num_emotions, config.query_dim)
        )
        nn.init.orthogonal_(self.emotion_embeddings)
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
        self.compatibility_head = nn.Sequential(
            nn.LayerNorm(anchor.config.hidden_dim),
            nn.Linear(anchor.config.hidden_dim, config.num_emotions),
        )
        nn.init.zeros_(self.compatibility_head[-1].weight)
        nn.init.zeros_(self.compatibility_head[-1].bias)
        self.context_projection = nn.Linear(
            anchor.config.hidden_dim,
            config.d_model,
        )
        self.query_input_projection = nn.Linear(
            config.query_dim,
            config.d_model,
        )
        block_factory = (
            (lambda: QueryConditionedMambaBlock(config))
            if config.temporal_backbone == "mamba"
            else (
                lambda: QueryConditionedFrameMLPBlock(
                    config,
                    hidden_dim=config.mlp_hidden_dim,
                )
            )
        )
        self.blocks = nn.ModuleList(
            [block_factory() for _ in range(config.layers)]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.to_3dmm = nn.Linear(config.d_model, 58) if config.listener_3dmm else None
        if self.to_3dmm is not None:
            nn.init.normal_(self.to_3dmm.weight, std=1e-3)
            nn.init.zeros_(self.to_3dmm.bias)
        self.to_residual = nn.Linear(
            config.d_model,
            anchor.config.output_dim,
        )
        nn.init.normal_(self.to_residual.weight, std=1e-3)
        nn.init.zeros_(self.to_residual.bias)
        if config.static_style_adapter:
            self.static_style_adapter = nn.Sequential(
                nn.LayerNorm(config.query_dim),
                nn.Linear(
                    config.query_dim,
                    config.style_adapter_hidden,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.style_adapter_hidden,
                    anchor.config.output_dim,
                ),
            )
            nn.init.zeros_(self.static_style_adapter[-1].weight)
            nn.init.zeros_(self.static_style_adapter[-1].bias)
        else:
            self.static_style_adapter = None

    def train(self, mode: bool = True) -> "EmotionQueryResidualMamba":
        super().train(mode)
        if self.freeze_anchor:
            self.anchor.eval()
        return self

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def set_gate_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("gate step must be non-negative")
        self._gate_step = int(step)

    def _gate_warmup_alpha(self) -> float:
        if not self.training or self.config.gate_warmup_steps == 0:
            return 1.0
        return min(
            1.0,
            self._gate_step / float(self.config.gate_warmup_steps),
        )

    def _anchor_state(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.freeze_anchor:
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
        else:
            anchor_output = self.anchor(
                speaker_audio,
                speaker_emotion,
                speaker_3dmm,
                lengths,
            )
            context = anchor_output["context"]
            base_raw = anchor_output["raw_prediction"]
            base_prediction = anchor_output["prediction"]
        valid_mask = anchor_output["valid_mask"]
        frames = context.shape[1]
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
        compatibility_logits = self.compatibility_head(pooled_context)
        if self.config.compatibility_mode == "legacy_relative_softmax":
            compatibility_probs = torch.softmax(
                compatibility_logits / self.config.gate_temperature,
                dim=-1,
            )
            relative_gates = (
                compatibility_probs
                / compatibility_probs.amax(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            relative_gates = (
                self.config.gate_floor
                + (1.0 - self.config.gate_floor) * relative_gates
            )
            gate_warmup_alpha = 1.0
        else:
            compatibility_probs = independent_support_probabilities(
                compatibility_logits,
                self.config.gate_temperature,
            )
            gate_warmup_alpha = self._gate_warmup_alpha()
            relative_gates = apply_support_gates(
                compatibility_probs,
                gate_floor=self.config.gate_floor,
                warmup_alpha=gate_warmup_alpha,
                top_m=self.config.gate_top_m,
            )
        if self.config.mixture_mode == "uniform":
            mixture_weights = torch.full_like(
                compatibility_probs,
                1.0 / self.config.num_emotions,
            )
        else:
            mixture_weights = normalize_support_mixture(
                compatibility_probs,
                top_m=(
                    self.config.gate_top_m
                    if self.config.compatibility_mode == "target_support"
                    else 0
                ),
            )
        if self.config.compatibility_mode == "target_support":
            compatibility_binary_logits = (
                compatibility_logits.float() / self.config.gate_temperature
            )
        else:
            compatibility_binary_logits = torch.logit(
                compatibility_probs.float().clamp(1e-6, 1.0 - 1e-6)
            )
        return {
            "context": context,
            "base_raw": base_raw,
            "base_prediction": base_prediction,
            "valid_mask": valid_mask,
            "lengths": lengths,
            "pooled_context": pooled_context,
            "compatibility_logits": compatibility_logits,
            "compatibility_binary_logits": compatibility_binary_logits,
            "compatibility_probs": compatibility_probs,
            "relative_gates": relative_gates,
            "mixture_weights": mixture_weights,
            "gate_warmup_alpha": context.new_tensor(gate_warmup_alpha),
        }

    def forward_compatibility(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict target support without running Mamba or residual heads."""
        state = self._anchor_state(
            speaker_audio,
            speaker_emotion,
            speaker_3dmm,
            lengths,
        )
        return {
            "compatibility_logits": state["compatibility_logits"],
            "compatibility_binary_logits": state[
                "compatibility_binary_logits"
            ],
            "compatibility_probs": state["compatibility_probs"],
            "relative_gates": state["relative_gates"],
            "mixture_weights": state["mixture_weights"],
            "valid_mask": state["valid_mask"],
        }

    def _run_query_codes(
        self,
        state: Dict[str, torch.Tensor],
        query_codes: torch.Tensor,
        query_gates: torch.Tensor,
        residual_scale: float,
        style_residual_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        if style_residual_scale is None:
            style_residual_scale = residual_scale
        if style_residual_scale < 0.0:
            raise ValueError("style_residual_scale must be non-negative")
        if query_codes.ndim != 3:
            raise ValueError("query_codes must have shape [B,Q,D]")
        if query_gates.shape != query_codes.shape[:2]:
            raise ValueError("query_gates must have shape [B,Q]")

        context = state["context"]
        valid_mask = state["valid_mask"]
        lengths = state["lengths"]
        pooled_context = state["pooled_context"]
        batch, queries, _ = query_codes.shape
        frames = context.shape[1]

        speaker_style = self.speaker_style_projection(pooled_context)
        speaker = speaker_style.unsqueeze(1).expand_as(query_codes)
        query_condition = self.query_interaction(
            torch.cat(
                (query_codes, speaker, query_codes * speaker),
                dim=-1,
            )
        )
        hidden = (
            self.context_projection(context).unsqueeze(1)
            + self.query_input_projection(query_condition).unsqueeze(2)
        )
        hidden = hidden.reshape(
            batch * queries,
            frames,
            self.config.d_model,
        )
        flat_query_condition = query_condition.reshape(
            batch * queries,
            self.config.query_dim,
        )
        flat_lengths = (
            lengths.unsqueeze(1).expand(batch, queries).reshape(-1)
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                flat_query_condition,
                flat_lengths,
            )
        dynamic_residual = (
            residual_scale
            * self.config.max_residual_logit
            * torch.tanh(self.to_residual(self.output_norm(hidden)))
        )
        dynamic_residual = dynamic_residual.view(
            batch,
            queries,
            frames,
            self.anchor.config.output_dim,
        )
        if self.static_style_adapter is None:
            style_residual = torch.zeros_like(dynamic_residual)
        else:
            style_residual = (
                style_residual_scale
                * self.config.max_style_logit
                * torch.tanh(self.static_style_adapter(query_condition))
            ).unsqueeze(2).expand(
                batch,
                queries,
                frames,
                self.anchor.config.output_dim,
            )
        query_gate = query_gates[:, :, None, None].to(dynamic_residual)
        frame_mask = valid_mask[:, None, :, None].to(dynamic_residual)
        dynamic_residual = dynamic_residual * query_gate * frame_mask
        style_residual = style_residual * query_gate * frame_mask
        residual = dynamic_residual + style_residual
        residual = (
            residual * frame_mask
        )
        query_raw = state["base_raw"].unsqueeze(1) + residual
        geometry_output = {}
        if self.to_3dmm is not None:
            geometry_output["prediction_3dmm"] = self.to_3dmm(
                self.output_norm(hidden)
            ).view(batch, queries, frames, 58) * frame_mask
        predictions = self.anchor.constrain_reaction(query_raw)
        predictions = (
            predictions
            * valid_mask[:, None, :, None].to(predictions)
        )
        return {
            "prediction": predictions,
            "raw_prediction": query_raw,
            **geometry_output,
            "residual_raw": residual,
            "dynamic_residual_raw": dynamic_residual,
            "style_residual_raw": style_residual,
            "query_condition": query_condition,
            "query_gates": query_gates,
        }

    def forward_selected(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
        emotion_indices: torch.Tensor,
        contrast_emotion_indices: Optional[torch.Tensor] = None,
        teacher_emotion_profile: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        style_residual_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train a target query, optional contrast query, and mixture query."""
        state = self._anchor_state(
            speaker_audio,
            speaker_emotion,
            speaker_3dmm,
            lengths,
        )
        batch = speaker_audio.shape[0]
        emotion_indices = torch.as_tensor(
            emotion_indices,
            device=speaker_audio.device,
            dtype=torch.long,
        )
        if emotion_indices.shape != (batch,):
            raise ValueError("emotion_indices must have shape [B]")
        if torch.any(emotion_indices < 0) or torch.any(
            emotion_indices >= self.config.num_emotions
        ):
            raise ValueError("emotion_indices are outside [0,7]")

        selected_codes = self.emotion_embeddings[emotion_indices]
        contrast_codes = None
        if contrast_emotion_indices is not None:
            contrast_emotion_indices = torch.as_tensor(
                contrast_emotion_indices,
                device=speaker_audio.device,
                dtype=torch.long,
            )
            if contrast_emotion_indices.shape != (batch,):
                raise ValueError(
                    "contrast_emotion_indices must have shape [B]"
                )
            if torch.any(contrast_emotion_indices < 0) or torch.any(
                contrast_emotion_indices >= self.config.num_emotions
            ):
                raise ValueError(
                    "contrast_emotion_indices are outside [0,7]"
                )
            if torch.any(contrast_emotion_indices == emotion_indices):
                raise ValueError(
                    "Contrast emotion must differ from selected emotion"
                )
            contrast_codes = self.emotion_embeddings[
                contrast_emotion_indices
            ]
        if (
            teacher_emotion_profile is not None
            and self.config.compatibility_mode
            == "legacy_relative_softmax"
            and self.config.mixture_mode == "predicted_support"
        ):
            if teacher_emotion_profile.shape != (
                batch,
                self.config.num_emotions,
            ):
                raise ValueError(
                    "teacher_emotion_profile must have shape [B,8]"
                )
            mixture_weights = teacher_emotion_profile.clamp_min(0.0)
            mixture_weights = mixture_weights / mixture_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        else:
            mixture_weights = state["mixture_weights"]
        mixture_codes = mixture_weights.to(selected_codes) @ (
            self.emotion_embeddings
        )
        selected_gates = state["relative_gates"].gather(
            1,
            emotion_indices.unsqueeze(1),
        )
        mixture_gates = torch.ones_like(selected_gates)
        if contrast_codes is None:
            query_codes = torch.stack(
                (selected_codes, mixture_codes),
                dim=1,
            )
            query_gates = torch.cat(
                (selected_gates, mixture_gates),
                dim=1,
            )
        else:
            contrast_gates = state["relative_gates"].gather(
                1,
                contrast_emotion_indices.unsqueeze(1),
            )
            query_codes = torch.stack(
                (selected_codes, contrast_codes, mixture_codes),
                dim=1,
            )
            query_gates = torch.cat(
                (selected_gates, contrast_gates, mixture_gates),
                dim=1,
            )
        query_output = self._run_query_codes(
            state,
            query_codes,
            query_gates,
            residual_scale,
            style_residual_scale=style_residual_scale,
        )
        return {
            **query_output,
            "base_prediction": state["base_prediction"],
            "base_raw_prediction": state["base_raw"],
            "context": state["context"],
            "valid_mask": state["valid_mask"],
            "compatibility_logits": state["compatibility_logits"],
            "compatibility_binary_logits": state[
                "compatibility_binary_logits"
            ],
            "compatibility_probs": state["compatibility_probs"],
            "mixture_weights": mixture_weights,
            "gate_warmup_alpha": state["gate_warmup_alpha"],
            "emotion_indices": emotion_indices,
            "contrast_emotion_indices": contrast_emotion_indices,
            "teacher_emotion_profile": mixture_weights,
        }

    def forward(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
        residual_scale: float = 1.0,
        style_residual_scale: Optional[float] = None,
        emotion_query_gates_override: Optional[torch.Tensor] = None,
        mixture_weights_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate q0 anchor, q1-q8 emotions, and q9 context mixture."""
        state = self._anchor_state(
            speaker_audio,
            speaker_emotion,
            speaker_3dmm,
            lengths,
        )
        batch = speaker_audio.shape[0]
        emotion_codes = self.emotion_embeddings.unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )
        if mixture_weights_override is None:
            mixture_weights = state["mixture_weights"]
        else:
            mixture_weights = torch.as_tensor(
                mixture_weights_override,
                dtype=state["mixture_weights"].dtype,
                device=state["mixture_weights"].device,
            )
            if mixture_weights.shape != (batch, self.config.num_emotions):
                raise ValueError(
                    "mixture_weights_override must have shape [B,8]"
                )
            if torch.any(mixture_weights < 0.0):
                raise ValueError("mixture_weights_override must be non-negative")
            mixture_weights = mixture_weights / mixture_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        mixture_codes = (
            mixture_weights @ self.emotion_embeddings
        ).unsqueeze(1)
        query_codes = torch.cat(
            (emotion_codes, mixture_codes),
            dim=1,
        )
        emotion_query_gates = state["relative_gates"]
        if emotion_query_gates_override is not None:
            emotion_query_gates = torch.as_tensor(
                emotion_query_gates_override,
                dtype=state["relative_gates"].dtype,
                device=state["relative_gates"].device,
            )
            if emotion_query_gates.shape != (
                batch,
                self.config.num_emotions,
            ):
                raise ValueError(
                    "emotion_query_gates_override must have shape [B,8]"
                )
            if torch.any(emotion_query_gates < 0.0) or torch.any(
                emotion_query_gates > 1.0
            ):
                raise ValueError(
                    "emotion_query_gates_override must be in [0,1]"
                )
        query_gates = torch.cat(
            (
                emotion_query_gates,
                torch.ones(
                    batch,
                    1,
                    device=speaker_audio.device,
                    dtype=state["relative_gates"].dtype,
                ),
            ),
            dim=1,
        )
        query_output = self._run_query_codes(
            state,
            query_codes,
            query_gates,
            residual_scale,
            style_residual_scale=style_residual_scale,
        )
        predictions = torch.cat(
            (
                state["base_prediction"].unsqueeze(1),
                query_output["prediction"],
            ),
            dim=1,
        )
        raw_predictions = torch.cat(
            (
                state["base_raw"].unsqueeze(1),
                query_output["raw_prediction"],
            ),
            dim=1,
        )
        return {
            **query_output,
            "prediction": predictions,
            "raw_prediction": raw_predictions,
            "base_prediction": state["base_prediction"],
            "base_raw_prediction": state["base_raw"],
            "context": state["context"],
            "valid_mask": state["valid_mask"],
            "compatibility_logits": state["compatibility_logits"],
            "compatibility_binary_logits": state[
                "compatibility_binary_logits"
            ],
            "compatibility_probs": state["compatibility_probs"],
            "emotion_query_gates": emotion_query_gates,
            "mixture_weights": mixture_weights,
            "original_mixture_weights": state["mixture_weights"],
            "gate_warmup_alpha": state["gate_warmup_alpha"],
        }


class EmotionQueryMambaLoss(nn.Module):
    """Paired appropriateness plus explicit semantic-query supervision."""

    def __init__(
        self,
        style_mean: torch.Tensor,
        style_std: torch.Tensor,
        emotion_prototypes: torch.Tensor,
        ccc_weight: float = 0.5,
        mse_weight: float = 0.25,
        velocity_weight: float = 0.05,
        emotion_weight: float = 0.25,
        query_identity_weight: float = 0.10,
        compatibility_weight: float = 0.10,
        prototype_weight: float = 0.05,
        contrast_identity_weight: float = 0.10,
        contrast_prototype_weight: float = 0.05,
        diversity_weight: float = 1.0,
        diversity_margin: float = 0.05,
        style_diversity_weight: float = 0.0,
        style_diversity_margin: float = 0.0,
        centered_diversity_weight: float = 0.0,
        mixture_weight: float = 0.25,
        residual_weight: float = 0.02,
    ):
        super().__init__()
        if style_mean.numel() != 75 or style_std.numel() != 75:
            raise ValueError("style normalization must contain 75 values")
        if emotion_prototypes.shape != (NUM_EMOTIONS, 75):
            raise ValueError("emotion_prototypes must have shape [8,75]")
        self.register_buffer("style_mean", style_mean.float().reshape(1, -1))
        self.register_buffer(
            "style_std",
            style_std.float().clamp_min(1e-4).reshape(1, -1),
        )
        self.register_buffer(
            "emotion_prototypes",
            emotion_prototypes.float(),
        )
        self.register_buffer(
            "auva_descriptor_indices",
            torch.tensor(AUVA_DESCRIPTOR_INDICES, dtype=torch.long),
        )
        self.ccc_weight = float(ccc_weight)
        self.mse_weight = float(mse_weight)
        self.velocity_weight = float(velocity_weight)
        self.emotion_weight = float(emotion_weight)
        self.query_identity_weight = float(query_identity_weight)
        self.compatibility_weight = float(compatibility_weight)
        self.prototype_weight = float(prototype_weight)
        self.contrast_identity_weight = float(contrast_identity_weight)
        self.contrast_prototype_weight = float(contrast_prototype_weight)
        self.diversity_weight = float(diversity_weight)
        self.diversity_margin = float(diversity_margin)
        self.style_diversity_weight = float(style_diversity_weight)
        self.style_diversity_margin = float(style_diversity_margin)
        self.centered_diversity_weight = float(
            centered_diversity_weight
        )
        self.mixture_weight = float(mixture_weight)
        self.residual_weight = float(residual_weight)
        if self.diversity_margin < 0.0:
            raise ValueError("diversity_margin must be non-negative")
        if self.style_diversity_margin < 0.0:
            raise ValueError("style_diversity_margin must be non-negative")

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid_mask.unsqueeze(-1).to(prediction)
        denominator = (
            weights.sum() * prediction.shape[-1]
        ).clamp_min(1.0)
        return (
            (prediction - target).square() * weights
        ).sum() / denominator

    @staticmethod
    def _masked_velocity(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape[1] <= 1:
            return prediction.new_zeros(())
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        weights = pair_mask.unsqueeze(-1).to(prediction)
        denominator = (
            weights.sum() * prediction.shape[-1]
        ).clamp_min(1.0)
        prediction_velocity = prediction[:, 1:] - prediction[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        return (
            (prediction_velocity - target_velocity).square()
            * weights
        ).sum() / denominator

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        paired_target: torch.Tensor,
        emotion_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        predictions = output["prediction"]
        valid_mask = output["valid_mask"]
        if predictions.ndim != 4 or predictions.shape[1] not in (2, 3):
            raise ValueError(
                "Selected training output must contain target/mixture or "
                "target/contrast/mixture queries"
            )
        selected_prediction = predictions[:, 0]
        has_contrast = predictions.shape[1] == 3
        contrast_prediction = predictions[:, 1] if has_contrast else None
        mixture_prediction = predictions[:, -1]

        selected_ccc = masked_ccc(
            selected_prediction,
            paired_target,
            valid_mask,
        ).mean()
        selected_mse = self._masked_mse(
            selected_prediction,
            paired_target,
            valid_mask,
        )
        velocity = self._masked_velocity(
            selected_prediction,
            paired_target,
            valid_mask,
        )
        mixture_ccc = masked_ccc(
            mixture_prediction,
            paired_target,
            valid_mask,
        ).mean()
        mixture_mse = self._masked_mse(
            mixture_prediction,
            paired_target,
            valid_mask,
        )

        target_profile = masked_emotion_profile(
            paired_target,
            valid_mask,
        )
        predicted_profile = masked_emotion_profile(
            selected_prediction,
            valid_mask,
        )
        log_predicted_profile = torch.log(
            predicted_profile.clamp_min(1e-8)
        )
        emotion_cross_entropy = -(
            target_profile * log_predicted_profile
        ).sum(dim=-1).mean()
        query_identity = F.nll_loss(
            log_predicted_profile,
            emotion_indices,
        )
        compatibility_binary_logits = output.get(
            "compatibility_binary_logits"
        )
        if compatibility_binary_logits is None:
            compatibility_binary_logits = torch.logit(
                output["compatibility_probs"].float().clamp(
                    1e-6,
                    1.0 - 1e-6,
                )
            )
        compatibility_bce = F.binary_cross_entropy_with_logits(
            compatibility_binary_logits,
            target_profile.float(),
        )

        predicted_descriptor = reaction_style_descriptor(
            selected_prediction,
            valid_mask,
        )[:, 0]
        target_prototype = self.emotion_prototypes[emotion_indices]
        normalized_difference = (
            (predicted_descriptor - target_prototype)
            / self.style_std
        )
        prototype_loss = normalized_difference.index_select(
            1,
            self.auva_descriptor_indices,
        ).square().mean()

        contrast_query_identity = predictions.new_zeros(())
        contrast_query_accuracy = predictions.new_zeros(())
        contrast_prototype_loss = predictions.new_zeros(())
        pairwise_smse = predictions.new_zeros(())
        style_pairwise_smse = predictions.new_zeros(())
        centered_pairwise_smse = predictions.new_zeros(())
        diversity_loss = predictions.new_zeros(())
        style_diversity_loss = predictions.new_zeros(())
        diversity_active_fraction = predictions.new_zeros(())
        style_diversity_active_fraction = predictions.new_zeros(())
        if has_contrast:
            contrast_emotion_indices = output.get(
                "contrast_emotion_indices"
            )
            if contrast_emotion_indices is None:
                raise ValueError(
                    "Three-query output requires contrast emotion indices"
                )
            contrast_profile = masked_emotion_profile(
                contrast_prediction,
                valid_mask,
            )
            contrast_log_profile = torch.log(
                contrast_profile.clamp_min(1e-8)
            )
            contrast_query_identity = F.nll_loss(
                contrast_log_profile,
                contrast_emotion_indices,
            )
            contrast_query_accuracy = (
                contrast_profile.argmax(dim=-1)
                == contrast_emotion_indices
            ).float().mean()
            contrast_descriptor = reaction_style_descriptor(
                contrast_prediction,
                valid_mask,
            )[:, 0]
            contrast_prototype = self.emotion_prototypes[
                contrast_emotion_indices
            ]
            contrast_normalized_difference = (
                (contrast_descriptor - contrast_prototype)
                / self.style_std
            )
            contrast_prototype_loss = (
                contrast_normalized_difference.index_select(
                    1,
                    self.auva_descriptor_indices,
                ).square().mean()
            )
            pairwise_values = masked_pairwise_smse(
                selected_prediction,
                contrast_prediction,
                valid_mask,
            )
            pairwise_smse = pairwise_values.mean()
            style_pairwise_values = masked_style_pairwise_smse(
                selected_prediction,
                contrast_prediction,
                valid_mask,
            )
            style_pairwise_smse = style_pairwise_values.mean()
            centered_pairwise_values = masked_centered_pairwise_smse(
                selected_prediction,
                contrast_prediction,
                valid_mask,
            )
            centered_pairwise_smse = centered_pairwise_values.mean()
            diversity_loss = F.relu(
                self.diversity_margin - pairwise_values
            ).mean()
            style_diversity_loss = F.relu(
                self.style_diversity_margin - style_pairwise_values
            ).mean()
            diversity_active_fraction = (
                pairwise_values < self.diversity_margin
            ).float().mean()
            style_diversity_active_fraction = (
                style_pairwise_values < self.style_diversity_margin
            ).float().mean()

        residual_l2 = output["residual_raw"].square().mean()
        mixture_reconstruction = (
            self.ccc_weight * (1.0 - mixture_ccc)
            + self.mse_weight * mixture_mse
        )
        total = (
            self.ccc_weight * (1.0 - selected_ccc)
            + self.mse_weight * selected_mse
            + self.velocity_weight * velocity
            + self.emotion_weight * emotion_cross_entropy
            + self.query_identity_weight * query_identity
            + self.compatibility_weight * compatibility_bce
            + self.prototype_weight * prototype_loss
            + self.contrast_identity_weight * contrast_query_identity
            + self.contrast_prototype_weight * contrast_prototype_loss
            + self.diversity_weight * diversity_loss
            + self.style_diversity_weight * style_diversity_loss
            + self.centered_diversity_weight * centered_pairwise_smse
            + self.mixture_weight * mixture_reconstruction
            + self.residual_weight * residual_l2
        )
        predicted_labels = predicted_profile.argmax(dim=-1)
        compatibility_labels = output[
            "compatibility_probs"
        ].argmax(dim=-1)
        return {
            "loss": total,
            "selected_ccc": selected_ccc,
            "selected_mse": selected_mse,
            "velocity": velocity,
            "emotion_ce": emotion_cross_entropy,
            "query_identity": query_identity,
            "query_accuracy": (
                predicted_labels == emotion_indices
            ).float().mean(),
            "compatibility_ce": compatibility_bce,
            "compatibility_bce": compatibility_bce,
            "compatibility_accuracy": (
                compatibility_labels == emotion_indices
            ).float().mean(),
            "prototype_loss": prototype_loss,
            "contrast_query_identity": contrast_query_identity,
            "contrast_query_accuracy": contrast_query_accuracy,
            "contrast_prototype_loss": contrast_prototype_loss,
            "pairwise_smse": pairwise_smse,
            "style_pairwise_smse": style_pairwise_smse,
            "centered_pairwise_smse": centered_pairwise_smse,
            "diversity_loss": diversity_loss,
            "style_diversity_loss": style_diversity_loss,
            "diversity_active_fraction": diversity_active_fraction,
            "style_diversity_active_fraction": (
                style_diversity_active_fraction
            ),
            "mixture_ccc": mixture_ccc,
            "mixture_mse": mixture_mse,
            "residual_l2": residual_l2,
            "selected_gate": output["query_gates"][:, 0].mean(),
        }


class EmotionQuerySessionSetLoss(nn.Module):
    """Match the nine learned queries to a real session reaction set.

    The set loss is permutation invariant, while the semantic term preserves
    the E0-E7 identity of the eight named emotion queries. All ten candidates
    retain paired CCC/MSE/velocity supervision through the base set loss.
    """

    def __init__(
        self,
        style_mean: torch.Tensor,
        style_std: torch.Tensor,
        style_weight: float = 1.0,
        paired_ccc_weight: float = 0.5,
        paired_mse_weight: float = 0.25,
        velocity_weight: float = 0.05,
        diversity_weight: float = 0.0,
        diversity_margin: float = 0.0,
        residual_weight: float = 0.02,
        semantic_identity_weight: float = 0.10,
        sinkhorn_temperature: float = 0.1,
        sinkhorn_iterations: int = 8,
    ):
        super().__init__()
        self.set_loss = QueryResidualMambaLoss(
            style_mean=style_mean,
            style_std=style_std,
            style_weight=style_weight,
            paired_ccc_weight=paired_ccc_weight,
            paired_mse_weight=paired_mse_weight,
            velocity_weight=velocity_weight,
            diversity_weight=diversity_weight,
            diversity_margin=diversity_margin,
            residual_weight=residual_weight,
            sinkhorn_temperature=sinkhorn_temperature,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        self.semantic_identity_weight = float(semantic_identity_weight)

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        paired_target: torch.Tensor,
        target_styles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        predictions = output["prediction"]
        valid_mask = output["valid_mask"]
        if predictions.ndim != 4 or predictions.shape[1] != 10:
            raise ValueError(
                "Session-set output must contain anchor + nine queries"
            )
        base_losses = self.set_loss(output, paired_target, target_styles)

        batch, _, frames, channels = predictions.shape
        emotion_predictions = predictions[:, 1:9].reshape(
            batch * NUM_EMOTIONS,
            frames,
            channels,
        )
        emotion_mask = valid_mask[:, None].expand(
            batch,
            NUM_EMOTIONS,
            frames,
        ).reshape(batch * NUM_EMOTIONS, frames)
        emotion_profiles = masked_emotion_profile(
            emotion_predictions,
            emotion_mask,
        )
        semantic_targets = torch.arange(
            NUM_EMOTIONS,
            device=predictions.device,
        ).unsqueeze(0).expand(batch, -1).reshape(-1)
        semantic_identity = F.nll_loss(
            torch.log(emotion_profiles.clamp_min(1e-8)),
            semantic_targets,
        )
        semantic_accuracy = (
            emotion_profiles.argmax(dim=-1) == semantic_targets
        ).float().mean()
        total = (
            base_losses["loss"]
            + self.semantic_identity_weight * semantic_identity
        )
        return {
            **base_losses,
            "loss": total,
            "semantic_identity": semantic_identity,
            "semantic_accuracy": semantic_accuracy,
        }


class EmotionQuerySequenceSetLoss(nn.Module):
    """Paired anchor timing plus alignment-tolerant set supervision."""

    def __init__(
        self,
        style_mean: torch.Tensor,
        style_std: torch.Tensor,
        anchor_ccc_weight: float = 10.0,
        learned_set_ccc_weight: float = 0.0,
        mse_weight: float = 0.25,
        velocity_weight: float = 0.05,
        style_weight: float = 0.5,
        semantic_identity_weight: float = 0.10,
        semantic_identity_mode: str = "support_weighted",
        semantic_identity_support_power: float = 0.5,
        semantic_identity_min_support: float = 0.0,
        compatibility_weight: float = 0.10,
        mixture_weight: float = 0.25,
        sequence_transport_weight: float = 0.0,
        sequence_transport_temperature: float = 0.05,
        sequence_distance_mode: str = "banded_softdtw",
        set_matching_strategy: str = "balanced",
        style_diversity_weight: float = 0.0,
        style_diversity_margin: float = 0.0,
        official_diversity_weight: float = 0.0,
        official_diversity_margin: float = 0.0,
        official_diversity_batch_weight: float = 0.0,
        channel_diversity_weight: float = 0.0,
        channel_diversity_budget: float = 0.15,
        channel_diversity_no_cap: bool = False,
        frd_surrogate_weight: float = 0.0,
        frd_surrogate_target: float = 160.0,
        frd_surrogate_stride: int = 12,
        frd_surrogate_calibration: float = 1.0,
        frd_surrogate_mode: str = "diagonal",
        frd_softdtw_stride: int = 6,
        frd_softdtw_band_ratio: float = 0.15,
        frd_softdtw_gamma: float = 0.1,
        frd_softdtw_pair_chunk_size: int = 256,
        frd_tail_weight: float = 0.0,
        frd_tail_fraction: float = 0.25,
        frd_query_excess_weight: float = 0.0,
        frd_query_excess_fraction: float = 0.5,
        frd_transport_weight: float = 0.0,
        frd_transport_temperature: float = 0.1,
        frd_transport_iterations: int = 32,
        sync_weight: float = 0.0,
        sync_max_lag: int = 48,
        sync_lag_step: int = 12,
        sync_temporal_stride: int = 4,
        sync_temperature: float = 0.05,
        sync_variance_epsilon: float = 1e-6,
        residual_weight: float = 0.02,
        sinkhorn_temperature: float = 0.1,
        sinkhorn_iterations: int = 8,
    ):
        super().__init__()
        if style_mean.numel() != 75 or style_std.numel() != 75:
            raise ValueError("style normalization must contain 75 values")
        if anchor_ccc_weight < 0.0 or learned_set_ccc_weight < 0.0:
            raise ValueError("CCC weights must be non-negative")
        if semantic_identity_mode not in {"support_weighted", "uniform"}:
            raise ValueError("Unknown semantic_identity_mode")
        if semantic_identity_support_power <= 0.0:
            raise ValueError(
                "semantic_identity_support_power must be positive"
            )
        if not 0.0 <= semantic_identity_min_support <= 1.0:
            raise ValueError(
                "semantic_identity_min_support must be in [0,1]"
            )
        if style_diversity_margin < 0.0:
            raise ValueError("style_diversity_margin must be non-negative")
        if official_diversity_margin < 0.0:
            raise ValueError("official_diversity_margin must be non-negative")
        if official_diversity_batch_weight < 0.0:
            raise ValueError(
                "official_diversity_batch_weight must be non-negative"
            )
        if channel_diversity_weight < 0.0:
            raise ValueError("channel_diversity_weight must be non-negative")
        if channel_diversity_budget <= 0.0:
            raise ValueError("channel_diversity_budget must be positive")
        if frd_surrogate_weight < 0.0:
            raise ValueError("frd_surrogate_weight must be non-negative")
        if frd_surrogate_target <= 0.0:
            raise ValueError("frd_surrogate_target must be positive")
        if frd_surrogate_stride <= 0:
            raise ValueError("frd_surrogate_stride must be positive")
        if frd_surrogate_calibration <= 0.0:
            raise ValueError("frd_surrogate_calibration must be positive")
        if frd_surrogate_mode not in {"diagonal", "banded_softdtw"}:
            raise ValueError("Unknown frd_surrogate_mode")
        if frd_softdtw_stride <= 0:
            raise ValueError("frd_softdtw_stride must be positive")
        if not 0.0 <= frd_softdtw_band_ratio <= 1.0:
            raise ValueError("frd_softdtw_band_ratio must be in [0,1]")
        if frd_softdtw_gamma <= 0.0:
            raise ValueError("frd_softdtw_gamma must be positive")
        if frd_softdtw_pair_chunk_size <= 0:
            raise ValueError("frd_softdtw_pair_chunk_size must be positive")
        if frd_tail_weight < 0.0:
            raise ValueError("frd_tail_weight must be non-negative")
        if not 0.0 < frd_tail_fraction <= 1.0:
            raise ValueError("frd_tail_fraction must be in (0, 1]")
        if frd_query_excess_weight < 0.0:
            raise ValueError("frd_query_excess_weight must be non-negative")
        if not 0.0 < frd_query_excess_fraction <= 1.0:
            raise ValueError(
                "frd_query_excess_fraction must be in (0, 1]"
            )
        if frd_transport_weight < 0.0:
            raise ValueError("frd_transport_weight must be non-negative")
        if frd_transport_temperature <= 0.0:
            raise ValueError("frd_transport_temperature must be positive")
        if frd_transport_iterations <= 0:
            raise ValueError("frd_transport_iterations must be positive")
        if sync_weight < 0.0:
            raise ValueError("sync_weight must be non-negative")
        if sync_max_lag <= 0:
            raise ValueError("sync_max_lag must be positive")
        if sync_lag_step <= 0 or sync_max_lag % sync_lag_step != 0:
            raise ValueError(
                "sync_lag_step must be positive and divide sync_max_lag"
            )
        if sync_temporal_stride <= 0:
            raise ValueError("sync_temporal_stride must be positive")
        if sync_temperature <= 0.0:
            raise ValueError("sync_temperature must be positive")
        if sync_variance_epsilon < 0.0:
            raise ValueError(
                "sync_variance_epsilon must be non-negative"
            )
        if sequence_transport_temperature <= 0.0:
            raise ValueError(
                "sequence_transport_temperature must be positive"
            )
        if sequence_distance_mode not in {"ccc", "banded_softdtw"}:
            raise ValueError("Unknown sequence_distance_mode")
        if set_matching_strategy not in {
            "balanced", "nearest", "paired", "row_softmax", "hard_hungarian"
        }:
            raise ValueError(
                "set_matching_strategy must be balanced, nearest, or paired"
            )
        self.register_buffer(
            "style_mean",
            style_mean.float().reshape(1, 1, -1),
        )
        self.register_buffer(
            "style_std",
            style_std.float().clamp_min(1e-4).reshape(1, 1, -1),
        )
        self.anchor_ccc_weight = float(anchor_ccc_weight)
        self.learned_set_ccc_weight = float(learned_set_ccc_weight)
        self.mse_weight = float(mse_weight)
        self.velocity_weight = float(velocity_weight)
        self.style_weight = float(style_weight)
        self.semantic_identity_weight = float(semantic_identity_weight)
        self.semantic_identity_mode = semantic_identity_mode
        self.semantic_identity_support_power = float(
            semantic_identity_support_power
        )
        self.semantic_identity_min_support = float(
            semantic_identity_min_support
        )
        self.compatibility_weight = float(compatibility_weight)
        self.mixture_weight = float(mixture_weight)
        self.sequence_transport_weight = float(sequence_transport_weight)
        self.sequence_transport_temperature = float(
            sequence_transport_temperature
        )
        self.sequence_distance_mode = sequence_distance_mode
        self.set_matching_strategy = set_matching_strategy
        self.style_diversity_weight = float(style_diversity_weight)
        self.style_diversity_margin = float(style_diversity_margin)
        self.official_diversity_weight = float(official_diversity_weight)
        self.official_diversity_margin = float(official_diversity_margin)
        self.official_diversity_batch_weight = float(
            official_diversity_batch_weight
        )
        self.channel_diversity_weight = float(channel_diversity_weight)
        self.channel_diversity_budget = float(channel_diversity_budget)
        self.channel_diversity_no_cap = bool(channel_diversity_no_cap)
        self.frd_surrogate_weight = float(frd_surrogate_weight)
        self.frd_surrogate_target = float(frd_surrogate_target)
        self.frd_surrogate_stride = int(frd_surrogate_stride)
        self.frd_surrogate_calibration = float(
            frd_surrogate_calibration
        )
        self.frd_surrogate_mode = frd_surrogate_mode
        self.frd_softdtw = BandedSoftDTWFRDSurrogate(
            temporal_stride=frd_softdtw_stride,
            band_ratio=frd_softdtw_band_ratio,
            gamma=frd_softdtw_gamma,
            pair_chunk_size=frd_softdtw_pair_chunk_size,
            calibration=frd_surrogate_calibration,
        )
        self.frd_tail_weight = float(frd_tail_weight)
        self.frd_tail_fraction = float(frd_tail_fraction)
        self.frd_query_excess_weight = float(frd_query_excess_weight)
        self.frd_query_excess_fraction = float(
            frd_query_excess_fraction
        )
        self.frd_transport_weight = float(frd_transport_weight)
        self.frd_transport_temperature = float(
            frd_transport_temperature
        )
        self.frd_transport_iterations = int(frd_transport_iterations)
        self.sync_weight = float(sync_weight)
        self.sync_max_lag = int(sync_max_lag)
        self.sync_lag_step = int(sync_lag_step)
        self.sync_temporal_stride = int(sync_temporal_stride)
        self.sync_temperature = float(sync_temperature)
        self.sync_variance_epsilon = float(sync_variance_epsilon)
        self.residual_weight = float(residual_weight)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

    def _set_matching_cost(
        self,
        cost: torch.Tensor,
        valid_targets: torch.Tensor,
        *,
        temperature: float,
        iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.set_matching_strategy == "nearest":
            return masked_nearest_cost(cost, valid_targets)
        if self.set_matching_strategy == "row_softmax":
            return masked_row_softmax_cost(cost, valid_targets, temperature)
        if self.set_matching_strategy == "hard_hungarian":
            return masked_hungarian_cost(cost, valid_targets)
        return masked_sinkhorn_cost(
            cost,
            valid_targets,
            temperature=temperature,
            iterations=iterations,
        )

    def _ccc_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, frames, channels = predictions.shape
        target_count = targets.shape[1]
        flat_targets = targets.reshape(
            batch * target_count,
            frames,
            channels,
        )
        flat_mask = pair_mask.reshape(batch * target_count, frames)
        rows = []
        for query_index in range(queries):
            flat_prediction = (
                predictions[:, query_index]
                .unsqueeze(1)
                .expand(batch, target_count, frames, channels)
                .reshape(batch * target_count, frames, channels)
            )
            rows.append(
                masked_ccc(
                    flat_prediction,
                    flat_targets,
                    flat_mask,
                ).mean(dim=-1).reshape(batch, target_count)
            )
        return torch.stack(rows, dim=1)

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_valid_mask: torch.Tensor,
        speaker_emotion: Optional[torch.Tensor] = None,
        compute_sync: bool = True,
        sync_loss_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        predictions = output["prediction"]
        valid_mask = output["valid_mask"]
        if predictions.ndim != 4 or predictions.shape[1] != 10:
            raise ValueError("Sequence-set predictions must be [B,10,T,25]")
        if (
            targets.ndim != 4
            or targets.shape[0] != predictions.shape[0]
            or targets.shape[2:] != predictions.shape[2:]
        ):
            raise ValueError("targets must have shape [B,K,T,25]")
        if target_valid_mask.shape != targets.shape[:3]:
            raise ValueError("target_valid_mask must have shape [B,K,T]")
        if sync_loss_scale < 0.0:
            raise ValueError("sync_loss_scale must be non-negative")

        batch, queries, frames, channels = predictions.shape
        target_count = targets.shape[1]
        effective_target_valid_mask = target_valid_mask
        if self.set_matching_strategy == "paired":
            effective_target_valid_mask = torch.zeros_like(target_valid_mask)
            effective_target_valid_mask[:, 0] = target_valid_mask[:, 0]
        pair_mask = effective_target_valid_mask & valid_mask[:, None]
        valid_targets = effective_target_valid_mask.any(dim=-1)
        # Same-position CCC over independently cropped non-paired reactions is
        # diagnostic by default. It enters autograd only for the explicit
        # learned_set_ccc_weight ablation.
        ccc_predictions = (
            predictions
            if self.learned_set_ccc_weight > 0.0
            else predictions.detach()
        )
        ccc_matrix = self._ccc_matrix(ccc_predictions, targets, pair_mask)
        ccc_matrix = ccc_matrix.masked_fill(
            ~valid_targets[:, None],
            -1e4,
        )
        # q0 is the deterministic anchor and retains strict paired temporal
        # diagnostics. Learned-query max CCC is not a training objective in
        # the final configuration because non-paired clips are not aligned.
        matching_ccc = ccc_matrix.clone()
        matching_ccc[:, 0, 1:] = -1e4
        best_ccc, best_indices = matching_ccc.max(dim=-1)
        query_ccc_valid = valid_targets.any(dim=-1, keepdim=True).expand(
            -1,
            queries,
        ).clone()
        query_ccc_valid[:, 0] = valid_targets[:, 0]
        query_ccc_weights = query_ccc_valid.to(best_ccc)
        set_max_ccc = (
            best_ccc * query_ccc_weights
        ).sum() / query_ccc_weights.sum().clamp_min(1.0)
        paired_weights = query_ccc_weights[:, 0]
        paired_anchor_ccc = (
            best_ccc[:, 0] * paired_weights
        ).sum() / paired_weights.sum().clamp_min(1.0)
        learned_weights = query_ccc_weights[:, 1:]
        learned_set_max_ccc = (
            best_ccc[:, 1:] * learned_weights
        ).sum() / learned_weights.sum().clamp_min(1.0)
        alignment_softdtw_output = None
        if (
            self.sequence_distance_mode == "banded_softdtw"
            and self.sequence_transport_weight > 0.0
        ):
            alignment_softdtw_output = self.frd_softdtw(
                predictions,
                targets,
                valid_mask,
                effective_target_valid_mask,
            )
            per_query_frd_target = self.frd_surrogate_target / queries
            sequence_pair_cost = (
                alignment_softdtw_output["pairwise_cost"]
                / per_query_frd_target
            )
            sequence_transport_loss, sequence_transport = (
                self._set_matching_cost(
                    sequence_pair_cost,
                    valid_targets,
                    temperature=self.sequence_transport_temperature,
                    iterations=self.sinkhorn_iterations,
                )
            )
            sequence_transport_ccc = predictions.new_zeros(())
        else:
            sequence_transport_loss, sequence_transport = (
                self._set_matching_cost(
                    1.0 - ccc_matrix,
                    valid_targets,
                    temperature=self.sequence_transport_temperature,
                    iterations=self.sinkhorn_iterations,
                )
            )
            sequence_transport_ccc = 1.0 - sequence_transport_loss

        paired_target = targets[:, 0]
        paired_mask = pair_mask[:, 0]
        weights = paired_mask.unsqueeze(-1).to(predictions)
        mse_denominator = (
            weights.sum() * channels
        ).clamp_min(1.0)
        matched_mse = (
            (predictions[:, 0] - paired_target).square() * weights
        ).sum() / mse_denominator

        if frames > 1:
            velocity_mask = paired_mask[:, 1:] & paired_mask[:, :-1]
            velocity_weights = velocity_mask.unsqueeze(-1).to(predictions)
            velocity_denominator = (
                velocity_weights.sum() * channels
            ).clamp_min(1.0)
            prediction_velocity = (
                predictions[:, 0, 1:] - predictions[:, 0, :-1]
            )
            target_velocity = paired_target[:, 1:] - paired_target[:, :-1]
            matched_velocity = (
                (prediction_velocity - target_velocity).square()
                * velocity_weights
            ).sum() / velocity_denominator
        else:
            matched_velocity = predictions.new_zeros(())

        predicted_styles = reaction_style_descriptor(
            predictions,
            valid_mask,
        )
        flat_targets = targets.reshape(
            batch * target_count,
            frames,
            channels,
        )
        flat_target_mask = effective_target_valid_mask.reshape(
            batch * target_count,
            frames,
        )
        target_styles = reaction_style_descriptor(
            flat_targets,
            flat_target_mask,
        )[:, 0].reshape(batch, target_count, -1)
        normalized_predictions = (
            predicted_styles - self.style_mean
        ) / self.style_std
        normalized_targets = (
            target_styles - self.style_mean
        ) / self.style_std
        style_cost = (
            normalized_predictions.unsqueeze(2)
            - normalized_targets.unsqueeze(1)
        ).square().mean(dim=-1)
        style_ot, transport = self._set_matching_cost(
            style_cost,
            valid_targets,
            temperature=self.sinkhorn_temperature,
            iterations=self.sinkhorn_iterations,
        )

        target_support = target_set_emotion_support(
            targets,
            effective_target_valid_mask,
        )
        emotion_predictions = predictions[:, 1:9].reshape(
            batch * NUM_EMOTIONS,
            frames,
            channels,
        )
        emotion_mask = valid_mask[:, None].expand(
            batch,
            NUM_EMOTIONS,
            frames,
        ).reshape(batch * NUM_EMOTIONS, frames)
        emotion_profiles = masked_emotion_profile(
            emotion_predictions,
            emotion_mask,
        ).reshape(batch, NUM_EMOTIONS, NUM_EMOTIONS)
        semantic_metrics = semantic_identity_from_support(
            emotion_profiles,
            target_support,
            mode=self.semantic_identity_mode,
            support_power=self.semantic_identity_support_power,
            min_support=self.semantic_identity_min_support,
        )
        semantic_identity = semantic_metrics["loss"]
        semantic_accuracy = semantic_metrics["accuracy"]
        semantic_weighted_accuracy = semantic_metrics["weighted_accuracy"]
        semantic_identity_mean_weight = semantic_metrics["mean_weight"]
        predicted_support = output["compatibility_probs"]
        if predicted_support.shape != target_support.shape:
            raise ValueError("compatibility_probs must have shape [B,8]")
        compatibility_binary_logits = output.get(
            "compatibility_binary_logits"
        )
        if compatibility_binary_logits is None:
            compatibility_binary_logits = torch.logit(
                predicted_support.float().clamp(1e-6, 1.0 - 1e-6)
            )
        compatibility_per_example = F.binary_cross_entropy_with_logits(
            compatibility_binary_logits,
            target_support.float(),
            reduction="none",
        ).mean(dim=-1)
        support_example_weights = valid_targets.any(dim=-1).to(predictions)
        compatibility_bce = (
            compatibility_per_example * support_example_weights
        ).sum() / support_example_weights.sum().clamp_min(1.0)
        mixture_profile = masked_emotion_profile(
            predictions[:, 9],
            valid_mask,
        )
        normalized_target_support = (
            target_support
            / target_support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        mixture_per_example = -(
            normalized_target_support
            * torch.log(mixture_profile.clamp_min(1e-8))
        ).sum(dim=-1)
        mixture_query_consistency = (
            mixture_per_example * support_example_weights
        ).sum() / support_example_weights.sum().clamp_min(1.0)
        compatibility_mean = predicted_support.mean()
        compatibility_min = predicted_support.amin()
        compatibility_max = predicted_support.amax()
        active_query_count_025 = (
            predicted_support >= 0.25
        ).float().sum(dim=-1).mean()
        active_query_count_050 = (
            predicted_support >= 0.50
        ).float().sum(dim=-1).mean()

        style_pairwise_smse, style_pairwise_values = (
            masked_all_query_style_pairwise_smse(
                predictions,
                valid_mask,
            )
        )
        per_example_style = style_pairwise_values.mean(dim=-1)
        style_diversity_loss = F.relu(
            self.style_diversity_margin - per_example_style
        ).mean()
        style_diversity_active_fraction = (
            per_example_style < self.style_diversity_margin
        ).float().mean()
        surrogate_official_frdiv, per_example_official_frdiv = (
            masked_all_query_official_frdiv(
                predictions,
                valid_mask,
            )
        )
        official_diversity_loss = F.relu(
            self.official_diversity_margin
            - per_example_official_frdiv
        ).mean()
        official_diversity_active_fraction = (
            per_example_official_frdiv
            < self.official_diversity_margin
        ).float().mean()
        official_diversity_batch_loss = F.relu(
            predictions.new_tensor(self.official_diversity_margin)
            - surrogate_official_frdiv
        )
        if self.channel_diversity_weight > 0.0:
            (
                prediction_frdiv_groups,
                prediction_frdiv_groups_per_example,
            ) = (
                masked_all_query_official_frdiv_groups(
                    predictions,
                    valid_mask,
                )
            )
            (
                _,
                target_frdiv_groups_per_example,
                target_group_valid,
            ) = masked_target_set_official_frdiv_groups(
                targets,
                effective_target_valid_mask,
            )
            target_frdiv_total = target_frdiv_groups_per_example.sum(dim=-1)
            channel_valid = (
                target_group_valid
                & (target_frdiv_total > 1e-8)
            )
            target_frdiv_share = (
                target_frdiv_groups_per_example
                / target_frdiv_total[:, None].clamp_min(1e-8)
            ).detach()
            if self.channel_diversity_no_cap:
                desired_total = target_frdiv_total.detach()
            else:
                desired_total = torch.minimum(
                    target_frdiv_total.detach(),
                    target_frdiv_total.new_full(
                        target_frdiv_total.shape,
                        self.channel_diversity_budget,
                    ),
                )
            desired_frdiv_groups_per_example = (
                target_frdiv_share * desired_total[:, None]
            )
            normalized_channel_error = F.smooth_l1_loss(
                prediction_frdiv_groups_per_example
                / self.channel_diversity_budget,
                desired_frdiv_groups_per_example
                / self.channel_diversity_budget,
                reduction="none",
            ).mean(dim=-1)
            channel_valid_weights = channel_valid.to(predictions)
            channel_diversity_loss = (
                normalized_channel_error * channel_valid_weights
            ).sum() / channel_valid_weights.sum().clamp_min(1.0)
            valid_denominator = channel_valid_weights.sum().clamp_min(1.0)
            target_frdiv_groups = (
                target_frdiv_groups_per_example
                * channel_valid_weights[:, None]
            ).sum(dim=0) / valid_denominator
            desired_frdiv_groups = (
                desired_frdiv_groups_per_example
                * channel_valid_weights[:, None]
            ).sum(dim=0) / valid_denominator
            channel_diversity_valid_fraction = channel_valid_weights.mean()
        else:
            prediction_frdiv_groups = predictions.new_zeros(3)
            target_frdiv_groups = predictions.new_zeros(3)
            desired_frdiv_groups = predictions.new_zeros(3)
            channel_diversity_loss = predictions.new_zeros(())
            channel_diversity_valid_fraction = predictions.new_zeros(())
        if (
            self.frd_surrogate_weight > 0.0
            or self.frd_tail_weight > 0.0
            or self.frd_query_excess_weight > 0.0
            or self.frd_transport_weight > 0.0
        ):
            if self.frd_surrogate_mode == "banded_softdtw":
                softdtw_output = alignment_softdtw_output
                if softdtw_output is None:
                    softdtw_output = self.frd_softdtw(
                        predictions,
                        targets,
                        valid_mask,
                        effective_target_valid_mask,
                    )
                surrogate_official_frd = softdtw_output["mean"]
                per_example_frd = softdtw_output["per_example"]
                frd_pair_cost = softdtw_output["pairwise_cost"]
            else:
                (
                    surrogate_official_frd,
                    per_example_frd,
                    frd_pair_cost,
                ) = masked_official_frd_surrogate(
                    predictions,
                    targets,
                    valid_mask,
                    effective_target_valid_mask,
                    temporal_stride=self.frd_surrogate_stride,
                    calibration=self.frd_surrogate_calibration,
                )
            valid_frd_targets = effective_target_valid_mask.any(dim=-1)
            valid_frd_examples = valid_frd_targets.any(dim=-1)
            frd_example_weights = valid_frd_examples.to(predictions)
            frd_example_denominator = frd_example_weights.sum().clamp_min(1.0)
            per_example_frd_hinge = F.relu(
                per_example_frd / self.frd_surrogate_target - 1.0
            )
            frd_surrogate_loss = (
                per_example_frd_hinge * frd_example_weights
            ).sum() / frd_example_denominator
            frd_surrogate_active_fraction = (
                per_example_frd > self.frd_surrogate_target
            ).to(predictions).mul(frd_example_weights).sum() / (
                frd_example_denominator
            )
            valid_per_example_frd = per_example_frd[valid_frd_examples]
            if self.frd_tail_weight > 0.0 and valid_per_example_frd.numel() > 0:
                tail_count = max(
                    1,
                    math.ceil(
                        valid_per_example_frd.numel()
                        * self.frd_tail_fraction
                    ),
                )
                tail_frd = torch.topk(
                    valid_per_example_frd,
                    k=tail_count,
                    largest=True,
                    sorted=False,
                ).values
                surrogate_official_frd_tail = tail_frd.mean()
                frd_tail_loss = F.relu(
                    tail_frd / self.frd_surrogate_target - 1.0
                ).mean()
                frd_tail_active_fraction = (
                    tail_frd > self.frd_surrogate_target
                ).float().mean()
            else:
                surrogate_official_frd_tail = predictions.new_zeros(())
                frd_tail_loss = predictions.new_zeros(())
                frd_tail_active_fraction = predictions.new_zeros(())
            best_query_frd = frd_pair_cost.min(dim=-1).values
            best_query_frd = torch.where(
                valid_frd_examples[:, None],
                best_query_frd,
                torch.zeros_like(best_query_frd),
            )
            query_excess = best_query_frd - best_query_frd[:, :1].detach()
            learned_query_excess = query_excess[:, 1:]
            mean_best_target_distance = (
                best_query_frd * frd_example_weights[:, None]
            ).sum(dim=0) / frd_example_denominator
            mean_query_excess = (
                query_excess * frd_example_weights[:, None]
            ).sum(dim=0) / frd_example_denominator
            worst_query_excess = (
                learned_query_excess.amax(dim=1) * frd_example_weights
            ).sum() / frd_example_denominator
            learned_queries_better_than_anchor_fraction = (
                learned_query_excess < 0.0
            ).to(predictions).mul(
                frd_example_weights[:, None]
            ).sum() / (
                frd_example_denominator * (queries - 1)
            )
            if self.frd_query_excess_weight > 0.0:
                query_tail_count = max(
                    1,
                    math.ceil(
                        (queries - 1) * self.frd_query_excess_fraction
                    ),
                )
                query_tail_excess = torch.topk(
                    learned_query_excess,
                    k=query_tail_count,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).values
                surrogate_official_frd_query_excess = (
                    learned_query_excess.mean(dim=1)
                    * frd_example_weights
                ).sum() / frd_example_denominator
                surrogate_official_frd_query_tail_excess = (
                    query_tail_excess.mean(dim=1)
                    * frd_example_weights
                ).sum() / frd_example_denominator
                frd_query_excess_loss = (
                    F.relu(query_tail_excess).mean(dim=1)
                    * frd_example_weights
                ).sum() / frd_example_denominator
                frd_query_excess_active_fraction = (
                    query_tail_excess > 0.0
                ).to(predictions).mean(dim=1).mul(
                    frd_example_weights
                ).sum() / frd_example_denominator
            else:
                surrogate_official_frd_query_excess = (
                    predictions.new_zeros(())
                )
                surrogate_official_frd_query_tail_excess = (
                    predictions.new_zeros(())
                )
                frd_query_excess_loss = predictions.new_zeros(())
                frd_query_excess_active_fraction = predictions.new_zeros(())
            frd_best_indices = frd_pair_cost.argmin(dim=-1)
            coverage_values = []
            for index in range(batch):
                valid_count = int(valid_frd_targets[index].sum())
                if valid_count == 0:
                    continue
                matched = frd_best_indices[index].unique()
                matched = matched[valid_frd_targets[index, matched]]
                coverage_values.append(matched.numel() / valid_count)
            frd_match_coverage = (
                predictions.new_tensor(coverage_values).mean()
                if coverage_values
                else predictions.new_zeros(())
            )
            if self.frd_transport_weight > 0.0:
                per_query_target = self.frd_surrogate_target / queries
                _, frd_transport = self._set_matching_cost(
                    frd_pair_cost / per_query_target,
                    valid_frd_targets,
                    temperature=self.frd_transport_temperature,
                    iterations=self.frd_transport_iterations,
                )
                safe_frd_pair_cost = frd_pair_cost.masked_fill(
                    ~valid_frd_targets[:, None],
                    0.0,
                )
                per_example_frd_transport = queries * (
                    frd_transport * safe_frd_pair_cost
                ).sum(dim=(1, 2))
                surrogate_official_frd_transport = (
                    per_example_frd_transport * frd_example_weights
                ).sum() / frd_example_denominator
                per_example_transport_hinge = F.relu(
                    per_example_frd_transport
                    / self.frd_surrogate_target
                    - 1.0
                )
                frd_transport_loss = (
                    per_example_transport_hinge * frd_example_weights
                ).sum() / frd_example_denominator
                frd_transport_active_fraction = (
                    per_example_frd_transport
                    > self.frd_surrogate_target
                ).to(predictions).mul(frd_example_weights).sum() / (
                    frd_example_denominator
                )
                frd_transport_best_indices = frd_transport.argmax(dim=-1)
                transport_coverage_values = []
                for index in range(batch):
                    valid_count = int(valid_frd_targets[index].sum())
                    if valid_count == 0:
                        continue
                    matched = frd_transport_best_indices[index].unique()
                    matched = matched[valid_frd_targets[index, matched]]
                    transport_coverage_values.append(
                        matched.numel() / valid_count
                    )
                frd_transport_coverage = (
                    predictions.new_tensor(transport_coverage_values).mean()
                    if transport_coverage_values
                    else predictions.new_zeros(())
                )
                per_example_transport_entropy = -(
                    frd_transport
                    * torch.log(frd_transport.clamp_min(1e-8))
                ).sum(dim=(1, 2))
                frd_transport_entropy = (
                    per_example_transport_entropy * frd_example_weights
                ).sum() / frd_example_denominator
            else:
                surrogate_official_frd_transport = predictions.new_zeros(())
                frd_transport_loss = predictions.new_zeros(())
                frd_transport_active_fraction = predictions.new_zeros(())
                frd_transport_coverage = predictions.new_zeros(())
                frd_transport_entropy = predictions.new_zeros(())
        else:
            surrogate_official_frd = predictions.new_zeros(())
            frd_surrogate_loss = predictions.new_zeros(())
            frd_surrogate_active_fraction = predictions.new_zeros(())
            surrogate_official_frd_tail = predictions.new_zeros(())
            frd_tail_loss = predictions.new_zeros(())
            frd_tail_active_fraction = predictions.new_zeros(())
            surrogate_official_frd_query_excess = predictions.new_zeros(())
            surrogate_official_frd_query_tail_excess = (
                predictions.new_zeros(())
            )
            frd_query_excess_loss = predictions.new_zeros(())
            frd_query_excess_active_fraction = predictions.new_zeros(())
            frd_match_coverage = predictions.new_zeros(())
            mean_best_target_distance = predictions.new_zeros(queries)
            mean_query_excess = predictions.new_zeros(queries)
            worst_query_excess = predictions.new_zeros(())
            learned_queries_better_than_anchor_fraction = (
                predictions.new_zeros(())
            )
            surrogate_official_frd_transport = predictions.new_zeros(())
            frd_transport_loss = predictions.new_zeros(())
            frd_transport_active_fraction = predictions.new_zeros(())
            frd_transport_coverage = predictions.new_zeros(())
            frd_transport_entropy = predictions.new_zeros(())
        if self.sync_weight > 0.0 and compute_sync:
            if speaker_emotion is None:
                raise ValueError(
                    "speaker_emotion is required when sync_weight is active"
                )
            sync_metrics = masked_valid_channel_soft_tlcc(
                predictions,
                speaker_emotion,
                valid_mask,
                max_lag=self.sync_max_lag,
                lag_step=self.sync_lag_step,
                temporal_stride=self.sync_temporal_stride,
                temperature=self.sync_temperature,
                variance_epsilon=self.sync_variance_epsilon,
            )
            sync_metrics["sync_active_fraction"] = predictions.new_ones(())
        else:
            sync_metrics = {
                "sync_loss": predictions.new_zeros(()),
                "surrogate_sync_expected_abs_lag": (
                    predictions.new_zeros(())
                ),
                "surrogate_sync_hard_abs_lag": predictions.new_zeros(()),
                "surrogate_sync_zero_lag_correlation": (
                    predictions.new_zeros(())
                ),
                "surrogate_sync_peak_correlation": (
                    predictions.new_zeros(())
                ),
                "sync_valid_pair_fraction": predictions.new_zeros(()),
                "sync_valid_channels_at_peak": predictions.new_zeros(()),
                "sync_active_fraction": predictions.new_zeros(()),
            }
        residual_l2 = output["residual_raw"].square().mean()
        ccc_coverage_values = []
        for index in range(batch):
            valid_count = int(valid_targets[index].sum())
            if valid_count == 0:
                continue
            matched = best_indices[index, query_ccc_valid[index]].unique()
            matched = matched[valid_targets[index, matched]]
            ccc_coverage_values.append(matched.numel() / valid_count)
        match_coverage = (
            predictions.new_tensor(ccc_coverage_values).mean()
            if ccc_coverage_values
            else predictions.new_zeros(())
        )

        loss_appropriateness = (
            self.anchor_ccc_weight * (1.0 - paired_anchor_ccc)
            + self.learned_set_ccc_weight * (1.0 - learned_set_max_ccc)
            + self.sequence_transport_weight * sequence_transport_loss
            + self.mse_weight * matched_mse
            + self.velocity_weight * matched_velocity
            + self.style_weight * style_ot
            + self.sync_weight
            * float(sync_loss_scale)
            * sync_metrics["sync_loss"]
        )
        loss_semantic_allocation = (
            self.semantic_identity_weight * semantic_identity
            + self.compatibility_weight * compatibility_bce
            + self.mixture_weight * mixture_query_consistency
        )
        loss_distribution_risk = (
            self.style_diversity_weight * style_diversity_loss
            + self.official_diversity_weight * official_diversity_loss
            + self.official_diversity_batch_weight
            * official_diversity_batch_loss
            + self.channel_diversity_weight * channel_diversity_loss
            + self.frd_surrogate_weight * frd_surrogate_loss
            + self.frd_tail_weight * frd_tail_loss
            + self.frd_query_excess_weight * frd_query_excess_loss
            + self.frd_transport_weight * frd_transport_loss
        )
        total = (
            loss_appropriateness
            + loss_semantic_allocation
            + loss_distribution_risk
            + self.residual_weight * residual_l2
        )
        return {
            "loss": total,
            "loss_appropriateness": loss_appropriateness,
            "loss_semantic_allocation": loss_semantic_allocation,
            "loss_distribution_risk": loss_distribution_risk,
            "set_max_ccc": set_max_ccc,
            "paired_anchor_ccc": paired_anchor_ccc,
            "learned_set_max_ccc": learned_set_max_ccc,
            "sequence_transport_ccc": sequence_transport_ccc,
            "sequence_transport_alignment_cost": (
                sequence_transport_loss
            ),
            "matched_mse": matched_mse,
            "matched_velocity": matched_velocity,
            "paired_anchor_mse": matched_mse,
            "paired_anchor_velocity": matched_velocity,
            "style_ot": style_ot,
            "semantic_identity": semantic_identity,
            "semantic_accuracy": semantic_accuracy,
            "semantic_weighted_accuracy": semantic_weighted_accuracy,
            "semantic_identity_mean_weight": semantic_identity_mean_weight,
            "compatibility_bce": compatibility_bce,
            "compatibility_mean": compatibility_mean,
            "compatibility_min": compatibility_min,
            "compatibility_max": compatibility_max,
            "active_query_count_025": active_query_count_025,
            "active_query_count_050": active_query_count_050,
            "mixture_query_consistency": mixture_query_consistency,
            "gate_warmup_alpha": output.get(
                "gate_warmup_alpha",
                predictions.new_tensor(1.0),
            ),
            **{
                f"target_support_e{emotion_index}": target_support[
                    :, emotion_index
                ].mean()
                for emotion_index in range(NUM_EMOTIONS)
            },
            **{
                f"predicted_support_e{emotion_index}": predicted_support[
                    :, emotion_index
                ].mean()
                for emotion_index in range(NUM_EMOTIONS)
            },
            "style_pairwise_smse": style_pairwise_smse,
            "style_diversity_loss": style_diversity_loss,
            "style_diversity_active_fraction": (
                style_diversity_active_fraction
            ),
            "surrogate_official_frdiv": surrogate_official_frdiv,
            "official_diversity_loss": official_diversity_loss,
            "official_diversity_active_fraction": (
                official_diversity_active_fraction
            ),
            "official_diversity_batch_loss": (
                official_diversity_batch_loss
            ),
            "channel_diversity_loss": channel_diversity_loss,
            "channel_diversity_valid_fraction": (
                channel_diversity_valid_fraction
            ),
            "prediction_frdiv_au": prediction_frdiv_groups[0],
            "prediction_frdiv_va": prediction_frdiv_groups[1],
            "prediction_frdiv_expression": prediction_frdiv_groups[2],
            "target_frdiv_au": target_frdiv_groups[0],
            "target_frdiv_va": target_frdiv_groups[1],
            "target_frdiv_expression": target_frdiv_groups[2],
            "desired_frdiv_au": desired_frdiv_groups[0],
            "desired_frdiv_va": desired_frdiv_groups[1],
            "desired_frdiv_expression": desired_frdiv_groups[2],
            "surrogate_official_frd": surrogate_official_frd,
            "frd_surrogate_loss": frd_surrogate_loss,
            "frd_surrogate_active_fraction": (
                frd_surrogate_active_fraction
            ),
            "surrogate_official_frd_tail": surrogate_official_frd_tail,
            "frd_tail_loss": frd_tail_loss,
            "frd_tail_active_fraction": frd_tail_active_fraction,
            "surrogate_official_frd_query_excess": (
                surrogate_official_frd_query_excess
            ),
            "surrogate_official_frd_query_tail_excess": (
                surrogate_official_frd_query_tail_excess
            ),
            "frd_query_excess_loss": frd_query_excess_loss,
            "frd_query_excess_active_fraction": (
                frd_query_excess_active_fraction
            ),
            "frd_match_coverage": frd_match_coverage,
            "target_match_coverage": frd_match_coverage,
            "worst_query_excess": worst_query_excess,
            "learned_queries_better_than_anchor_fraction": (
                learned_queries_better_than_anchor_fraction
            ),
            **{
                f"frd_query_{query_index}_mean_best_target": (
                    mean_best_target_distance[query_index]
                )
                for query_index in range(queries)
            },
            **{
                f"frd_query_{query_index}_mean_excess": (
                    mean_query_excess[query_index]
                )
                for query_index in range(queries)
            },
            "surrogate_official_frd_transport": (
                surrogate_official_frd_transport
            ),
            "frd_transport_loss": frd_transport_loss,
            "frd_transport_active_fraction": (
                frd_transport_active_fraction
            ),
            "frd_transport_coverage": frd_transport_coverage,
            "frd_transport_entropy": frd_transport_entropy,
            **sync_metrics,
            "residual_l2": residual_l2,
            "match_coverage": match_coverage,
            "transport_entropy": -(
                transport * torch.log(transport.clamp_min(1e-8))
            ).sum(dim=(1, 2)).mean(),
            "sequence_transport_entropy": -(
                sequence_transport
                * torch.log(sequence_transport.clamp_min(1e-8))
            ).sum(dim=(1, 2)).mean(),
        }
