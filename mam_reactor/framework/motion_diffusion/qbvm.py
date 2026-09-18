"""Quality-budgeted temporal variance objective for 83D reactions.

The objective is deliberately computed per generated sequence along time.
It is not a cross-candidate diversity loss.  Variance is rewarded
monotonically, while velocity, acceleration, range, and optional paired CCC
constraints prevent trivial jitter or drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    end: int


FEATURE_GROUPS: Tuple[FeatureGroup, ...] = (
    FeatureGroup("au", 0, 15),
    FeatureGroup("va", 15, 17),
    FeatureGroup("emotion", 17, 25),
    FeatureGroup("expression", 25, 77),
    FeatureGroup("rotation", 77, 80),
    FeatureGroup("translation", 80, 83),
)


def _validate_sequence_inputs(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim != 4 or prediction.shape[-1] != 83:
        raise ValueError(
            "QBVM prediction must have shape [B,K,T,83], got "
            f"{tuple(prediction.shape)}."
        )
    if target.shape != prediction.shape:
        raise ValueError(
            "QBVM target must match prediction, got "
            f"{tuple(target.shape)} vs {tuple(prediction.shape)}."
        )
    if valid_mask.shape != prediction.shape[:3]:
        raise ValueError(
            "QBVM valid_mask must have shape [B,K,T], got "
            f"{tuple(valid_mask.shape)}."
        )
    valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    if bool((valid_mask.sum(dim=-1) < 1).any()):
        raise ValueError("QBVM received a sequence with no valid frames.")
    return valid_mask


def _masked_group_variance(
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
        group: FeatureGroup,
        scale: torch.Tensor,
) -> torch.Tensor:
    """Return per-[B,K] mean feature variance along valid time frames."""
    values = sequence[..., group.start:group.end] / scale
    mask = valid_mask[..., None].to(values.dtype)
    count = mask.sum(dim=2).clamp_min(1.0)
    mean = (values * mask).sum(dim=2) / count
    centered = (values - mean[:, :, None, :]) * mask
    variance = centered.square().sum(dim=2) / count
    return variance.mean(dim=-1)


def _masked_difference_energy(
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
        group: FeatureGroup,
        scale: torch.Tensor,
        order: int,
) -> torch.Tensor:
    """Return normalized first/second-difference energy per [B,K]."""
    values = sequence[..., group.start:group.end] / scale
    mask = valid_mask
    for _ in range(order):
        values = values[:, :, 1:] - values[:, :, :-1]
        mask = mask[:, :, 1:] & mask[:, :, :-1]
    mask_f = mask[..., None].to(values.dtype)
    count = (mask_f.sum(dim=2) * values.shape[-1]).clamp_min(1.0)
    return (values.square() * mask_f).sum(dim=(2, 3)) / count.squeeze(-1)


def _masked_group_range_energy(
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
        group: FeatureGroup,
        scale: torch.Tensor,
) -> torch.Tensor:
    """Return squared normalized temporal range, averaged over features."""
    values = sequence[..., group.start:group.end] / scale
    mask = valid_mask[..., None]
    max_values = values.masked_fill(~mask, -torch.inf).amax(dim=2)
    min_values = values.masked_fill(~mask, torch.inf).amin(dim=2)
    return (max_values - min_values).square().mean(dim=-1)


def _masked_ccc(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        eps: float,
) -> torch.Tensor:
    """Mean per-dimension CCC over the official first 25 dimensions."""
    prediction = prediction[..., :25]
    target = target[..., :25]
    mask = valid_mask[..., None].to(prediction.dtype)
    count = mask.sum(dim=2).clamp_min(1.0)
    pred_mean = (prediction * mask).sum(dim=2) / count
    target_mean = (target * mask).sum(dim=2) / count
    pred_centered = (prediction - pred_mean[:, :, None, :]) * mask
    target_centered = (target - target_mean[:, :, None, :]) * mask
    pred_var = pred_centered.square().sum(dim=2) / count
    target_var = target_centered.square().sum(dim=2) / count
    covariance = (pred_centered * target_centered).sum(dim=2) / count
    ccc = (2.0 * covariance) / (
        pred_var
        + target_var
        + (pred_mean - target_mean).square()
        + eps
    )
    return ccc.mean(dim=-1)


class QualityBudgetedVarianceObjective(nn.Module):
    """Monotonically reward per-sequence variance under motion constraints."""

    def __init__(
        self,
        variance_weight: float,
        group_weights: Sequence[float],
        group_scales: Sequence[float],
        variance_epsilon: float = 1e-4,
        energy_floor: float = 0.05,
        velocity_ratio_limit: float = 1.20,
        acceleration_ratio_limit: float = 1.20,
        range_ratio_limit: float = 1.20,
        velocity_penalty_weight: float = 0.05,
        acceleration_penalty_weight: float = 0.10,
        range_penalty_weight: float = 0.05,
        ccc_minimum: float = 0.0,
        ccc_penalty_weight: float = 0.0,
        use_flow_time_weight: bool = True,
    ) -> None:
        super().__init__()
        if len(group_weights) != len(FEATURE_GROUPS):
            raise ValueError("QBVM requires six group weights.")
        if len(group_scales) != len(FEATURE_GROUPS):
            raise ValueError("QBVM requires six group scales.")
        if any(value <= 0 for value in group_scales):
            raise ValueError("QBVM group scales must all be positive.")
        if variance_weight < 0:
            raise ValueError("QBVM variance_weight must be non-negative.")

        weights = torch.tensor(group_weights, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        self.register_buffer("group_weights", weights)
        self.register_buffer(
            "group_scales",
            torch.tensor(group_scales, dtype=torch.float32),
        )
        self.variance_weight = float(variance_weight)
        self.variance_epsilon = float(variance_epsilon)
        self.energy_floor = float(energy_floor)
        self.velocity_ratio_limit = float(velocity_ratio_limit)
        self.acceleration_ratio_limit = float(acceleration_ratio_limit)
        self.range_ratio_limit = float(range_ratio_limit)
        self.velocity_penalty_weight = float(velocity_penalty_weight)
        self.acceleration_penalty_weight = float(acceleration_penalty_weight)
        self.range_penalty_weight = float(range_penalty_weight)
        self.ccc_minimum = float(ccc_minimum)
        self.ccc_penalty_weight = float(ccc_penalty_weight)
        self.use_flow_time_weight = bool(use_flow_time_weight)

    @staticmethod
    def _ratio_violation(
        prediction_energy: torch.Tensor,
        target_energy: torch.Tensor,
        ratio_limit: float,
        floor: float,
    ) -> torch.Tensor:
        # The zero set is the same as a raw ratio hinge, but the log keeps
        # early from-scratch predictions from producing unbounded quadratic
        # gradients when they exceed the paired motion budget by 100x.
        log_ratio = torch.log(
            (prediction_energy + floor)
            / (ratio_limit * (target_energy + floor) + 1e-8)
        )
        return F.relu(log_ratio).square()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        flow_time: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        valid_mask = _validate_sequence_inputs(prediction, target, valid_mask)
        # Temporal variances and motion-energy ratios can span several orders
        # of magnitude early in training. Keep these statistics in FP32 even
        # when the denoiser runs under BF16 autocast; `.float()` preserves the
        # gradient path back to the generated sequence.
        prediction = prediction.float()
        target = target.float()
        weights = self.group_weights.to(
            device=prediction.device, dtype=torch.float32
        )
        scales = self.group_scales.to(
            device=prediction.device, dtype=torch.float32
        )

        variances = []
        velocity_violations = []
        acceleration_violations = []
        range_violations = []
        for group_index, group in enumerate(FEATURE_GROUPS):
            scale = scales[group_index]
            pred_variance = _masked_group_variance(
                prediction, valid_mask, group, scale
            )
            variances.append(pred_variance)

            pred_velocity = _masked_difference_energy(
                prediction, valid_mask, group, scale, order=1
            )
            target_velocity = _masked_difference_energy(
                target, valid_mask, group, scale, order=1
            )
            velocity_violations.append(
                self._ratio_violation(
                    pred_velocity,
                    target_velocity,
                    self.velocity_ratio_limit,
                    self.energy_floor,
                )
            )

            pred_acceleration = _masked_difference_energy(
                prediction, valid_mask, group, scale, order=2
            )
            target_acceleration = _masked_difference_energy(
                target, valid_mask, group, scale, order=2
            )
            acceleration_violations.append(
                self._ratio_violation(
                    pred_acceleration,
                    target_acceleration,
                    self.acceleration_ratio_limit,
                    self.energy_floor,
                )
            )

            pred_range = _masked_group_range_energy(
                prediction, valid_mask, group, scale
            )
            target_range = _masked_group_range_energy(
                target, valid_mask, group, scale
            )
            range_violations.append(
                self._ratio_violation(
                    pred_range,
                    target_range,
                    self.range_ratio_limit,
                    self.energy_floor,
                )
            )

        variance_tensor = torch.stack(variances, dim=-1)
        velocity_tensor = torch.stack(velocity_violations, dim=-1)
        acceleration_tensor = torch.stack(acceleration_violations, dim=-1)
        range_tensor = torch.stack(range_violations, dim=-1)

        per_sequence_reward = (
            weights
            * torch.log(variance_tensor + self.variance_epsilon)
        ).sum(dim=-1)
        per_sequence_velocity = (weights * velocity_tensor).sum(dim=-1)
        per_sequence_acceleration = (weights * acceleration_tensor).sum(dim=-1)
        per_sequence_range = (weights * range_tensor).sum(dim=-1)
        per_sequence_ccc = _masked_ccc(
            prediction, target, valid_mask, self.variance_epsilon
        )
        per_sequence_ccc_violation = F.relu(
            self.ccc_minimum - per_sequence_ccc
        ).square()

        if flow_time is None or not self.use_flow_time_weight:
            time_weight = torch.ones_like(per_sequence_reward)
        else:
            if flow_time.shape[:3] != prediction.shape[:3]:
                raise ValueError(
                    "QBVM flow_time must start with [B,K,T], got "
                    f"{tuple(flow_time.shape)}."
                )
            time = flow_time[..., 0] if flow_time.ndim == 4 else flow_time
            time = time.mean(dim=2).detach()
            time_weight = 4.0 * time * (1.0 - time)

        reward = (time_weight * per_sequence_reward).mean()
        velocity_violation = (time_weight * per_sequence_velocity).mean()
        acceleration_violation = (
            time_weight * per_sequence_acceleration
        ).mean()
        range_violation = (time_weight * per_sequence_range).mean()
        ccc_violation = (time_weight * per_sequence_ccc_violation).mean()

        variance_loss = -self.variance_weight * reward
        velocity_penalty = self.velocity_penalty_weight * velocity_violation
        acceleration_penalty = (
            self.acceleration_penalty_weight * acceleration_violation
        )
        range_penalty = self.range_penalty_weight * range_violation
        ccc_penalty = self.ccc_penalty_weight * ccc_violation
        loss = (
            variance_loss
            + velocity_penalty
            + acceleration_penalty
            + range_penalty
            + ccc_penalty
        )

        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "reward": reward,
            "variance_loss": variance_loss,
            "velocity_violation": velocity_violation,
            "acceleration_violation": acceleration_violation,
            "range_violation": range_violation,
            "ccc": per_sequence_ccc.mean(),
            "ccc_violation": ccc_violation,
            "normalized_variance": (
                weights * variance_tensor
            ).sum(dim=-1).mean(),
        }
        for group_index, group in enumerate(FEATURE_GROUPS):
            result[f"variance_{group.name}"] = variance_tensor[
                ..., group_index
            ].mean()
        return result
