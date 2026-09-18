"""Shared primitives for isolated EQR VA-channel risk calibration.

These helpers do not define a new model component. They restrict an auxiliary
optimizer to rows 15:17 of the existing ``to_residual`` projection and compute
the same VA-only risk/preservation objective used by post-hoc calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class IsolatedVACalibrationConfig:
    tail_fraction: float = 0.25
    mean_risk_weight: float = 0.25
    proximal_weight: float = 0.05
    diversity_preservation_weight: float = 10.0
    variance_preservation_weight: float = 10.0
    preservation_ratio: float = 0.99

    def validate(self) -> None:
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0,1]")
        if not 0.0 < self.preservation_ratio <= 1.0:
            raise ValueError("preservation_ratio must be in (0,1]")
        weights = (
            self.mean_risk_weight,
            self.proximal_weight,
            self.diversity_preservation_weight,
            self.variance_preservation_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("VA calibration loss weights must be non-negative")


def va_only(values: torch.Tensor) -> torch.Tensor:
    """Project 25-D facial attributes onto VA channels without slicing shape."""
    if values.shape[-1] != 25:
        raise ValueError("VA projection requires a 25-D final channel axis")
    result = torch.zeros_like(values)
    result[..., 15:17] = values[..., 15:17]
    return result


def masked_frdiv(
    predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Official-style candidate SMSE averaged over valid variable lengths."""
    values = []
    for sample, mask in zip(predictions, valid_mask):
        valid = sample[:, mask]
        queries, frames, channels = valid.shape
        if frames == 0 or queries < 2:
            raise ValueError("FRDiv preservation requires valid frames and queries")
        flattened = valid.reshape(queries, frames * channels)
        squared = torch.cdist(flattened, flattened).square()
        values.append(
            squared.sum() / (queries * (queries - 1) * frames * channels)
        )
    return torch.stack(values).mean()


def masked_frvar(
    predictions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Official-style temporal variance averaged over valid variable lengths."""
    values = []
    for sample, mask in zip(predictions, valid_mask):
        valid = sample[:, mask]
        if valid.shape[1] < 2:
            raise ValueError("FRVar preservation requires at least two frames")
        values.append(valid.var(dim=1).mean())
    return torch.stack(values).mean()


def isolated_va_objective(
    predictions: torch.Tensor,
    reference_predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    surrogate: torch.nn.Module,
    config: IsolatedVACalibrationConfig,
) -> dict[str, torch.Tensor]:
    """Compute VA risk relative to a stop-gradient pre-microstep reference."""
    config.validate()
    if predictions.shape != reference_predictions.shape:
        raise ValueError("Predictions and VA reference must share shape")
    if predictions.ndim != 4 or predictions.shape[1] != 10:
        raise ValueError("Expected predictions with shape [B,10,T,25]")

    predictions = predictions.float()
    reference_predictions = reference_predictions.detach().float()
    targets = targets.float()

    learned_predictions = va_only(predictions[:, 1:])
    learned_reference = va_only(reference_predictions[:, 1:])
    va_targets = va_only(targets)
    current = surrogate(
        learned_predictions,
        va_targets,
        prediction_valid_mask,
        target_valid_mask,
    )
    with torch.no_grad():
        reference = surrogate(
            learned_reference,
            va_targets,
            prediction_valid_mask,
            target_valid_mask,
        )
    risk_ratio = current["per_example"] / reference["per_example"].clamp_min(
        1e-4
    )
    tail_count = max(
        1,
        int(math.ceil(risk_ratio.numel() * config.tail_fraction)),
    )
    tail_risk = risk_ratio.topk(tail_count).values.mean()
    mean_risk = risk_ratio.mean()

    reference_energy = (
        reference_predictions[:, 1:, :, 15:17]
        - reference_predictions[:, :1, :, 15:17]
    ).square().mean().clamp_min(1e-5)
    proximal = (
        predictions[:, 1:, :, 15:17]
        - reference_predictions[:, 1:, :, 15:17].detach()
    ).square().mean() / reference_energy

    current_diversity = masked_frdiv(predictions, prediction_valid_mask)
    reference_diversity = masked_frdiv(
        reference_predictions.detach(), prediction_valid_mask
    )
    current_variance = masked_frvar(predictions, prediction_valid_mask)
    reference_variance = masked_frvar(
        reference_predictions.detach(), prediction_valid_mask
    )
    diversity_ratio = current_diversity / reference_diversity.clamp_min(1e-6)
    variance_ratio = current_variance / reference_variance.clamp_min(1e-6)
    diversity_shortfall = torch.relu(
        config.preservation_ratio - diversity_ratio
    ).square()
    variance_shortfall = torch.relu(
        config.preservation_ratio - variance_ratio
    ).square()
    loss = (
        tail_risk
        + config.mean_risk_weight * mean_risk
        + config.proximal_weight * proximal
        + config.diversity_preservation_weight * diversity_shortfall
        + config.variance_preservation_weight * variance_shortfall
    )
    return {
        "loss": loss,
        "tail_risk_ratio": tail_risk,
        "mean_risk_ratio": mean_risk,
        "proximal": proximal,
        "FRDiv_ratio": diversity_ratio,
        "FRVar_ratio": variance_ratio,
        "diversity_shortfall": diversity_shortfall,
        "variance_shortfall": variance_shortfall,
    }


def set_only_output_head_trainable(
    model: torch.nn.Module,
) -> dict[str, bool]:
    """Temporarily freeze the model except the existing residual projection."""
    states = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to_residual.weight.requires_grad_(True)
    model.to_residual.bias.requires_grad_(True)
    return states


def restore_requires_grad(
    model: torch.nn.Module,
    states: dict[str, bool],
) -> None:
    names = {name for name, _ in model.named_parameters()}
    if names != set(states):
        raise RuntimeError("Model parameter set changed during VA micro-update")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(states[name])


def mask_output_head_gradients_to_va(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Hard-mask output-head gradients so only rows/biases 15:17 can move."""
    weight_gradient = model.to_residual.weight.grad
    bias_gradient = model.to_residual.bias.grad
    if weight_gradient is None or bias_gradient is None:
        raise RuntimeError("VA micro-update did not produce output-head gradients")
    with torch.no_grad():
        weight_gradient[:15].zero_()
        weight_gradient[17:].zero_()
        bias_gradient[:15].zero_()
        bias_gradient[17:].zero_()
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if name not in {"to_residual.weight", "to_residual.bias"}
        and parameter.grad is not None
    ]
    if unexpected:
        raise RuntimeError(f"VA micro-update leaked gradients: {unexpected}")
    return {
        "weight_nonzero_gradient_scalars": int(
            torch.count_nonzero(weight_gradient[15:17]).item()
        ),
        "bias_nonzero_gradient_scalars": int(
            torch.count_nonzero(bias_gradient[15:17]).item()
        ),
        "unexpected_gradient_parameter_names": unexpected,
    }
