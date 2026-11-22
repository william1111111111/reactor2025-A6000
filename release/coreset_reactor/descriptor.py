"""Deterministic, differentiable reaction descriptors and TRAIN-only scaling."""

from __future__ import annotations

import math
import torch
from torch import nn


def reaction_descriptor(reactions: torch.Tensor, segments: int = 8) -> torch.Tensor:
    """Map [...,T,25] to [...,602]; no model parameters or GT access."""
    if reactions.shape[-1] != 25 or reactions.ndim < 2:
        raise ValueError("reaction must be [...,T,25]")
    leading = reactions.shape[:-2]
    frames = reactions.shape[-2]
    if frames < segments + 1:
        raise ValueError("reaction too short for descriptor segments")
    x = reactions.reshape(-1, frames, 25)
    velocity = x[:, 1:] - x[:, :-1]
    parts = [x.mean(1), x.std(1, unbiased=False), x.amin(1), x.amax(1),
             velocity.abs().mean(1), velocity.std(1, unbiased=False)]
    for s in range(segments):
        lo = s * frames // segments
        hi = (s + 1) * frames // segments
        block = x[:, lo:hi]
        parts.extend((block.mean(1),
                      (block[:, 1:] - block[:, :-1]).abs().mean(1)))
    va = x[..., 15:17]
    half = frames // 2
    parts.extend((va[:, :half].mean(1), va[:, half:].mean(1),
                  va[:, -1] - va[:, 0]))
    phase = (torch.arange(frames, device=x.device, dtype=x.dtype) + .5) / frames
    for frequency in (1, 2, 3):
        basis = (phase * (math.pi * frequency)).cos()
        parts.append((va * basis[None, :, None]).mean(1))
    expression = x[..., 17:25].clamp_min(1e-6)
    mean_expression = expression.mean(1)
    parts.extend((mean_expression,
                  mean_expression.max(dim=-1).values[:, None],
                  -(expression * expression.log()).sum(-1).mean(1)[:, None]))
    au = x[..., :15]
    parts.extend((torch.sigmoid((au - .5) / .08).mean(1),
                  torch.sigmoid((velocity[..., :15].abs() - .1) / .03).mean(1)))
    result = torch.cat(parts, dim=-1)
    if result.shape[-1] != 602:
        raise AssertionError(f"descriptor width changed: {result.shape[-1]}")
    return result.reshape(*leading, result.shape[-1])


class DescriptorScaler(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, floor: float = .03):
        super().__init__()
        if mean.shape != (602,) or std.shape != (602,):
            raise ValueError("descriptor statistics must have shape [602]")
        self.register_buffer("mean", mean.float().detach())
        self.register_buffer("std", std.float().detach().clamp_min(floor))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.mean) / self.std
