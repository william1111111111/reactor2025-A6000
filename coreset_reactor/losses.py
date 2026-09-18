"""Many-to-one set compression and sampled temporal matching objectives."""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from mam_reactor.regnn.frd_surrogate import grouped_banded_soft_dtw_cost


def softmin(cost: torch.Tensor, dimension: int, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return -temperature * torch.logsumexp(-cost / temperature, dim=dimension)


def descriptor_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred [B,K,D], gt [B,N,D] -> [B,K,N].
    return torch.cdist(pred, gt, p=2) / math.sqrt(pred.shape[-1])


def descriptor_set_loss(distance: torch.Tensor, temperature: float,
                        valid_weight: float, load_weight: float,
                        max_usage: float) -> dict[str, torch.Tensor]:
    # Every GT may be represented by one prediction; no Hungarian pairing.
    cover = softmin(distance, 1, temperature).mean()
    valid = softmin(distance, 2, temperature).mean()
    responsibility = torch.softmax(-distance / temperature, dim=1)
    usage = responsibility.mean(dim=-1)
    load = F.relu(usage.max(dim=1).values - max_usage).square().mean()
    return {"cover": cover, "valid": valid, "load": load,
            "total": cover + valid_weight * valid + load_weight * load}


def paired_cost(pred: torch.Tensor, target: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
    # pred [B,K,T,25], target [B,T,25] -> [B,K].
    frames = pred.shape[2]
    mask = (torch.arange(frames, device=pred.device)[None] < lengths[:, None])
    mask = mask[:, None, :, None].float()
    target = target[:, None]
    denominator = mask.sum(2).clamp_min(1)
    pmean = (pred * mask).sum(2) / denominator
    tmean = (target * mask).sum(2) / denominator
    pc = pred - pmean[:, :, None]
    tc = target - tmean[:, :, None]
    covariance = (pc * tc * mask).sum(2) / denominator
    pvar = (pc.square() * mask).sum(2) / denominator
    tvar = (tc.square() * mask).sum(2) / denominator
    ccc = 2 * covariance / (pvar + tvar + (pmean - tmean).square() + 1e-6)
    mse = ((pred - target).square() * mask).sum((2, 3)) / (mask.sum((2, 3)) * 25).clamp_min(1)
    velocity_mask = mask[:, :, 1:] * mask[:, :, :-1]
    velocity = ((pred[:, :, 1:] - pred[:, :, :-1] -
                 (target[:, :, 1:] - target[:, :, :-1])).square() * velocity_mask)
    velocity = velocity.sum((2, 3)) / (velocity_mask.sum((2, 3)) * 25).clamp_min(1)
    return (1 - ccc.mean(-1)) + mse + .2 * velocity


def unpaired_softdtw_cost(pred: torch.Tensor, target: torch.Tensor,
                          frames: int = 32, gamma: float = .1,
                          band_ratio: float = .25,
                          prediction_lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Batched differentiable SoftDTW, only for sampled unpaired GTs.

    pred [B,K,T,25], target [B,M,T,25] -> [B,K,M].
    This reuses only the existing loss kernel, not any Mam-Reactor model.
    """
    batch, candidates, time, _ = pred.shape
    count = target.shape[1]
    if count == 0:
        return pred.new_zeros(batch, candidates, 0)
    if prediction_lengths is None:
        valid_lengths = [time] * batch
    else:
        if prediction_lengths.shape != (batch,):
            raise ValueError("prediction_lengths must have shape [B]")
        valid_lengths = [int(length) for length in prediction_lengths.detach().cpu().tolist()]
        if any(length < 1 or length > time for length in valid_lengths):
            raise ValueError("prediction_lengths must be between 1 and T")
    if all(length == time for length in valid_lengths):
        p = F.interpolate(pred.flatten(0, 1).transpose(1, 2), size=frames,
                          mode="linear", align_corners=True).transpose(1, 2)
        p = p.reshape(batch, candidates, frames, 25)
    else:
        p = torch.stack([
            F.interpolate(pred[index, :, :length].transpose(1, 2), size=frames,
                          mode="linear", align_corners=True).transpose(1, 2)
            for index, length in enumerate(valid_lengths)
        ])
    g = F.interpolate(target.flatten(0, 1).transpose(1, 2), size=frames,
                      mode="linear", align_corners=True).transpose(1, 2)
    g = g.reshape(batch, count, frames, 25)
    left = p[:, :, None].expand(-1, -1, count, -1, -1).reshape(-1, frames, 25)
    right = g[:, None].expand(-1, candidates, -1, -1, -1).reshape(-1, frames, 25)
    lengths = torch.full((left.shape[0],), frames, device=pred.device, dtype=torch.long)
    group = grouped_banded_soft_dtw_cost(left, right, lengths, lengths,
                                         gamma=gamma, band_ratio=band_ratio)
    weights = group.new_tensor((1 / 15, 1, 1 / 8))
    return (group * weights).sum(-1).reshape(batch, candidates, count) / frames
