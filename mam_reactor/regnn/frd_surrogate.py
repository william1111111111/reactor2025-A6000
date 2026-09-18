"""Differentiable alignment-aware surrogate for the official REACT FRD.

This module preserves the official AU/VA/expression feature grouping but is
not the official metric. Final FRD must still be computed with
``tools/compute_frd_resumable.py``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


OFFICIAL_FRD_GROUPS = (
    (0, 15, 1.0 / 15.0),
    (15, 17, 1.0),
    (17, 25, 1.0 / 8.0),
)


def _gather_diagonal(
    diagonal: torch.Tensor | None,
    diagonal_low: int,
    coordinates: torch.Tensor,
    large_value: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if diagonal is None or diagonal.shape[1] == 0:
        return torch.full(
            (batch_size, coordinates.numel()),
            large_value,
            dtype=torch.float32,
            device=device,
        )
    indices = coordinates - diagonal_low
    valid = (indices >= 0) & (indices < diagonal.shape[1])
    safe_indices = indices.clamp(0, diagonal.shape[1] - 1)
    gathered = diagonal.index_select(1, safe_indices)
    return torch.where(
        valid[None],
        gathered,
        gathered.new_full((), large_value),
    )


def banded_soft_dtw_cost(
    local_cost: torch.Tensor,
    prediction_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    gamma: float,
    band_ratio: float,
) -> torch.Tensor:
    """Compute batched Soft-DTW using vectorized anti-diagonals.

    ``local_cost`` is ``[N,Tp,Tt]`` and contains squared Euclidean costs.
    The normalized Sakoe-Chiba band supports unequal valid lengths. Only
    ``Tp+Tt`` sequential steps are launched; cells on one anti-diagonal are
    evaluated together.
    """
    if local_cost.ndim != 3:
        raise ValueError("local_cost must have shape [N,Tp,Tt]")
    if local_cost.dtype != torch.float32:
        raise ValueError("local_cost must be float32")
    if prediction_lengths.shape != (local_cost.shape[0],):
        raise ValueError("prediction_lengths must have shape [N]")
    if target_lengths.shape != (local_cost.shape[0],):
        raise ValueError("target_lengths must have shape [N]")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if not 0.0 <= band_ratio <= 1.0:
        raise ValueError("band_ratio must be in [0,1]")

    pair_count, max_prediction, max_target = local_cost.shape
    device = local_cost.device
    prediction_lengths = prediction_lengths.to(device=device, dtype=torch.long)
    target_lengths = target_lengths.to(device=device, dtype=torch.long)
    large_value = 1e6
    result = local_cost.new_full((pair_count,), large_value)

    diagonals: list[torch.Tensor | None] = [None] * (
        max_prediction + max_target + 1
    )
    diagonal_lows = [0] * len(diagonals)
    diagonals[0] = local_cost.new_zeros(pair_count, 1)
    diagonal_lows[0] = 0

    for diagonal_index in range(2, len(diagonals)):
        low = max(1, diagonal_index - max_target)
        high = min(max_prediction, diagonal_index - 1)
        if low > high:
            continue
        prediction_indices = torch.arange(low, high + 1, device=device)
        target_indices = diagonal_index - prediction_indices

        previous = diagonals[diagonal_index - 1]
        previous_low = diagonal_lows[diagonal_index - 1]
        previous_previous = diagonals[diagonal_index - 2]
        previous_previous_low = diagonal_lows[diagonal_index - 2]
        up = _gather_diagonal(
            previous,
            previous_low,
            prediction_indices - 1,
            large_value,
            pair_count,
            device,
        )
        left = _gather_diagonal(
            previous,
            previous_low,
            prediction_indices,
            large_value,
            pair_count,
            device,
        )
        diagonal = _gather_diagonal(
            previous_previous,
            previous_previous_low,
            prediction_indices - 1,
            large_value,
            pair_count,
            device,
        )
        predecessors = torch.stack((up, left, diagonal), dim=-1)
        reachable = predecessors.amin(dim=-1) < large_value * 0.5
        soft_minimum = -float(gamma) * torch.logsumexp(
            -predecessors / float(gamma),
            dim=-1,
        )

        local = local_cost[
            :,
            prediction_indices - 1,
            target_indices - 1,
        ]
        prediction_length = prediction_lengths[:, None].clamp_min(1)
        target_length = target_lengths[:, None].clamp_min(1)
        normalized_offset = (
            prediction_indices[None].to(local_cost) / prediction_length
            - target_indices[None].to(local_cost) / target_length
        ).abs()
        discretization_tolerance = 1.0 / torch.maximum(
            prediction_length,
            target_length,
        ).to(local_cost)
        inside_band = normalized_offset <= (
            float(band_ratio) + discretization_tolerance
        )
        inside_lengths = (
            prediction_indices[None] <= prediction_lengths[:, None]
        ) & (target_indices[None] <= target_lengths[:, None])
        valid_cells = inside_band & inside_lengths & reachable
        values = torch.where(
            valid_cells,
            local + soft_minimum,
            local.new_full((), large_value),
        )
        diagonals[diagonal_index] = values
        diagonal_lows[diagonal_index] = low

        endpoint_indices = prediction_lengths - low
        endpoint_present = (
            prediction_lengths + target_lengths == diagonal_index
        ) & (endpoint_indices >= 0) & (endpoint_indices < values.shape[1])
        safe_endpoint_indices = endpoint_indices.clamp(0, values.shape[1] - 1)
        endpoint_values = values.gather(
            1,
            safe_endpoint_indices[:, None],
        ).squeeze(1)
        result = torch.where(endpoint_present, endpoint_values, result)

    valid_pairs = (prediction_lengths > 0) & (target_lengths > 0)
    return torch.where(
        valid_pairs,
        result,
        result.new_full((), large_value),
    )


def grouped_banded_soft_dtw_cost(
    left: torch.Tensor,
    right: torch.Tensor,
    left_lengths: torch.Tensor,
    right_lengths: torch.Tensor,
    gamma: float,
    band_ratio: float,
) -> torch.Tensor:
    """Return raw Soft-DTW costs for the three official feature groups.

    Inputs have shape ``[N,T,25]`` and the result has shape ``[N,3]``.
    Keeping this operation grouped lets the dynamic-programming kernel process
    AU, VA, and expression costs in one call.
    """
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("left and right must have shape [N,T,25]")
    if left.shape[0] != right.shape[0] or left.shape[-1] != 25:
        raise ValueError("left and right must share N and have 25 channels")
    if right.shape[-1] != 25:
        raise ValueError("left and right must have 25 channels")
    group_local_costs = []
    for feature_start, feature_end, _ in OFFICIAL_FRD_GROUPS:
        group_local_costs.append(
            torch.cdist(
                left[..., feature_start:feature_end],
                right[..., feature_start:feature_end],
                p=2.0,
                compute_mode="use_mm_for_euclid_dist",
            ).square()
        )
    group_count = len(OFFICIAL_FRD_GROUPS)
    stacked_local_cost = torch.stack(group_local_costs, dim=0)
    return banded_soft_dtw_cost(
        stacked_local_cost.flatten(0, 1),
        left_lengths.repeat(group_count),
        right_lengths.repeat(group_count),
        gamma=gamma,
        band_ratio=band_ratio,
    ).reshape(group_count, -1).transpose(0, 1)


class BandedSoftDTWFRDSurrogate(nn.Module):
    """Chunked Soft-DTW pair costs for ``[B,Q] x [B,K]`` reactions."""

    def __init__(
        self,
        temporal_stride: int = 6,
        band_ratio: float = 0.15,
        gamma: float = 0.1,
        pair_chunk_size: int = 256,
        calibration: float = 1.0,
    ):
        super().__init__()
        if temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive")
        if not 0.0 <= band_ratio <= 1.0:
            raise ValueError("band_ratio must be in [0,1]")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if calibration <= 0.0:
            raise ValueError("calibration must be positive")
        self.temporal_stride = int(temporal_stride)
        self.band_ratio = float(band_ratio)
        self.gamma = float(gamma)
        self.pair_chunk_size = int(pair_chunk_size)
        self.calibration = float(calibration)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prediction_valid_mask: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if predictions.ndim != 4 or predictions.shape[-1] != 25:
            raise ValueError("predictions must have shape [B,Q,T,25]")
        if targets.ndim != 4 or targets.shape[-1] != 25:
            raise ValueError("targets must have shape [B,K,T,25]")
        if (
            predictions.shape[0] != targets.shape[0]
            or predictions.shape[2:] != targets.shape[2:]
        ):
            raise ValueError("predictions and targets must share B,T,25")
        expected_prediction_mask_shape = (
            predictions.shape[0],
            predictions.shape[2],
        )
        if prediction_valid_mask.shape != expected_prediction_mask_shape:
            raise ValueError("prediction_valid_mask must have shape [B,T]")
        if target_valid_mask.shape != targets.shape[:3]:
            raise ValueError("target_valid_mask must have shape [B,K,T]")

        rounded_prediction_au = predictions[..., :15] + (
            torch.round(predictions[..., :15]) - predictions[..., :15]
        ).detach()
        official_predictions = torch.cat(
            (rounded_prediction_au, predictions[..., 15:]),
            dim=-1,
        ).float()
        official_targets = torch.cat(
            (torch.round(targets[..., :15]), targets[..., 15:]),
            dim=-1,
        ).float()

        sampled_predictions = official_predictions[
            :, :, :: self.temporal_stride
        ]
        sampled_targets = official_targets[:, :, :: self.temporal_stride]
        sampled_prediction_mask = prediction_valid_mask[
            :, :: self.temporal_stride
        ]
        sampled_target_mask = target_valid_mask[:, :, :: self.temporal_stride]

        batch, query_count = predictions.shape[:2]
        target_count = targets.shape[1]
        sampled_prediction_lengths = sampled_prediction_mask.sum(dim=-1)
        sampled_target_lengths = sampled_target_mask.sum(dim=-1)
        full_prediction_lengths = prediction_valid_mask.sum(dim=-1).float()
        full_target_lengths = target_valid_mask.sum(dim=-1).float()

        flat_predictions = sampled_predictions.flatten(0, 1)
        flat_prediction_lengths = sampled_prediction_lengths[:, None].expand(
            -1,
            query_count,
        ).reshape(-1)
        flat_full_prediction_lengths = full_prediction_lengths[
            :, None
        ].expand(-1, query_count).reshape(-1)
        prediction_self_cost_chunks = []
        for chunk_start in range(
            0,
            flat_predictions.shape[0],
            self.pair_chunk_size,
        ):
            chunk_end = min(
                flat_predictions.shape[0],
                chunk_start + self.pair_chunk_size,
            )
            chunk_sequences = flat_predictions[chunk_start:chunk_end]
            chunk_lengths = flat_prediction_lengths[chunk_start:chunk_end]
            chunk_full_lengths = flat_full_prediction_lengths[
                chunk_start:chunk_end
            ]
            raw_self_cost = grouped_banded_soft_dtw_cost(
                chunk_sequences,
                chunk_sequences,
                chunk_lengths,
                chunk_lengths,
                gamma=self.gamma,
                band_ratio=self.band_ratio,
            )
            self_scale = chunk_full_lengths / chunk_lengths.float().clamp_min(
                1.0
            )
            prediction_self_cost_chunks.append(
                raw_self_cost * self_scale[:, None]
            )
        prediction_self_cost = torch.cat(prediction_self_cost_chunks, dim=0)

        flat_targets = sampled_targets.flatten(0, 1)
        flat_target_lengths = sampled_target_lengths.reshape(-1)
        flat_full_target_lengths = full_target_lengths.reshape(-1)
        target_self_cost_chunks = []
        for chunk_start in range(
            0,
            flat_targets.shape[0],
            self.pair_chunk_size,
        ):
            chunk_end = min(
                flat_targets.shape[0],
                chunk_start + self.pair_chunk_size,
            )
            chunk_sequences = flat_targets[chunk_start:chunk_end]
            chunk_lengths = flat_target_lengths[chunk_start:chunk_end]
            chunk_full_lengths = flat_full_target_lengths[
                chunk_start:chunk_end
            ]
            raw_self_cost = grouped_banded_soft_dtw_cost(
                chunk_sequences,
                chunk_sequences,
                chunk_lengths,
                chunk_lengths,
                gamma=self.gamma,
                band_ratio=self.band_ratio,
            )
            self_scale = chunk_full_lengths / chunk_lengths.float().clamp_min(
                1.0
            )
            target_self_cost_chunks.append(
                raw_self_cost * self_scale[:, None]
            )
        target_self_cost = torch.cat(target_self_cost_chunks, dim=0)

        pair_ids = torch.arange(
            batch * query_count * target_count,
            device=predictions.device,
        )
        batch_indices = pair_ids // (query_count * target_count)
        within_batch = pair_ids % (query_count * target_count)
        query_indices = within_batch // target_count
        target_indices = within_batch % target_count

        chunks = []
        for chunk_start in range(0, pair_ids.numel(), self.pair_chunk_size):
            chunk_end = min(
                pair_ids.numel(),
                chunk_start + self.pair_chunk_size,
            )
            batch_index = batch_indices[chunk_start:chunk_end]
            query_index = query_indices[chunk_start:chunk_end]
            target_index = target_indices[chunk_start:chunk_end]
            chunk_predictions = sampled_predictions[batch_index, query_index]
            chunk_targets = sampled_targets[batch_index, target_index]
            prediction_lengths = sampled_prediction_mask[
                batch_index
            ].sum(dim=-1)
            target_lengths = sampled_target_mask[
                batch_index,
                target_index,
            ].sum(dim=-1)
            chunk_full_prediction_lengths = prediction_valid_mask[
                batch_index
            ].sum(dim=-1).float()
            chunk_full_target_lengths = target_valid_mask[
                batch_index,
                target_index,
            ].sum(dim=-1).float()
            scale = 0.5 * (
                chunk_full_prediction_lengths
                / prediction_lengths.float().clamp_min(1.0)
                + chunk_full_target_lengths
                / target_lengths.float().clamp_min(1.0)
            )
            cross_cost = grouped_banded_soft_dtw_cost(
                chunk_predictions,
                chunk_targets,
                prediction_lengths,
                target_lengths,
                gamma=self.gamma,
                band_ratio=self.band_ratio,
            )
            flat_prediction_index = batch_index * query_count + query_index
            flat_target_index = batch_index * target_count + target_index
            divergence = (
                cross_cost * scale[:, None]
                - 0.5 * prediction_self_cost[flat_prediction_index]
                - 0.5 * target_self_cost[flat_target_index]
            )
            group_distance = torch.sqrt(divergence.clamp_min(1e-12))
            group_weights = group_distance.new_tensor(
                [group[2] for group in OFFICIAL_FRD_GROUPS]
            )
            chunk_cost = (
                group_distance * group_weights[None]
            ).sum(dim=-1)
            valid_pairs = (prediction_lengths > 0) & (target_lengths > 0)
            chunk_cost = torch.where(
                valid_pairs,
                chunk_cost * self.calibration,
                chunk_cost.new_full((), torch.finfo(torch.float32).max),
            )
            chunks.append(chunk_cost)

        pairwise_cost = torch.cat(chunks).reshape(
            batch,
            query_count,
            target_count,
        )
        valid_targets = target_valid_mask.any(dim=-1)
        pairwise_cost = pairwise_cost.masked_fill(
            ~valid_targets[:, None],
            torch.finfo(pairwise_cost.dtype).max,
        )
        valid_examples = valid_targets.any(dim=-1)
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
        return {
            "mean": mean,
            "per_example": per_example,
            "pairwise_cost": pairwise_cost,
            "best_target_cost": best_target_cost,
            "valid_targets": valid_targets,
            "valid_examples": valid_examples,
        }
