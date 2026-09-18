"""Numerically robust diagnostic variant of the REACT2025 TLCC metric.

The challenge implementation in :mod:`framework.metrics.TLCC` is intentionally
left untouched for reproducibility.  In that implementation one constant
feature makes the correlation at every lag NaN, and the prediction loop returns
after query zero.  This module provides an explicitly named diagnostic metric
that filters undefined feature correlations and evaluates every prediction.

``compute_TLCC_robust`` is *not* a drop-in replacement for the official score.
Its primary value is ``mean_all_queries``, matching the apparent intent of the
commented non-multiprocessing implementation in the original metric.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _shift_numpy(
    prediction: np.ndarray,
    speaker: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return prediction[:, lag:, :], speaker[:-lag, :]
    if lag < 0:
        return prediction[:, :lag, :], speaker[-lag:, :]
    return prediction, speaker


def _finite_correlations(
    prediction: np.ndarray,
    speaker: np.ndarray,
    *,
    variance_epsilon: float,
    min_overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean correlation and valid-channel count for every query."""

    num_queries, overlap, num_features = prediction.shape
    scores = np.full(num_queries, np.nan, dtype=np.float64)
    valid_counts = np.zeros(num_queries, dtype=np.int64)
    if overlap < min_overlap:
        return scores, valid_counts

    if np.isfinite(prediction).all() and np.isfinite(speaker).all():
        count = float(overlap)
        prediction_sum = prediction.sum(axis=1, dtype=np.float64)
        speaker_sum = speaker.sum(axis=0, dtype=np.float64)
        prediction_square_sum = np.einsum(
            "qtd,qtd->qd",
            prediction,
            prediction,
            dtype=np.float64,
        )
        speaker_square_sum = np.einsum(
            "td,td->d",
            speaker,
            speaker,
            dtype=np.float64,
        )
        cross_sum = np.einsum(
            "qtd,td->qd",
            prediction,
            speaker,
            dtype=np.float64,
        )

        prediction_ss = (
            prediction_square_sum
            - prediction_sum * prediction_sum / count
        )
        speaker_ss = speaker_square_sum - speaker_sum * speaker_sum / count
        # Round-off can make the centered sum of squares slightly negative.
        prediction_ss = np.maximum(prediction_ss, 0.0)
        speaker_ss = np.maximum(speaker_ss, 0.0)
        covariance = cross_sum - prediction_sum * speaker_sum[None, :] / count

        valid = (
            (prediction_ss / count > variance_epsilon)
            & (speaker_ss[None, :] / count > variance_epsilon)
        )
        denominator = np.sqrt(prediction_ss * speaker_ss[None, :])
        correlations = np.zeros(
            (num_queries, num_features),
            dtype=np.float64,
        )
        np.divide(
            covariance,
            denominator,
            out=correlations,
            where=valid,
        )
        correlations = np.clip(correlations, -1.0, 1.0)
        valid_counts = valid.sum(axis=1, dtype=np.int64)
        defined = valid_counts > 0
        scores[defined] = (
            (correlations * valid).sum(axis=1)[defined]
            / valid_counts[defined]
        )
        return scores, valid_counts

    # Slow path for non-finite inputs.  Pairwise finite observations are used
    # independently for each query and feature instead of poisoning the sample.
    for query_index in range(num_queries):
        correlations: List[float] = []
        for feature_index in range(num_features):
            x = prediction[query_index, :, feature_index]
            y = speaker[:, feature_index]
            finite = np.isfinite(x) & np.isfinite(y)
            if int(finite.sum()) < min_overlap:
                continue
            x = x[finite]
            y = y[finite]
            x_centered = x - x.mean(dtype=np.float64)
            y_centered = y - y.mean(dtype=np.float64)
            x_variance = float(np.mean(x_centered * x_centered))
            y_variance = float(np.mean(y_centered * y_centered))
            if (
                x_variance <= variance_epsilon
                or y_variance <= variance_epsilon
            ):
                continue
            denominator = np.sqrt(
                np.sum(x_centered * x_centered)
                * np.sum(y_centered * y_centered)
            )
            correlation = float(
                np.sum(x_centered * y_centered) / denominator
            )
            if np.isfinite(correlation):
                correlations.append(float(np.clip(correlation, -1.0, 1.0)))
        if correlations:
            scores[query_index] = float(np.mean(correlations))
            valid_counts[query_index] = len(correlations)
    return scores, valid_counts


def calculate_tlcc_robust(
    prediction_item: torch.Tensor | np.ndarray,
    speaker_item: torch.Tensor | np.ndarray,
    *,
    seconds: float = 2.0,
    fps: int = 25,
    variance_epsilon: float = 1e-12,
    min_overlap: int = 3,
) -> Dict[str, np.ndarray]:
    """Calculate robust TLCC for every prediction of one example.

    Args:
        prediction_item: Array with shape ``[K, T, D]`` (or ``[T, D]``).
        speaker_item: Array with shape ``[T, D]``.
        seconds: Official lag-window duration.
        fps: Frame rate used to convert the lag window to frames.
        variance_epsilon: Channels with variance at or below this value are
            excluded for the corresponding lag.
        min_overlap: Minimum number of paired frames required at a lag.

    Returns:
        Arrays of absolute offsets, signed selected lags, peak correlations,
        and valid-channel counts. Undefined queries contain NaN offsets.
    """

    prediction = _to_numpy(prediction_item)
    speaker = _to_numpy(speaker_item)
    if prediction.ndim == 2:
        prediction = prediction[None, ...]
    if prediction.ndim != 3 or speaker.ndim != 2:
        raise ValueError(
            "prediction_item must be [K,T,D] or [T,D], and speaker_item "
            "must be [T,D]"
        )
    if prediction.shape[1:] != speaker.shape:
        raise ValueError(
            "prediction/speaker shape mismatch: "
            f"{prediction.shape[1:]} != {speaker.shape}"
        )
    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must be positive")
    if variance_epsilon < 0:
        raise ValueError("variance_epsilon must be non-negative")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")

    max_lag = int(seconds * fps - 1)
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    num_queries = prediction.shape[0]
    lag_scores = np.full(
        (num_queries, len(lags)),
        np.nan,
        dtype=np.float64,
    )
    lag_valid_counts = np.zeros(
        (num_queries, len(lags)),
        dtype=np.int64,
    )

    for lag_index, lag in enumerate(lags):
        shifted_prediction, shifted_speaker = _shift_numpy(
            prediction,
            speaker,
            int(lag),
        )
        scores, valid_counts = _finite_correlations(
            shifted_prediction,
            shifted_speaker,
            variance_epsilon=variance_epsilon,
            min_overlap=min_overlap,
        )
        lag_scores[:, lag_index] = scores
        lag_valid_counts[:, lag_index] = valid_counts

    absolute_offsets = np.full(num_queries, np.nan, dtype=np.float64)
    selected_lags = np.full(num_queries, np.nan, dtype=np.float64)
    peak_correlations = np.full(num_queries, np.nan, dtype=np.float64)
    selected_valid_counts = np.zeros(num_queries, dtype=np.int64)
    for query_index in range(num_queries):
        finite_lags = np.isfinite(lag_scores[query_index])
        if not finite_lags.any():
            continue
        finite_indices = np.flatnonzero(finite_lags)
        relative_peak = int(
            np.argmax(lag_scores[query_index, finite_lags])
        )
        peak_index = int(finite_indices[relative_peak])
        selected_lag = int(lags[peak_index])
        selected_lags[query_index] = float(selected_lag)
        absolute_offsets[query_index] = float(abs(selected_lag))
        peak_correlations[query_index] = float(
            lag_scores[query_index, peak_index]
        )
        selected_valid_counts[query_index] = int(
            lag_valid_counts[query_index, peak_index]
        )

    return {
        "absolute_offsets": absolute_offsets,
        "selected_lags": selected_lags,
        "peak_correlations": peak_correlations,
        "valid_channels_at_peak": selected_valid_counts,
    }


def _calculate_tlcc_robust_worker(
    prediction_item: torch.Tensor | np.ndarray,
    speaker_item: torch.Tensor | np.ndarray,
    seconds: float,
    fps: int,
    variance_epsilon: float,
    min_overlap: int,
) -> Dict[str, np.ndarray]:
    return calculate_tlcc_robust(
        prediction_item,
        speaker_item,
        seconds=seconds,
        fps=fps,
        variance_epsilon=variance_epsilon,
        min_overlap=min_overlap,
    )


def _calculate_tlcc_robust_star(
    arguments: Tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray,
        float,
        int,
        float,
        int,
    ],
) -> Dict[str, np.ndarray]:
    return _calculate_tlcc_robust_worker(*arguments)


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return float(values[finite].mean())


def compute_TLCC_robust(
    predictions: Sequence[torch.Tensor | np.ndarray],
    speakers: Sequence[torch.Tensor | np.ndarray],
    *,
    p: int = 1,
    seconds: float = 2.0,
    fps: int = 25,
    variance_epsilon: float = 1e-12,
    min_overlap: int = 3,
) -> Dict[str, object]:
    """Compute the robust all-query TLCC diagnostic over a dataset.

    The official metric is a scalar.  This diagnostic deliberately returns a
    structured result so callers cannot accidentally substitute it into an
    official result table without acknowledging the changed definition.
    """

    if len(predictions) != len(speakers):
        raise ValueError(
            "predictions and speakers must have the same number of examples"
        )
    if not predictions:
        raise ValueError("predictions and speakers must not be empty")
    if p <= 0:
        raise ValueError("p must be positive")

    worker_args = [
        (
            prediction,
            speaker,
            seconds,
            fps,
            variance_epsilon,
            min_overlap,
        )
        for prediction, speaker in zip(predictions, speakers)
    ]
    if p == 1:
        sample_results = [
            _calculate_tlcc_robust_worker(*arguments)
            for arguments in worker_args
        ]
    else:
        # NumPy releases the GIL for the heavy reductions below. Threads keep
        # mmap-backed tensors shared and avoid serializing multi-GB caches into
        # worker processes.
        with ThreadPoolExecutor(max_workers=p) as pool:
            sample_results = list(
                pool.map(_calculate_tlcc_robust_star, worker_args)
            )

    query_counts = [
        len(result["absolute_offsets"])
        for result in sample_results
    ]
    max_queries = max(query_counts)
    per_query_offsets: List[List[float]] = [
        [] for _ in range(max_queries)
    ]
    all_offsets: List[float] = []
    oracle_offsets: List[float] = []
    selected_valid_counts: List[int] = []
    per_query_boundary_rates: List[float | None] = []
    max_lag = int(seconds * fps - 1)

    for result in sample_results:
        offsets = result["absolute_offsets"]
        valid_counts = result["valid_channels_at_peak"]
        finite = np.isfinite(offsets)
        all_offsets.extend(offsets[finite].tolist())
        selected_valid_counts.extend(valid_counts[finite].tolist())
        if finite.any():
            oracle_offsets.append(float(np.min(offsets[finite])))
        for query_index, offset in enumerate(offsets):
            if np.isfinite(offset):
                per_query_offsets[query_index].append(float(offset))

    per_query_means = [
        float(np.mean(offsets)) if offsets else None
        for offsets in per_query_offsets
    ]
    for offsets in per_query_offsets:
        if not offsets:
            per_query_boundary_rates.append(None)
        else:
            array = np.asarray(offsets, dtype=np.float64)
            per_query_boundary_rates.append(
                float(np.mean(array == max_lag))
            )

    total_pairs = int(sum(query_counts))
    defined_pairs = len(all_offsets)
    all_offsets_array = np.asarray(all_offsets, dtype=np.float64)
    valid_counts_array = np.asarray(selected_valid_counts, dtype=np.float64)
    return {
        "FRSyn_robust": _finite_mean(all_offsets_array),
        "mean_all_queries": _finite_mean(all_offsets_array),
        "q0_mean": per_query_means[0],
        "oracle_best_of_k_mean": _finite_mean(
            np.asarray(oracle_offsets, dtype=np.float64)
        ),
        "per_query_means": per_query_means,
        "boundary_rate_all_queries": (
            float(np.mean(all_offsets_array == max_lag))
            if defined_pairs
            else None
        ),
        "per_query_boundary_rates": per_query_boundary_rates,
        "mean_valid_channels_at_selected_lag": _finite_mean(
            valid_counts_array
        ),
        "num_examples": len(sample_results),
        "num_predictions_min": min(query_counts),
        "num_predictions_max": max(query_counts),
        "defined_prediction_pairs": defined_pairs,
        "undefined_prediction_pairs": total_pairs - defined_pairs,
        "undefined_examples": int(
            sum(
                not np.isfinite(result["absolute_offsets"]).any()
                for result in sample_results
            )
        ),
        "seconds": float(seconds),
        "fps": int(fps),
        "max_lag_frames": max_lag,
        "variance_epsilon": float(variance_epsilon),
        "min_overlap": int(min_overlap),
        "candidate_policy": "mean over every defined sample-query pair",
        "constant_channel_policy": (
            "exclude zero/near-zero-variance and non-finite channel-lag "
            "correlations"
        ),
        "official_metric_unchanged": True,
        "lower_is_better": True,
    }
