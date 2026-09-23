import numpy as np

from coreset_reactor.quality_gated_maxfrdiv import (
    _select_gated,
    eligible_indices,
)


def test_quantile_gate_has_minimum_alternatives_and_records_widening():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    result = eligible_indices(scores, paired_index=0, quantile=.30)
    assert result["eligible_indices"] == [1, 2, 3, 4, 5, 6, 7, 8, 9][:9]
    assert result["eligible_count"] == 9
    assert result["minimum_count_widened"]
    assert result["effective_quantile"] == 1.0


def test_quantile_gate_is_deterministic_on_ties():
    scores = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    left = eligible_indices(scores, paired_index=0, quantile=.30)
    right = eligible_indices(scores, paired_index=0, quantile=.30)
    assert left == right
    assert left["eligible_indices"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_gated_selection_keeps_paired_target_and_selects_only_legal_pool():
    points = np.arange(12 * 3, dtype=np.float64).reshape(12, 3)
    distance = ((points[:, None] - points[None]) ** 2).mean(2)
    values = points.copy()
    selected = _select_gated(
        distance, paired_index=0, legal_alternatives=list(range(1, 10)),
        values=values, centers=values[1:10].copy(),
    )
    assert selected[0] == 0
    assert len(selected) == len(set(selected)) == 10
    assert set(selected) <= set(range(10))
