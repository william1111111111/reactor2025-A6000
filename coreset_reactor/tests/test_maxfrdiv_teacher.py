import numpy as np

from coreset_reactor.maxfrdiv_teacher_manifest import (
    _assign_canonical_slots,
    paired_farthest_indices,
)
from coreset_reactor.dataset_gt_oracle import frdiv, official_distance_matrix


def test_official_frdiv_matches_ordered_pair_formula():
    values = np.arange(10 * 4 * 25, dtype=np.float64).reshape(10, 4, 25)
    distance = official_distance_matrix(__import__("torch").from_numpy(values.astype("float32")))
    selected = list(range(10))
    expected = float(distance.sum() / (10 * 9))
    assert np.isclose(frdiv(distance, selected), expected)


def test_paired_farthest_keeps_anchor_and_distinct_targets():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(24, 5))
    distance = ((points[:, None] - points[None]) ** 2).mean(2)
    selected = paired_farthest_indices(distance, paired_index=3, count=10)
    assert selected[0] == 3
    assert len(selected) == len(set(selected)) == 10


def test_canonical_slots_keep_paired_rank_zero():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(20, 6))
    centers = values[:9].copy()
    selected = [4, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ordered = _assign_canonical_slots(selected, 4, values, centers)
    assert ordered[0] == 4
    assert len(ordered) == len(set(ordered)) == 10
