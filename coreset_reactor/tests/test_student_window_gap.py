import torch

from coreset_reactor.eval_student_window_gap import (
    center_weights,
    overlap_add,
    position_starts,
    window_starts,
)


def test_window_starts_cover_and_align_tail():
    assert window_starts(245) == [0]
    assert window_starts(750) == [0]
    assert window_starts(751) == [0, 1]
    assert window_starts(1500) == [0, 375, 750]
    assert window_starts(1501) == [0, 375, 750, 751]


def test_position_starts():
    assert position_starts(245) == {"head": 0, "center": 0, "tail": 0}
    assert position_starts(750) == {"head": 0, "center": 0, "tail": 0}
    assert position_starts(751) == {"head": 0, "center": 0, "tail": 1}
    assert position_starts(1500) == {"head": 0, "center": 375, "tail": 750}


def test_center_weights_are_positive_and_symmetric():
    weights = center_weights(750)
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights, weights.flip(0))


def test_overlap_add_restores_identical_windows():
    starts = [0, 375, 750]
    windows = torch.full((3, 10, 750, 25), 0.25)
    for weighting in ("uniform", "center"):
        result = overlap_add(windows, starts, 1500, weighting)
        torch.testing.assert_close(result, torch.full_like(result, 0.25))


def test_overlap_add_preserves_expert_identity():
    starts = [0, 375]
    windows = torch.zeros((2, 10, 750, 25))
    for expert in range(10):
        windows[:, expert] = expert
    result = overlap_add(windows, starts, 1125, "center")
    for expert in range(10):
        torch.testing.assert_close(result[expert], torch.full_like(result[expert], expert))
