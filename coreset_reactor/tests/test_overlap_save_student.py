import torch

from coreset_reactor.eval_overlap_save_student import (
    overlap_save,
    powered_center_overlap,
)
from coreset_reactor.query0_au_transition import stabilize_query0_au_transitions
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    center_weights,
    overlap_add,
)


def test_single_window_is_exact_identity():
    prediction = torch.rand(1, 10, WINDOW, 25)
    result, selected, switches = overlap_save(prediction, [0], WINDOW)
    assert torch.equal(result, prediction[0])
    assert torch.equal(selected, torch.zeros(WINDOW, dtype=torch.long))
    assert switches == []


def test_each_frame_uses_largest_center_weight_without_averaging():
    length = WINDOW + 375
    predictions = torch.stack((
        torch.zeros(10, WINDOW, 25),
        torch.ones(10, WINDOW, 25),
    ))
    result, selected, switches = overlap_save(predictions, [0, 375], length)
    weights = center_weights(WINDOW)
    expected = torch.zeros(length, dtype=torch.long)
    for frame in range(length):
        candidates = []
        for index, start in enumerate((0, 375)):
            if start <= frame < start + WINDOW:
                candidates.append((float(weights[frame - start]), -index, index))
        expected[frame] = max(candidates)[2]
    assert torch.equal(selected, expected)
    assert torch.equal(result[:, :, 0], expected[None].expand(10, -1))
    assert switches == [int((expected[1:] != expected[:-1]).nonzero()[0]) + 1]


def test_power_one_matches_original_center_weighting():
    torch.manual_seed(4)
    predictions = torch.rand(2, 10, WINDOW, 25)
    starts = [0, 375]
    expected = overlap_add(predictions, starts, WINDOW + 375, "center")
    actual = powered_center_overlap(predictions, starts, WINDOW + 375, 1)
    torch.testing.assert_close(actual, expected)


def test_query0_transition_changes_each_constant_au_once():
    length = 12
    prediction = torch.zeros(10, length, 25)
    ramp = torch.linspace(0, 1, length)
    prediction[:, :, 2:] = ramp[None, :, None]
    prediction[0, :, 1] = 1
    source = torch.stack([
        ramp + channel / 100 for channel in range(25)
    ], dim=1)
    source[:, 0] = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    source[:, 1] = torch.tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])

    result = stabilize_query0_au_transitions(prediction, source, margin=2)

    difference = result != prediction
    assert int(difference[0, :, 0].sum()) == 1
    assert int(difference[0, :, 1].sum()) == 1
    assert not difference[1:].any()
    assert not difference[:, :, 2:].any()
    assert result[0, 3, 0] == 1
    assert result[0, 5, 1] == 0


def test_query0_transition_is_noop_when_source_has_constant_channel():
    prediction = torch.zeros(10, 12, 25)
    source = torch.arange(12, dtype=torch.float32)[:, None].repeat(1, 25)
    source[:, 7] = 0
    result = stabilize_query0_au_transitions(prediction, source, margin=2)
    assert torch.equal(result, prediction)
