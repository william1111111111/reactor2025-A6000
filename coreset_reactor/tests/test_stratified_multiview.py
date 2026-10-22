from pathlib import Path

import torch

from regnn.conditional_data import ConditionalReactionDataset


def dataset(mode="stratified", count=3):
    value = object.__new__(ConditionalReactionDataset)
    value.clip_length = 750
    value.crop_stride = 1
    value.training = True
    value.crop_view_mode = mode
    value.crop_view_count = count
    value.seed = 1
    value._epoch = torch.tensor(0)
    return value


def test_three_stratified_views_are_head_center_tail():
    value = dataset()
    starts = [value._choose_start(Path("speaker/session0/example.npy"), 2001, i)
              for i in range(3)]
    assert starts == [0, 625, 1251]


def test_short_source_has_one_effective_view():
    value = dataset()
    assert value._choose_start(Path("speaker/session0/example.npy"), 749, 0) == 0


def test_stratified_view_range_is_validated():
    value = dataset()
    try:
        value._choose_start(Path("speaker/session0/example.npy"), 2000, 3)
    except ValueError as error:
        assert "out of range" in str(error)
    else:
        raise AssertionError("out-of-range view did not fail")


def test_historical_random_view_zero_formula_is_preserved():
    value = dataset(mode="random", count=1)
    record = Path("speaker/session0/example.npy")
    expected = value._stable_random(
        record, purpose="source-crop", include_epoch=True, view_id=0
    ).randint(0, 1250)
    assert value._choose_start(record, 2000, 0) == expected
