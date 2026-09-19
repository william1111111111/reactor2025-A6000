import sys
from pathlib import Path

import torch
import pytest

MAM_ROOT = Path(__file__).resolve().parents[2] / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_data import (
    ConditionalReactionDataset,
    diffusion_listener_crop,
)


def _synthetic_dataset(rank: int) -> ConditionalReactionDataset:
    dataset = ConditionalReactionDataset.__new__(ConditionalReactionDataset)
    dataset.seed = 1
    dataset.fixed_pair_seed = 1
    dataset.fixed_pair_rank = rank
    dataset.fixed_pair_count = 10
    dataset.training = True
    dataset._epoch = torch.zeros((), dtype=torch.long)
    targets = [Path("listener/session1") / f"gt_{index:03d}.npy"
               for index in range(96)]
    targets.append(Path("listener/session1/source.npy"))
    dataset.session_targets = {("listener", "session1"): targets}
    return dataset


def test_anchor_fixed_pair_rank_zero_is_true_pair_and_others_are_distinct():
    record = Path("speaker/session1/source.npy")
    selections = [dataset._fixed_pair_targets(record)
                  for dataset in (_synthetic_dataset(rank) for rank in range(10))]
    assert all(selection == selections[0] for selection in selections)
    assert selections[0][0] == Path("listener/session1/source.npy")
    assert len(selections[0]) == len(set(selections[0])) == 10


def test_diffusion_listener_crop_uses_speaker_tail_for_short_target():
    target = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    speaker = torch.arange(100, 110, dtype=torch.float32).reshape(5, 2)

    value = diffusion_listener_crop(target, speaker, start=1, length=5)

    expected = torch.cat((target[1:], speaker[-3:]), dim=0)
    assert torch.equal(value, expected)


def test_fixed_pair_rank_validation_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="fixed_pair_count must be positive"):
        ConditionalReactionDataset(
            "/unused", "train", fixed_pair_count=0
        )
    with pytest.raises(ValueError, match="target_mode=paired"):
        ConditionalReactionDataset(
            "/unused", "train", target_mode="session",
            fixed_pair_rank=0,
        )
