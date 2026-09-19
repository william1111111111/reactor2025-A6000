import sys
from pathlib import Path

MAM_ROOT = Path(__file__).resolve().parents[2] / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from dataset.react_2025 import fixed_target_choices


def test_fixed_diffusion_target_choices_are_deterministic_and_distinct():
    candidates = [Path("listener/session1") / f"gt_{index:03d}"
                  for index in range(96)]
    source = Path("speaker/session1/source_007")
    first = fixed_target_choices(candidates, source, seed=20260918, count=10)
    second = fixed_target_choices(list(reversed(candidates)), source,
                                  seed=20260918, count=10)
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert first != fixed_target_choices(candidates, source, seed=20260919,
                                         count=10)
