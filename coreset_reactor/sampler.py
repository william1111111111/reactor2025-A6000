"""Fixed random exact-target draws shared by B0/B1/B3."""

from __future__ import annotations

import hashlib
import random


def exact_reference_indices(reference_ids: list[int], paired_id: int,
                            count: int, seed: int, step: int,
                            source_id: str) -> list[int]:
    """Return `count-1` unpaired IDs; the paired crop is supplied separately."""
    if count < 1:
        raise ValueError("exact target count must be positive")
    alternatives = [idx for idx in reference_ids if idx != paired_id]
    if len(alternatives) < count - 1:
        raise ValueError("not enough distinct session GT references")
    digest = hashlib.sha256(f"{seed}|{step}|{source_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return rng.sample(alternatives, count - 1)
