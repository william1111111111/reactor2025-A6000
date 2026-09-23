"""Build a frozen TRAIN-only speaker-to-expert target assignment manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


PREPROCESSING_VERSION = "fixed-pair-v1"


def stable_rng(seed: int, sample_id: str) -> random.Random:
    payload = f"{seed}|{sample_id}|fixed-random-assignment".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def assign_fixed_random(
    candidate_ids: list[str],
    native_pair_id: str,
    sample_id: str,
    assignment_seed: int,
    k: int,
    preserve_native_pair: bool = True,
) -> tuple[list[str], list[bool]]:
    """Choose K stable targets, sampling without replacement when possible."""
    if k < 1:
        raise ValueError("k must be positive")
    candidates = sorted(set(candidate_ids))
    if not candidates:
        raise ValueError(f"No legal candidates for {sample_id}")
    if native_pair_id not in candidates:
        raise ValueError(f"Native pair is absent for {sample_id}")
    rng = stable_rng(assignment_seed, sample_id)
    selected = [native_pair_id] if preserve_native_pair else []
    pool = [value for value in candidates if value not in selected]
    rng.shuffle(pool)
    selected.extend(pool[: k - len(selected)])
    while len(selected) < k:
        # Repeats are deterministic and explicit; they are not extra modes.
        selected.append(candidates[rng.randrange(len(candidates))])
    seen: set[str] = set()
    duplicate = []
    for target in selected:
        duplicate.append(target in seen)
        seen.add(target)
    return selected, duplicate


def array_length(path: Path) -> int:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2 or value.shape[1] != 25:
        raise ValueError(f"Expected [T,25] facial attributes: {path}: {value.shape}")
    if value.shape[0] <= 0:
        raise ValueError(f"Zero-length facial attributes: {path}")
    return int(value.shape[0])


def build_manifest(
    data_root: Path,
    split: str,
    assignment_seed: int,
    k: int,
    preserve_native_pair: bool,
) -> dict:
    if split != "train":
        raise ValueError("Fixed training assignments must be built from train only")
    facial = data_root / split / "facial-attributes"
    paths = sorted(facial.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"No facial attributes under {facial}")
    ids = [path.relative_to(facial).as_posix() for path in paths]
    lookup = dict(zip(ids, paths))
    sessions: dict[tuple[str, str], list[str]] = {}
    for sample_id in ids:
        role, session, _ = Path(sample_id).parts
        sessions.setdefault((role, session), []).append(sample_id)
    lengths = {sample_id: array_length(path) for sample_id, path in lookup.items()}
    mapping, records = {}, []
    duplicate_slots = 0
    for sample_id in ids:
        role, session, basename = Path(sample_id).parts
        opposite = "listener" if role == "speaker" else "speaker"
        native = f"{opposite}/{session}/{basename}"
        candidates = sorted(sessions.get((opposite, session), []))
        targets, duplicate = assign_fixed_random(
            candidates, native, sample_id, assignment_seed, k,
            preserve_native_pair,
        )
        mapping[sample_id] = targets
        duplicate_slots += sum(duplicate)
        records.append({
            "sample_id": sample_id,
            "split": split,
            "session_id": session,
            "speaker_role": role,
            "speaker_id_if_available": None,
            "native_pair_id": native,
            "speaker_length": lengths[sample_id],
            "candidate_ids": candidates,
            "candidate_lengths": [lengths[value] for value in candidates],
            "assignments": [{
                "expert_id": expert_id,
                "target_id": target_id,
                "target_length": lengths[target_id],
                "assignment_method": "fixed_random",
                "is_native_pair": target_id == native,
                "duplicate_target": duplicate[expert_id],
                "alignment_policy": "relative_time_masked",
            } for expert_id, target_id in enumerate(targets)],
        })
    manifest = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "split": split,
        "assignment_method": "fixed_random",
        "assignment_seed": assignment_seed,
        "k": k,
        "preserve_native_pair": preserve_native_pair,
        "alignment_policy": "relative_time_masked",
        "contexts": len(records),
        "candidate_universe_size": len(ids),
        "duplicate_slots": duplicate_slots,
        "mapping": mapping,
        "records": records,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--assignment-seed", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-native-pair", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(
        args.data_root.resolve(), args.split, args.assignment_seed, args.k,
        not args.no_native_pair,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "contexts", "candidate_universe_size", "duplicate_slots",
        "manifest_hash",
    )}, indent=2))


if __name__ == "__main__":
    main()
