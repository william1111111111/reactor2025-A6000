"""Audit that A2 changes targets while preserving every input view and slot ID."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


VIEW_KEYS = (
    "sample_index", "sample_id", "view_id", "view_epoch", "start", "length",
    "generated_length", "teacher_seed",
)
BUILDER_VIEW_KEYS = VIEW_KEYS[:-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = json.loads(args.teacher_manifest.read_text())
    direct = json.loads(args.direct_manifest.read_text())
    shared_manifest_fields = (
        "split", "contexts", "source_contexts", "num_experts", "view_epoch",
        "source_seed", "clip_length", "crop_views", "crop_view_mode",
        "crop_view_policy", "source_length_counts", "sampler_config_hash",
        "model_config", "shards",
    )
    manifest_checks = {
        key: teacher[key] == direct[key] for key in shared_manifest_fields
    }
    manifest_checks["source_manifest_sha256"] = (
        direct["source_view_manifest_sha256"] == sha256(args.teacher_manifest)
    )
    manifest_checks["target_kind"] = direct["target_kind"] == "real_gt"
    manifest_checks["production_target_parity_examples"] = (
        direct["production_target_parity_examples"] == 300
    )
    if not all(manifest_checks.values()):
        raise ValueError(f"A2 manifest audit failed: {manifest_checks}")

    contexts = 0
    metadata_digest = hashlib.sha256()
    direct_digest = hashlib.sha256()
    for shard in teacher["shards"]:
        teacher_rows = torch.load(
            args.teacher_manifest.parent / shard, map_location="cpu", weights_only=False)
        direct_rows = torch.load(
            args.direct_manifest.parent / shard, map_location="cpu", weights_only=False)
        if len(teacher_rows) != len(direct_rows):
            raise ValueError(f"Shard length differs: {shard}")
        for teacher_row, direct_row in zip(teacher_rows, direct_rows):
            left = [teacher_row[key] for key in VIEW_KEYS]
            right = [direct_row[key] for key in VIEW_KEYS]
            if left != right:
                raise ValueError(f"View metadata differs at context {contexts}")
            target = direct_row["teacher_predictions"]
            if target.shape != teacher_row["teacher_predictions"].shape:
                raise ValueError(f"Target tensor shape differs at context {contexts}")
            if direct_row["target_kind"] != "real_gt":
                raise ValueError(f"A2 target kind differs at context {contexts}")
            identities = direct_row["fixed_gt_ids"]
            if len(identities) != direct["num_experts"] or len(set(identities)) != len(identities):
                raise ValueError(f"Invalid fixed target identities at context {contexts}")
            stored_view = [direct_row[key] for key in BUILDER_VIEW_KEYS]
            encoded = json.dumps(stored_view, separators=(",", ":")).encode()
            metadata_digest.update(encoded)
            direct_digest.update(json.dumps(left, separators=(",", ":")).encode())
            direct_digest.update(json.dumps(identities, separators=(",", ":")).encode())
            contexts += 1
    if contexts != teacher["contexts"]:
        raise ValueError("Incomplete A2 cache audit")
    if metadata_digest.hexdigest() != direct["view_audit_sha256"]:
        raise ValueError("A2 stored view digest does not match audited rows")
    result = {
        "status": "passed",
        "statement": (
            "A2 preserves all 8,820 input contexts, view starts, valid lengths, "
            "slot count and ordered fixed target IDs; only the supervision tensor "
            "changes from teacher AFR to its preassigned real reference AFR."
        ),
        "contexts": contexts,
        "slots": direct["num_experts"],
        "manifest_checks": manifest_checks,
        "view_metadata_sha256": metadata_digest.hexdigest(),
        "view_and_target_id_sha256": direct_digest.hexdigest(),
        "teacher_manifest_sha256": sha256(args.teacher_manifest),
        "direct_manifest_sha256": sha256(args.direct_manifest),
        "teacher_cache_hash": teacher["cache_manifest_hash"],
        "direct_cache_hash": direct["cache_manifest_hash"],
        "target_alignment": direct["target_alignment"],
        "production_target_parity_examples": direct["production_target_parity_examples"],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
