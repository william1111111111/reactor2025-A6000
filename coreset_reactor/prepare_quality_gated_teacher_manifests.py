"""Export selected quality-gated target arms into rank-wise Mam manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def export_rank_manifests(
    source: Path,
    output_dir: Path,
    arms: tuple[str, ...],
) -> dict:
    source = source.resolve()
    payload = json.loads(source.read_text())
    if payload.get("split") != "train":
        raise ValueError("quality-gated source manifest must be TRAIN-only")
    records = payload.get("records", [])
    if not records:
        raise ValueError("source manifest has no records")
    for arm in arms:
        if any(arm not in record for record in records):
            raise ValueError(f"source manifest is missing arm {arm}")
        if any(len(record[arm]["target_ids"]) != 10
               or record[arm]["target_ids"][0] != record["paired_target_id"]
               for record in records):
            raise ValueError(f"arm {arm} does not preserve paired rank zero")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_bytes(source.read_bytes())
    result = {
        "source_manifest_path": str(source),
        "source_manifest_file_sha256": source_sha,
        "source_manifest_declared_sha256": payload.get("manifest_sha256"),
        "arms": list(arms),
        "contexts": len(records),
        "target_set_summary": {
            arm: payload["summary"][arm] for arm in arms
        },
        "ranks": {},
    }
    for arm in arms:
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        mapping = {
            record["sample_id"]: record[arm]["target_ids"]
            for record in records
        }
        for rank in range(10):
            manifest = {
                "schema_version": 1,
                "preprocessing_version": "quality-gated-maxfrdiv-teacher-v1",
                "split": "train",
                "assignment_method": record_method(payload, arm),
                "assignment_seed": payload["seed"],
                "k": 10,
                "preserve_native_pair": True,
                "alignment_policy": "relative_time_masked",
                "fixed_pair_alignment_policy": "relative_time_masked",
                "source_manifest_file_sha256": source_sha,
                "source_manifest_declared_sha256": payload.get("manifest_sha256"),
                "target_set_FRDiv": payload["summary"][arm]["FRDiv"],
                "target_selection_arm": arm,
                "rank": rank,
                "mapping": mapping,
                "contexts": len(mapping),
            }
            canonical = json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
            manifest["manifest_hash"] = sha256_bytes(canonical)
            path = arm_dir / f"rank_{rank:02d}.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            result["ranks"][f"{arm}/rank_{rank:02d}"] = {
                "path": str(path),
                "manifest_sha256": sha256_bytes(path.read_bytes()),
                "declared_manifest_hash": manifest["manifest_hash"],
            }
    (output_dir / "manifest_index.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def record_method(payload: dict, arm: str) -> str:
    if arm in {"T0", "T1"}:
        return arm
    return f"quality_gated_{arm.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    result = export_rank_manifests(args.source, args.output_dir, tuple(args.arms))
    print(json.dumps({
        "source_manifest_file_sha256": result["source_manifest_file_sha256"],
        "arms": result["arms"],
        "contexts": result["contexts"],
        "manifests": len(result["ranks"]),
    }, indent=2))


if __name__ == "__main__":
    main()
