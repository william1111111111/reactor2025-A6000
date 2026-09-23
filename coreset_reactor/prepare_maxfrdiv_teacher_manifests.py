"""Convert the frozen T0/T1 target-set manifest into rank-wise Mam manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def export_rank_manifests(source: Path, output_dir: Path) -> dict:
    source = source.resolve()
    payload = json.loads(source.read_text())
    if payload.get("split") != "train" or payload.get("source_role") != "speaker":
        raise ValueError("source manifest must be the TRAIN speaker T0/T1 manifest")
    records = payload["records"]
    if not records or any(len(record["T0"]["target_ids"]) != 10
                          or len(record["T1"]["target_ids"]) != 10
                          for record in records):
        raise ValueError("every context must provide ten T0 and T1 targets")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "source_manifest_path": str(source),
        "source_manifest_file_sha256": sha256_bytes(source.read_bytes()),
        "source_manifest_declared_sha256": payload.get("manifest_sha256"),
        "target_set_summary": {
            name: payload["summary"][name] for name in ("T0", "T1", "T2")
        },
        "ranks": {},
    }
    for arm in ("T0", "T1"):
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for rank in range(10):
            mapping = {
                record["sample_id"]: record[arm]["target_ids"]
                for record in records
            }
            manifest = {
                "schema_version": 1,
                "preprocessing_version": "maxfrdiv-teacher-v1",
                "split": "train",
                "assignment_method": arm,
                "assignment_seed": payload["seed"],
                "k": 10,
                "preserve_native_pair": True,
                "alignment_policy": "relative_time_masked",
                "fixed_pair_alignment_policy": "relative_time_masked",
                "source_manifest_file_sha256": result["source_manifest_file_sha256"],
                "source_manifest_declared_sha256": payload.get("manifest_sha256"),
                "target_set_FRDiv": payload["summary"][arm]["FRDiv"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_rank_manifests(args.source, args.output_dir)
    print(json.dumps({
        "source_manifest_file_sha256": result["source_manifest_file_sha256"],
        "target_set_summary": result["target_set_summary"],
        "manifests": len(result["ranks"]),
    }, indent=2))


if __name__ == "__main__":
    main()
