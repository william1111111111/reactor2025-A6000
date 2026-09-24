"""Convert the frozen T0/T1 target-set manifest into rank-wise Mam manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.dataset import full_reaction_descriptor
from coreset_reactor.dataset_gt_oracle import official_distance_matrix
from coreset_reactor.fixed_pair_manifest import assign_fixed_random
from coreset_reactor.maxfrdiv_teacher_manifest import (
    _assign_canonical_slots,
    _load_raw,
    _relative_resample,
    paired_farthest_indices,
)

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _augment_reverse_role(payload: dict, data_root: Path) -> dict[str, list[str]]:
    """Add listener->speaker mappings without changing frozen speaker mappings."""
    facial = data_root.resolve() / "train" / "facial-attributes"
    descriptor_mean = np.asarray(payload["descriptor_mean"], dtype=np.float32)
    descriptor_std = np.asarray(payload["descriptor_std"], dtype=np.float32)
    centers = np.asarray(payload["prototype_centers"], dtype=np.float32)
    mapping_by_arm = {
        arm: {record["sample_id"]: list(record[arm]["target_ids"])
              for record in payload["records"]}
        for arm in ("T0", "T1")
    }
    target_by_session: dict[str, list[Path]] = {}
    source_paths = sorted((facial / "listener").glob("*/*.npy"))
    for path in sorted((facial / "speaker").glob("*/*.npy")):
        target_by_session.setdefault(path.parent.name, []).append(path)
    target_cache = {}
    for session, target_paths in target_by_session.items():
        if len(target_paths) < 10:
            continue
        target_ids = [path.relative_to(facial).as_posix() for path in target_paths]
        raw = [_load_raw(path) for path in target_paths]
        reactions = torch.stack([_relative_resample(value, int(payload["frames"]))
                                 for value in raw])
        distance = official_distance_matrix(reactions)
        descriptors = np.stack([
            full_reaction_descriptor(value).numpy() for value in raw
        ])
        values = (descriptors - descriptor_mean) / descriptor_std
        target_cache[session] = (
            target_ids, distance, values,
        )
    for source in source_paths:
        session = source.parent.name
        if session not in target_cache:
            continue
        target_ids, distance, values = target_cache[session]
        lookup = {value: index for index, value in enumerate(target_ids)}
        paired_id = f"speaker/{session}/{source.name}"
        paired_index = lookup[paired_id]
        source_id = source.relative_to(facial).as_posix()
        t0_ids, _ = assign_fixed_random(
            target_ids, paired_id, source_id, int(payload["seed"]), 10, True
        )
        t1_indices = paired_farthest_indices(distance, paired_index, 10)
        t1_indices = _assign_canonical_slots(
            t1_indices, paired_index, values, centers
        )
        mapping_by_arm["T0"][source_id] = t0_ids
        mapping_by_arm["T1"][source_id] = [target_ids[index] for index in t1_indices]
    return mapping_by_arm


def export_rank_manifests(source: Path, output_dir: Path,
                          data_root: Path | None = None) -> dict:
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
    mapping_by_arm = {
        arm: {
            record["sample_id"]: record[arm]["target_ids"]
            for record in records
        }
        for arm in ("T0", "T1")
    }
    if data_root is not None:
        reverse = _augment_reverse_role(payload, data_root)
        mapping_by_arm = reverse
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
                "mapping": mapping_by_arm[arm],
                "contexts": len(mapping_by_arm[arm]),
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
    parser.add_argument("--data-root", type=Path,
                        help="Add the unchanged algorithm's reverse-role mappings")
    parser.add_argument("--mam-root", type=Path,
                        help="Mam-Reactor source root for relative-time resampling")
    args = parser.parse_args()
    if args.mam_root is not None:
        import sys
        sys.path.insert(0, str(args.mam_root.resolve()))
    result = export_rank_manifests(args.source, args.output_dir, args.data_root)
    print(json.dumps({
        "source_manifest_file_sha256": result["source_manifest_file_sha256"],
        "target_set_summary": result["target_set_summary"],
        "manifests": len(result["ranks"]),
    }, indent=2))


if __name__ == "__main__":
    main()
