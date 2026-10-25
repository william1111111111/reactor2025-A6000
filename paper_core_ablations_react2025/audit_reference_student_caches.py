"""Audit fixed/uniform student caches against the target-specific source views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from paper_core_ablations_react2025.assemble_a0_a1_results import sha256, write_json


ROW_KEYS = ("sample_index", "sample_id", "view_id", "view_epoch", "start",
            "length", "generated_length")
MANIFEST_KEYS = ("split", "contexts", "source_contexts", "num_experts",
                 "view_epoch", "source_seed", "clip_length", "crop_views",
                 "crop_view_mode", "crop_view_policy", "sampler_config_hash",
                 "preprocessing_version", "model_config")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def row_digest(path: Path, manifest: dict) -> str:
    digest = hashlib.sha256()
    count = 0
    for shard in manifest["shards"]:
        for row in torch.load(path.parent / shard, map_location="cpu", weights_only=False):
            digest.update(json.dumps([row[key] for key in ROW_KEYS],
                                     separators=(",", ":")).encode())
            count += 1
    if count != manifest["contexts"]:
        raise ValueError(f"Cache row count mismatch: {path}")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--uniform-manifest", type=Path, required=True)
    parser.add_argument("--bank-audit", type=Path, required=True)
    parser.add_argument("--a0-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = {
        "fixed": read(args.fixed_manifest),
        "uniform": read(args.uniform_manifest),
        "target_specific": read(args.target_manifest),
    }
    bank_audit = read(args.bank_audit)
    if bank_audit["status"] != "passed":
        raise ValueError("Teacher-bank audit has not passed")
    checks = {
        key: len({json.dumps(value[key], sort_keys=True)
                  for value in manifests.values()}) == 1
        for key in MANIFEST_KEYS
    }
    checks["formal_multiview_population"] = (
        manifests["fixed"]["contexts"] == 8820
        and manifests["fixed"]["source_contexts"] == 3320
        and manifests["fixed"]["crop_views"] == 3
        and manifests["fixed"]["crop_view_mode"] == "stratified"
    )
    for name, manifest in manifests.items():
        checks[f"{name}_teacher_hashes"] = (
            manifest["teacher_checkpoint_hashes"]
            == bank_audit["banks"][name]["checkpoint_hashes"]
        )
    row_hashes = {name: row_digest(path, manifests[name]) for name, path in {
        "fixed": args.fixed_manifest,
        "uniform": args.uniform_manifest,
        "target_specific": args.target_manifest,
    }.items()}
    checks["identical_ordered_source_views"] = len(set(row_hashes.values())) == 1
    if not all(checks.values()):
        raise ValueError(f"Reference student-cache audit failed: {checks}")

    a0 = read(args.a0_provenance)
    calibration = a0["relation_calibration"]
    if calibration["objective"] != "centered_residual":
        raise ValueError("Expected A0 centered-residual calibration")
    args.calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_files = {}
    for name in ("fixed", "uniform"):
        payload = {
            "cache_manifest_hash": manifests[name]["cache_manifest_hash"],
            "initialization_checkpoint_hash": a0["initialization_checkpoint_hash"],
            "initial_student_state_sha256": a0["initial_student_state_sha256"],
            "relation_calibration": {
                "objective": "centered_residual",
                "scale": calibration["scale"],
                "weight": calibration["weight"],
                "source": "frozen A0 target-specific seed-1 TRAIN calibration",
                "reason": "keep the numerical student loss identical across teacher banks",
            },
        }
        path = args.calibration_dir / f"{name}_A0_frozen_calibration.json"
        write_json(path, payload)
        calibration_files[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    result = {
        "status": "passed", "checks": checks, "ordered_view_hashes": row_hashes,
        "manifests": {name: {
            "path": str(path.resolve()), "sha256": sha256(path),
            "cache_manifest_hash": manifests[name]["cache_manifest_hash"],
        } for name, path in {
            "fixed": args.fixed_manifest, "uniform": args.uniform_manifest,
            "target_specific": args.target_manifest,
        }.items()},
        "shared_student_loss": {
            "objective": "centered_residual",
            "scale": calibration["scale"], "weight": calibration["weight"],
            "velocity_weight": 0.05,
            "calibration_files": calibration_files,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
