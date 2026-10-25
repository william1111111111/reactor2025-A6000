"""Audit the matched fixed, uniform, and target-specific teacher banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_core_ablations_react2025.assemble_a0_a1_results import sha256, write_json


def combined_hash(paths: list[Path]) -> str:
    hashes = [sha256(path) for path in paths]
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    fixed_root = root / "coreset_reactor/reports/teacher_convergence_2400_seed1/arms/T1"
    uniform_root = root / "coreset_reactor/reports/react2025_key_component_ablations_seed1/uniform"
    target_root = root / "coreset_reactor/reports/mode_reference_teacher_v2_seed1"
    manifest_root = root / "coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3/manifests/T1"
    q0 = fixed_root / "rank_00/conditional-step000600-seed1.pth"
    banks = {
        "fixed": [fixed_root / f"rank_{rank:02d}/conditional-step000600-seed1.pth"
                  for rank in range(10)],
        "uniform": [q0] + [uniform_root / f"rank_{rank:02d}/conditional-step000600-seed1.pth"
                            for rank in range(1, 10)],
        "target_specific": [q0] + [target_root / f"rank_{rank:02d}/conditional-step000600-seed1.pth"
                                    for rank in range(1, 10)],
    }
    for name, paths in banks.items():
        if len(paths) != 10 or not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Incomplete {name} teacher bank")

    rank_audits = {}
    reference_keys = (
        "global_step", "initial_state_sha256", "observed_source_schedule_sha256",
        "parameter_count", "epoch_index", "completed_batches_in_epoch",
        "batches_per_epoch", "scheduler_position",
    )
    for rank in range(1, 10):
        arm_rows = {
            "fixed": read(fixed_root / f"rank_{rank:02d}/conditional-step000600-seed1.json"),
            "uniform": read(uniform_root / f"rank_{rank:02d}/conditional-step000600-seed1.json"),
            "target_specific": read(target_root / f"rank_{rank:02d}/conditional-step000600-seed1.json"),
        }
        checks = {
            key: len({json.dumps(row[key], sort_keys=True) for row in arm_rows.values()}) == 1
            for key in reference_keys
        }
        configs = {
            "fixed": read(fixed_root / f"rank_{rank:02d}/config.json"),
            "uniform": read(uniform_root / f"rank_{rank:02d}/config.json"),
            "target_specific": read(target_root / f"rank_{rank:02d}/config.json"),
        }
        train = {name: value["training_config"] for name, value in configs.items()}
        for key in (
            "fixed_pair_manifest_sha256", "fixed_pair_rank", "fixed_pair_seed",
            "fixed_pair_alignment_policy", "batch_size", "lr", "seed",
            "velocity_weight", "weight_decay", "precision",
        ):
            checks[f"config_{key}"] = len({json.dumps(value[key], sort_keys=True)
                                           for value in train.values()}) == 1
        checks["model_config"] = len({json.dumps(value["model_config"], sort_keys=True)
                                      for value in configs.values()}) == 1
        checks["expected_target_manifest"] = (
            train["fixed"]["fixed_pair_manifest_sha256"]
            == sha256(manifest_root / f"rank_{rank:02d}.json")
        )
        checks["uniform_reference_rule"] = (
            "latent_reference_temperature" in train["uniform"]
            and "reference_effective_count" not in train["uniform"]
        )
        checks["target_specific_reference_rule"] = (
            train["target_specific"].get("reference_effective_count") == 8.0
            and "reference_prior" in train["target_specific"]
        )
        if not all(checks.values()):
            raise ValueError(f"Teacher rank {rank} audit failed: {checks}")
        rank_audits[str(rank)] = {"status": "passed", "checks": checks}

    target_cache = read(root / (
        "coreset_reactor/reports/multiview_boundary_student_seed1_20261018/"
        "train_cache_head_center_tail/manifest.json"))
    teacher_hashes = {name: [sha256(path) for path in paths]
                      for name, paths in banks.items()}
    checks = {
        "shared_q0": teacher_hashes["fixed"][0]
        == teacher_hashes["uniform"][0] == teacher_hashes["target_specific"][0],
        "distinct_non_q0_banks": len({tuple(value[1:]) for value in teacher_hashes.values()}) == 3,
        "target_bank_matches_current_multiview_cache": (
            teacher_hashes["target_specific"] == target_cache["teacher_checkpoint_hashes"]
        ),
        "ten_teachers_each": all(len(value) == 10 for value in teacher_hashes.values()),
        "all_600_updates": all(
            read((path.parent / "conditional-step000600-seed1.json"))["global_step"] == 600
            for paths in banks.values() for path in paths[1:]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Teacher bank audit failed: {checks}")
    result = {
        "status": "passed", "checks": checks, "rank_audits": rank_audits,
        "definition": {
            "fixed": "assigned target only for MSE/CCC",
            "uniform": "same assigned MSE target; uniform all-reference CCC",
            "target_specific": "same assigned MSE target; target-centered reference-prior CCC",
        },
        "banks": {
            name: {
                "paths": [str(path.resolve()) for path in paths],
                "checkpoint_hashes": teacher_hashes[name],
                "bank_hash": combined_hash(paths),
                "updates_per_teacher": [
                    int(read(path.with_suffix(".json"))["global_step"])
                    for path in paths
                ],
                "summed_training_seconds": float(sum(
                    read(path.with_suffix(".json"))["elapsed_training_seconds"]
                    for path in paths
                )),
                "initial_state_hashes": [
                    read(path.with_suffix(".json"))["initial_state_sha256"]
                    for path in paths
                ],
                "observed_source_schedule_hashes": [
                    read(path.with_suffix(".json"))["observed_source_schedule_sha256"]
                    for path in paths
                ],
                "parameter_counts": [
                    int(read(path.with_suffix(".json"))["parameter_count"])
                    for path in paths
                ],
            }
            for name, paths in banks.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "checks": checks,
        "bank_hashes": {name: value["bank_hash"] for name, value in result["banks"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
