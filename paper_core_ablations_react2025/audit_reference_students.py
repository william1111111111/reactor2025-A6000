"""Audit the three matched reference-teacher student arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from paper_core_ablations_react2025.assemble_a0_a1_results import (
    combined_hash,
    sha256,
    write_json,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def load(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "state": state,
        "config": read(path.parent / "config.json"),
        "provenance": state["provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--target-specific", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--bank-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    students = {
        "fixed": load(args.fixed), "uniform": load(args.uniform),
        "target_specific": load(args.target_specific),
    }
    cache_audit, bank_audit = read(args.cache_audit), read(args.bank_audit)
    if cache_audit["status"] != "passed" or bank_audit["status"] != "passed":
        raise ValueError("Upstream reference audit did not pass")
    checks = {}
    for key in ("initialization_checkpoint_hash", "initial_student_state_sha256",
                "observed_source_schedule_sha256"):
        checks[key] = len({student["provenance"][key]
                           for student in students.values()}) == 1
    for key in ("seed", "epochs", "batch_size", "lr", "velocity_weight",
                "expert_embedding_dim"):
        checks[f"config_{key}"] = len({json.dumps(student["config"][key], sort_keys=True)
                                       for student in students.values()}) == 1
    checks["updates"] = len({student["state"]["global_step"]
                             for student in students.values()}) == 1
    checks["expected_updates"] = students["fixed"]["state"]["global_step"] == 8820
    checks["student_config"] = len({json.dumps(student["state"]["student_config"], sort_keys=True)
                                    for student in students.values()}) == 1
    checks["teacher_banks"] = all(
        students[name]["provenance"]["teacher_checkpoint_hashes"]
        == bank_audit["banks"][name]["checkpoint_hashes"] for name in students
    )
    checks["cache_hashes"] = all(
        students[name]["provenance"]["cache_manifest_hash"]
        == cache_audit["manifests"][name]["cache_manifest_hash"] for name in students
    )
    checks["centered_objective"] = all(
        student["config"]["relation_objective"] == "centered_residual"
        for student in students.values())
    checks["same_relation_scale"] = len({
        json.dumps(student["state"]["relation_calibration"]["scale"])
        for student in students.values()}) == 1
    checks["same_relation_weight"] = len({student["config"]["relation_weight"]
                                          for student in students.values()}) == 1
    architecture = ("mam_reactor/regnn/expert_student.py",
                    "mam_reactor/regnn/conditional_model.py")
    checks["architecture_sources"] = all(
        len({student["provenance"]["source_hashes"][source]
             for student in students.values()}) == 1 for source in architecture)
    if not all(checks.values()):
        raise ValueError(f"Reference student audit failed: {checks}")
    result = {
        "status": "passed", "checks": checks,
        "note": (
            "The target-specific A0 training entry predates addition of the unused "
            "A3 group-weighted objective; architecture hashes and executed centered "
            "objective are matched."
        ),
        "students": {
            name: {
                "checkpoint": value["path"], "checkpoint_sha256": value["sha256"],
                "cache_hash": value["provenance"]["cache_manifest_hash"],
                "teacher_bank_hash": bank_audit["banks"][name]["bank_hash"],
                "initial_state_hash": value["provenance"]["initial_student_state_sha256"],
                "schedule_hash": value["provenance"]["observed_source_schedule_sha256"],
                "source_snapshot_hash": combined_hash(
                    value["provenance"].get("source_hashes", {})),
                "relation_scale": value["state"]["relation_calibration"]["scale"],
                "relation_weight": value["config"]["relation_weight"],
                "updates": int(value["state"]["global_step"]),
                "training_seconds": value["state"].get("training_seconds",
                                                         value["config"].get("training_seconds")),
                "peak_memory_bytes": value["state"].get("peak_memory_bytes",
                                                          value["config"].get("peak_memory_bytes")),
            } for name, value in students.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
