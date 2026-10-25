"""Fail fast unless the seed-2/3 A0/A1 training controls are matched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from paper_core_ablations_react2025.assemble_a0_a1_results import sha256, write_json


def load(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256(path),
        "state": state,
        "config": json.loads((path.parent / "config.json").read_text()),
        "provenance": state["provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "passed", "seeds": {}}
    for seed in (2, 3):
        arms = {
            "A0": load(args.experiment_dir / "runs" / f"a0_full_seed{seed}" / "student.pth"),
            "A1": load(args.experiment_dir / "runs" / f"a1_trajectory_seed{seed}" / "student.pth"),
        }
        p0, p1 = arms["A0"]["provenance"], arms["A1"]["provenance"]
        s0, s1 = arms["A0"]["state"], arms["A1"]["state"]
        c0, c1 = arms["A0"]["config"], arms["A1"]["config"]
        checks = {
            key: p0[key] == p1[key]
            for key in (
                "cache_manifest_hash", "teacher_checkpoint_hashes",
                "initialization_checkpoint_hash", "initial_student_state_sha256",
                "observed_source_schedule_sha256", "source_hashes",
            )
        }
        checks.update({
            "seed": c0["seed"] == c1["seed"] == seed,
            "epochs": c0["epochs"] == c1["epochs"] == 4,
            "batch_size": c0["batch_size"] == c1["batch_size"] == 4,
            "learning_rate": c0["lr"] == c1["lr"] == 1e-4,
            "velocity_weight": c0["velocity_weight"] == c1["velocity_weight"] == 0.05,
            "updates": s0["global_step"] == s1["global_step"] == 8820,
            "student_config": s0["student_config"] == s1["student_config"],
            "loss_toggle": (
                c0["relation_objective"] == "centered_residual"
                and c0["relation_weight"] > 0
                and c1["relation_weight"] == 0
            ),
        })
        if not all(checks.values()):
            result["status"] = "failed"
        result["seeds"][str(seed)] = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "initial_state_hash": p0["initial_student_state_sha256"],
            "schedule_hash": p0["observed_source_schedule_sha256"],
            "a0_relation_weight": c0["relation_weight"],
            "checkpoints": {
                arm: {key: value[key] for key in ("checkpoint", "checkpoint_sha256")}
                for arm, value in arms.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
