"""Assemble the predeclared seed-1 A2/A3 REACT2025 controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from paper_core_ablations_react2025.assemble_a0_a1_results import (
    METRICS,
    aggregate,
    audit_population,
    load_a1,
    load_comparison,
    paired_analysis,
    read_json,
    sha256,
    training_metadata,
    write_json,
)


ARCHITECTURE_SOURCES = (
    "mam_reactor/regnn/expert_student.py",
    "mam_reactor/regnn/conditional_model.py",
)


def load_training(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": path,
        "state": state,
        "config": read_json(path.parent / "config.json"),
        "provenance": state["provenance"],
        "metadata": training_metadata(path),
    }


def same(values: dict, accessor) -> bool:
    observed = [json.dumps(accessor(value), sort_keys=True) for value in values.values()]
    return len(set(observed)) == 1


def audit_a0_a3(values: dict) -> dict:
    checks = {}
    for key in (
        "cache_manifest_hash", "teacher_checkpoint_hashes",
        "initialization_checkpoint_hash", "initial_student_state_sha256",
        "observed_source_schedule_sha256",
    ):
        checks[key] = same(values, lambda value, key=key: value["provenance"][key])
    checks["global_step"] = same(values, lambda value: value["state"]["global_step"])
    for key in ("epochs", "batch_size", "lr", "seed", "velocity_weight"):
        checks[key] = same(values, lambda value, key=key: value["config"][key])
    checks["student_config"] = same(values, lambda value: value["state"]["student_config"])
    checks["architecture_sources"] = all(
        values["A0"]["provenance"]["source_hashes"][source]
        == values["A3"]["provenance"]["source_hashes"][source]
        for source in ARCHITECTURE_SOURCES
    )
    a0_calibration = values["A0"]["state"]["relation_calibration"]
    a3_calibration = values["A3"]["state"]["relation_calibration"]
    checks["same_scale"] = a0_calibration["scale"] == a3_calibration["scale"]
    checks["same_weight"] = (
        values["A0"]["config"]["relation_weight"]
        == values["A3"]["config"]["relation_weight"]
        == a3_calibration["weight"]
    )
    checks["planned_objectives"] = (
        values["A0"]["config"]["relation_objective"] == "centered_residual"
        and values["A3"]["config"]["relation_objective"] == "group_weighted_mse"
    )
    if not all(checks.values()):
        raise ValueError(f"A0/A3 audit failed: {checks}")
    return {"status": "passed", "checks": checks}


def audit_target_pair(kd: dict, direct: dict, expected_velocity: float) -> dict:
    checks = {
        "initialization_checkpoint_hash": (
            kd["provenance"]["initialization_checkpoint_hash"]
            == direct["provenance"]["initialization_checkpoint_hash"]
        ),
        "initial_student_state_sha256": (
            kd["provenance"]["initial_student_state_sha256"]
            == direct["provenance"]["initial_student_state_sha256"]
        ),
        "observed_source_schedule_sha256": (
            kd["provenance"]["observed_source_schedule_sha256"]
            == direct["provenance"]["observed_source_schedule_sha256"]
        ),
        "global_step": kd["state"]["global_step"] == direct["state"]["global_step"],
        "student_config": kd["state"]["student_config"] == direct["state"]["student_config"],
        "architecture_sources": all(
            kd["provenance"]["source_hashes"][source]
            == direct["provenance"]["source_hashes"][source]
            for source in ARCHITECTURE_SOURCES
        ),
        "target_kinds": (
            kd["provenance"]["target_kind"] == "teacher_predictions"
            and direct["provenance"]["target_kind"] == "real_gt"
        ),
        "velocity_weight": (
            kd["config"]["velocity_weight"]
            == direct["config"]["velocity_weight"] == expected_velocity
        ),
        "no_auxiliary": (
            kd["config"]["relation_weight"] == 0
            and direct["config"]["relation_weight"] == 0
        ),
    }
    for key in ("epochs", "batch_size", "lr", "seed"):
        checks[key] = kd["config"][key] == direct["config"][key]
    if not all(checks.values()):
        raise ValueError(f"A1/A2 audit failed: {checks}")
    return {"status": "passed", "checks": checks}


def paired(left: list[dict], right: list[dict], left_name: str,
           right_name: str, bootstrap: int, seed: int) -> dict:
    value = paired_analysis(left, right, bootstrap, seed)
    value["definition"] = (
        f"{left_name} minus {right_name}; 95% CI resamples sessions with replacement"
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-eval", type=Path, required=True)
    parser.add_argument("--a0-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-raw", type=Path, required=True)
    parser.add_argument("--a1-transition", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    for arm in ("a3", "a2_velocity", "a1_no_velocity", "a2_no_velocity"):
        parser.add_argument(f"--{arm.replace('_', '-')}-eval", type=Path, required=True)
        parser.add_argument(f"--{arm.replace('_', '-')}-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-diagnostic", type=Path, required=True)
    parser.add_argument("--gradient-diagnostic", type=Path, required=True)
    parser.add_argument("--a2-cache-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    args = parser.parse_args()

    a1_raw, a1_post = load_a1(args.a1_raw, args.a1_transition)
    eval_dirs = {
        "A0": args.a0_eval,
        "A3": args.a3_eval,
        "A2_velocity": args.a2_velocity_eval,
        "A1_no_velocity": args.a1_no_velocity_eval,
        "A2_no_velocity": args.a2_no_velocity_eval,
    }
    rows = {"A1_raw": a1_raw, "A1_postprocessed": a1_post}
    for arm, directory in eval_dirs.items():
        rows[f"{arm}_raw"] = load_comparison(directory / "comparison_raw.json")
        rows[f"{arm}_postprocessed"] = load_comparison(
            directory / "comparison_postprocessed.json")
    audit_population(rows)

    checkpoint_paths = {
        "A0": args.a0_checkpoint,
        "A1": args.a1_checkpoint,
        "A3": args.a3_checkpoint,
        "A2_velocity": args.a2_velocity_checkpoint,
        "A1_no_velocity": args.a1_no_velocity_checkpoint,
        "A2_no_velocity": args.a2_no_velocity_checkpoint,
    }
    training = {name: load_training(path) for name, path in checkpoint_paths.items()}
    a2_cache_audit = read_json(args.a2_cache_audit)
    if a2_cache_audit["status"] != "passed":
        raise ValueError("A2 cache audit is not passed")
    audits = {
        "A0_vs_A3": audit_a0_a3({name: training[name] for name in ("A0", "A3")}),
        "A1_vs_A2_velocity": audit_target_pair(
            training["A1"], training["A2_velocity"], 0.05),
        "A1_vs_A2_no_velocity": audit_target_pair(
            training["A1_no_velocity"], training["A2_no_velocity"], 0.0),
        "A2_cache": a2_cache_audit,
    }
    comparisons = {
        "A0_minus_A1": ("A0", "A1"),
        "A0_minus_A3": ("A0", "A3"),
        "A1_minus_A2_velocity": ("A1", "A2_velocity"),
        "A1_no_velocity_minus_A2_no_velocity": (
            "A1_no_velocity", "A2_no_velocity"),
    }
    paired_results = {}
    for label, (left, right) in comparisons.items():
        paired_results[label] = {}
        for kind in ("raw", "postprocessed"):
            paired_results[label][kind] = paired(
                rows[f"{left}_{kind}"], rows[f"{right}_{kind}"], left, right,
                args.bootstrap, args.bootstrap_seed)

    metadata = {name: value["metadata"] for name, value in training.items()}
    for name, value in metadata.items():
        value["target_kind"] = training[name]["provenance"]["target_kind"]
    result = {
        "protocol": (
            "REACT2025 complete bidirectional VAL1142; full p=4 stitched sequences; "
            "all opposite-role same-session GT; official Processor, AU rounding and metrics"
        ),
        "audits": audits,
        "training": metadata,
        "aggregate": {name: aggregate(value) for name, value in rows.items()},
        "paired": paired_results,
        "diagnostics": {
            "difference_transfer": read_json(args.a3_diagnostic),
            "gradient_norms": read_json(args.gradient_diagnostic),
        },
        "inputs": {
            "a3_diagnostic": [str(args.a3_diagnostic.resolve()), sha256(args.a3_diagnostic)],
            "gradient_diagnostic": [str(args.gradient_diagnostic.resolve()), sha256(args.gradient_diagnostic)],
            "a2_cache_audit": [str(args.a2_cache_audit.resolve()), sha256(args.a2_cache_audit)],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "SEED1_CONTROLS.json", result)
    fields = (
        "dataset", "population", "seed", "arm", "output_kind", "target_kind",
        "cache_hash", "initial_state_hash", "updates", "velocity_weight",
        "relation_objective", "relation_weight", "window", "stride", "power",
        "postprocess", "FRC", "exact_FRD", "FRDiv", "FRVar", "FRSyn",
        "training_seconds", "peak_memory_bytes", "status",
    )
    with (args.output_dir / "SEED1_CONTROLS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in checkpoint_paths:
            for kind in ("raw", "postprocessed"):
                values = result["aggregate"][f"{arm}_{kind}"]["both_directions"]
                meta = metadata[arm]
                writer.writerow({
                    "dataset": "REACT2025", "population": "VAL1142_bidir_all_session_gt",
                    "seed": 1, "arm": arm, "output_kind": kind,
                    "target_kind": meta["target_kind"], "cache_hash": meta["cache_hash"],
                    "initial_state_hash": meta["initial_state_hash"], "updates": meta["updates"],
                    "velocity_weight": meta["velocity_weight"],
                    "relation_objective": meta["relation_objective"],
                    "relation_weight": meta["relation_weight"], "window": 750,
                    "stride": 375, "power": 4,
                    "postprocess": "raw" if kind == "raw" else "fixed_query0_transition",
                    "FRC": values["FRC"], "exact_FRD": values["exact_FRD"],
                    "FRDiv": values["FRDiv"], "FRVar": values["FRVar"],
                    "FRSyn": values["TLCC"], "training_seconds": meta["training_seconds"],
                    "peak_memory_bytes": meta["peak_memory_bytes"], "status": "complete",
                })
    print(json.dumps({
        "status": "complete", "rows": len(checkpoint_paths) * 2,
        "audits": {key: value["status"] for key, value in audits.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
