"""Assemble the predeclared three-seed A0/A1 REACT2025 comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from paper_core_ablations_react2025.assemble_a0_a1_results import (
    METRICS,
    aggregate,
    audit_population,
    load_a1,
    load_comparison,
    paired_analysis,
    sha256,
    training_metadata,
    write_json,
)


KINDS = ("raw", "postprocessed")


def load_eval(directory: Path, kind: str) -> list[dict]:
    return load_comparison(directory / f"comparison_{kind}.json")


def summed_context_seconds(path: Path) -> float:
    payload = json.loads(path.read_text())
    return float(sum(float(row["seconds"]) for row in payload["rows"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--seed1-a1-raw", type=Path, required=True)
    parser.add_argument("--seed1-a1-transition", type=Path, required=True)
    parser.add_argument("--seed1-a1-checkpoint", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    args = parser.parse_args()

    rows: dict[int, dict[str, dict[str, list[dict]]]] = {}
    checkpoints: dict[int, dict[str, Path]] = {}
    seed1_a1_raw, seed1_a1_post = load_a1(
        args.seed1_a1_raw, args.seed1_a1_transition)
    for seed in (1, 2, 3):
        a0_dir = args.experiment_dir / "runs" / f"a0_full_seed{seed}"
        a1_dir = args.experiment_dir / "runs" / f"a1_trajectory_seed{seed}"
        checkpoints[seed] = {
            "A0": a0_dir / "student.pth",
            "A1": (args.seed1_a1_checkpoint if seed == 1
                   else a1_dir / "student.pth"),
        }
        rows[seed] = {
            "A0": {kind: load_eval(a0_dir / "eval_combined", kind)
                   for kind in KINDS},
            "A1": ({"raw": seed1_a1_raw, "postprocessed": seed1_a1_post}
                   if seed == 1 else
                   {kind: load_eval(a1_dir / "eval_combined", kind)
                    for kind in KINDS}),
        }

    flat_rows = {
        f"seed{seed}_{arm}_{kind}": values
        for seed, seed_rows in rows.items()
        for arm, arm_rows in seed_rows.items()
        for kind, values in arm_rows.items()
    }
    audit_population(flat_rows)

    training = {
        seed: {arm: training_metadata(path)
               for arm, path in checkpoints[seed].items()}
        for seed in (1, 2, 3)
    }
    evaluation = {}
    for seed in (1, 2, 3):
        evaluation[seed] = {}
        for arm in ("A0", "A1"):
            if seed == 1 and arm == "A1":
                source = args.seed1_a1_raw
            else:
                source = (args.experiment_dir / "runs" /
                          f"{'a0_full' if arm == 'A0' else 'a1_trajectory'}_seed{seed}" /
                          "eval_combined" / "comparison_raw.json")
            evaluation[seed][arm] = {
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "summed_context_seconds": summed_context_seconds(source),
                "window": 750, "stride": 375, "power": 4.0,
            }
    match_audits = {}
    for seed in (1, 2, 3):
        checks = {
            key: training[seed]["A0"][key] == training[seed]["A1"][key]
            for key in (
                "teacher_bank_hash", "cache_hash", "initial_state_hash",
                "source_snapshot_hash", "observed_source_schedule_sha256",
                "updates", "velocity_weight",
            )
        }
        checks["loss_toggle"] = (
            training[seed]["A0"]["relation_objective"] == "centered_residual"
            and training[seed]["A0"]["relation_weight"] > 0
            and training[seed]["A1"]["relation_weight"] == 0
        )
        if not all(checks.values()):
            raise ValueError(f"Seed {seed} A0/A1 match audit failed: {checks}")
        match_audits[str(seed)] = {"status": "passed", "checks": checks}

    per_seed = {}
    for seed in (1, 2, 3):
        per_seed[str(seed)] = {
            "aggregate": {
                f"{arm}_{kind}": aggregate(rows[seed][arm][kind])
                for arm in ("A0", "A1") for kind in KINDS
            },
            "paired_A0_minus_A1": {
                kind: paired_analysis(
                    rows[seed]["A0"][kind], rows[seed]["A1"][kind],
                    args.bootstrap, args.bootstrap_seed,
                )
                for kind in KINDS
            },
        }

    across_seed = {"arms": {}, "paired_A0_minus_A1": {}}
    for arm in ("A0", "A1"):
        across_seed["arms"][arm] = {}
        for kind in KINDS:
            across_seed["arms"][arm][kind] = {}
            for metric in METRICS:
                values = np.asarray([
                    per_seed[str(seed)]["aggregate"][f"{arm}_{kind}"]
                    ["both_directions"][metric]
                    for seed in (1, 2, 3)
                ], dtype=np.float64)
                across_seed["arms"][arm][kind][metric] = {
                    "values": values.tolist(),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                }
    for kind in KINDS:
        across_seed["paired_A0_minus_A1"][kind] = {}
        for metric in METRICS:
            values = np.asarray([
                per_seed[str(seed)]["paired_A0_minus_A1"][kind]
                ["metrics"][metric]["mean_delta"]
                for seed in (1, 2, 3)
            ], dtype=np.float64)
            relative_values = np.asarray([
                per_seed[str(seed)]["paired_A0_minus_A1"][kind]
                ["metrics"][metric]["relative_delta_percent"]
                for seed in (1, 2, 3)
            ], dtype=np.float64)
            across_seed["paired_A0_minus_A1"][kind][metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "relative_delta_percent_values": relative_values.tolist(),
                "relative_delta_percent_mean": float(relative_values.mean()),
                "relative_delta_percent_sample_std": float(
                    relative_values.std(ddof=1)),
                "a0_better_seeds": int(sum(
                    value > 0 if metric in ("FRC", "FRDiv", "FRVar")
                    else value < 0 for value in values
                )),
            }

    result = {
        "protocol": (
            "REACT2025 complete bidirectional VAL1142; full p=4 stitched "
            "sequences; all opposite-role same-session GT; official Processor, "
            "AU rounding and five metrics"
        ),
        "training_seed_definition": (
            "Fixed teacher bank and cache; seeds alter the student initial state, "
            "training order and stochastic training only"
        ),
        "session_bootstrap": {
            "replicates": args.bootstrap, "seed": args.bootstrap_seed,
            "note": "Computed within each training seed; separate from across-seed SD",
        },
        "match_audits": match_audits,
        "training": training,
        "evaluation": evaluation,
        "per_seed": per_seed,
        "across_seed": across_seed,
    }
    write_json(args.experiment_dir / "MULTISEED_A0_A1.json", result)

    fields = (
        "dataset", "population", "seed", "arm", "output_kind", "FRC",
        "exact_FRD", "FRDiv", "FRVar", "FRSyn", "teacher_bank_hash",
        "cache_hash", "initial_state_hash", "source_snapshot_hash",
        "observed_source_schedule_sha256", "relation_weight", "updates",
        "training_seconds", "peak_memory_bytes", "window", "stride", "power",
        "postprocess", "summed_context_seconds", "status",
    )
    with (args.experiment_dir / "MULTISEED_A0_A1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for seed in (1, 2, 3):
            for arm in ("A0", "A1"):
                for kind in KINDS:
                    values = per_seed[str(seed)]["aggregate"][f"{arm}_{kind}"]["both_directions"]
                    meta = training[seed][arm]
                    writer.writerow({
                        "dataset": "REACT2025",
                        "population": "VAL1142_bidir_all_session_gt",
                        "seed": seed, "arm": arm, "output_kind": kind,
                        "FRC": values["FRC"], "exact_FRD": values["exact_FRD"],
                        "FRDiv": values["FRDiv"], "FRVar": values["FRVar"],
                        "FRSyn": values["TLCC"],
                        "teacher_bank_hash": meta["teacher_bank_hash"],
                        "cache_hash": meta["cache_hash"],
                        "initial_state_hash": meta["initial_state_hash"],
                        "source_snapshot_hash": meta["source_snapshot_hash"],
                        "observed_source_schedule_sha256": meta["observed_source_schedule_sha256"],
                        "relation_weight": meta["relation_weight"],
                        "updates": meta["updates"],
                        "training_seconds": meta["training_seconds"],
                        "peak_memory_bytes": meta["peak_memory_bytes"],
                        "window": evaluation[seed][arm]["window"],
                        "stride": evaluation[seed][arm]["stride"],
                        "power": evaluation[seed][arm]["power"],
                        "postprocess": ("raw" if kind == "raw" else
                                        "fixed_query0_transition"),
                        "summed_context_seconds": evaluation[seed][arm]["summed_context_seconds"],
                        "status": "complete",
                    })
    print(json.dumps({
        "status": "complete", "training_seeds": 3,
        "match_audits": {k: v["status"] for k, v in match_audits.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
