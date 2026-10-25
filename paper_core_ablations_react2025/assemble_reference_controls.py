"""Assemble matched reference-prior teacher and student VAL1142 controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from paper_core_ablations_react2025.assemble_a0_a1_results import (
    METRICS,
    aggregate,
    audit_population,
    combined_hash,
    load_comparison,
    paired_analysis,
    read_json,
    sha256,
    write_json,
)


ARMS = ("fixed", "uniform", "target_specific")
KINDS = ("raw", "postprocessed")


def paired(left, right, left_name, right_name, bootstrap, seed):
    value = paired_analysis(left, right, bootstrap, seed)
    value["definition"] = (
        f"{left_name} minus {right_name}; 95% CI resamples sessions with replacement"
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-eval", type=Path, required=True)
    parser.add_argument("--fixed-student-eval", type=Path, required=True)
    parser.add_argument("--uniform-student-eval", type=Path, required=True)
    parser.add_argument("--target-student-eval", type=Path, required=True)
    parser.add_argument("--bank-audit", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--student-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    args = parser.parse_args()
    audits = {
        "teacher_banks": read_json(args.bank_audit),
        "student_caches": read_json(args.cache_audit),
        "students": read_json(args.student_audit),
    }
    if any(value["status"] != "passed" for value in audits.values()):
        raise ValueError("A reference-control audit did not pass")

    teacher_raw_payload = read_json(
        args.teacher_eval / "comparison_fixed_raw.json")
    teacher_shared_seconds = float(sum(
        row["seconds_shared_three_banks"] for row in teacher_raw_payload["rows"]))

    student_dirs = {
        "fixed": args.fixed_student_eval,
        "uniform": args.uniform_student_eval,
        "target_specific": args.target_student_eval,
    }
    rows = {}
    evaluation = {"teacher": {}, "student": {}}
    for arm in ARMS:
        for kind in KINDS:
            rows[f"teacher_{arm}_{kind}"] = load_comparison(
                args.teacher_eval / f"comparison_{arm}_{kind}.json")
            rows[f"student_{arm}_{kind}"] = load_comparison(
                student_dirs[arm] / f"comparison_{kind}.json")
        student_payload = read_json(student_dirs[arm] / "comparison_raw.json")
        evaluation["teacher"][arm] = {
            "summed_context_seconds": teacher_shared_seconds,
            "seconds_basis": "shared joint evaluation of all three teacher banks",
        }
        evaluation["student"][arm] = {
            "summed_context_seconds": float(sum(
                row["seconds"] for row in student_payload["rows"])),
            "seconds_basis": "this student arm only",
        }
    audit_population(rows)

    comparisons = {
        "uniform_minus_fixed": ("uniform", "fixed"),
        "target_specific_minus_fixed": ("target_specific", "fixed"),
        "target_specific_minus_uniform": ("target_specific", "uniform"),
    }
    paired_results = {}
    for stage in ("teacher", "student"):
        paired_results[stage] = {}
        for label, (left, right) in comparisons.items():
            paired_results[stage][label] = {
                kind: paired(
                    rows[f"{stage}_{left}_{kind}"],
                    rows[f"{stage}_{right}_{kind}"],
                    left, right, args.bootstrap, args.bootstrap_seed,
                ) for kind in KINDS
            }
    result = {
        "protocol": (
            "REACT2025 complete bidirectional VAL1142; full p=4 stitched "
            "sequences; all opposite-role same-session GT; official Processor, "
            "AU rounding and five metrics"
        ),
        "arms": {
            "fixed": "assigned target only for teacher MSE/CCC",
            "uniform": "same assigned MSE target; uniform all-reference teacher CCC",
            "target_specific": "same assigned MSE target; target-centered reference-prior teacher CCC",
        },
        "student_control": (
            "Same seed-1 architecture, initialization, multiview source rows, "
            "schedule, budget, velocity and frozen centered-residual loss; only "
            "the teacher-output bank changes"
        ),
        "audits": audits,
        "evaluation": evaluation,
        "aggregate": {name: aggregate(value) for name, value in rows.items()},
        "paired": paired_results,
        "inputs": {
            "teacher_eval_manifest": [str((args.teacher_eval / "manifest.json").resolve()),
                                      sha256(args.teacher_eval / "manifest.json")],
            "bank_audit": [str(args.bank_audit.resolve()), sha256(args.bank_audit)],
            "cache_audit": [str(args.cache_audit.resolve()), sha256(args.cache_audit)],
            "student_audit": [str(args.student_audit.resolve()), sha256(args.student_audit)],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "REFERENCE_CONTROLS.json", result)
    fields = (
        "dataset", "population", "stage", "seed", "arm", "output_kind",
        "teacher_bank_hash", "cache_hash", "initial_state_hash",
        "source_snapshot_hash", "observed_source_schedule_hash", "updates",
        "training_seconds", "peak_memory_bytes", "window", "stride", "power",
        "postprocess", "summed_context_seconds", "evaluation_seconds_basis",
        "FRC", "exact_FRD", "FRDiv", "FRVar", "FRSyn", "status",
    )
    with (args.output_dir / "REFERENCE_CONTROLS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stage in ("teacher", "student"):
            for arm in ARMS:
                for kind in KINDS:
                    values = result["aggregate"][f"{stage}_{arm}_{kind}"]["both_directions"]
                    if stage == "student":
                        run = audits["students"]["students"][arm]
                        cache_hash = run["cache_hash"]
                        initial_state_hash = run["initial_state_hash"]
                        source_snapshot_hash = run["source_snapshot_hash"]
                        schedule_hash = run["schedule_hash"]
                        updates = run["updates"]
                        training_seconds = run["training_seconds"]
                        peak_memory_bytes = run["peak_memory_bytes"]
                    else:
                        bank = audits["teacher_banks"]["banks"][arm]
                        cache_hash = "n/a_teacher_stage"
                        initial_state_hash = combined_hash(bank["initial_state_hashes"])
                        source_snapshot_hash = "unavailable_in_historical_teacher_assets"
                        schedule_hash = combined_hash(
                            bank["observed_source_schedule_hashes"])
                        updates = "+".join(map(str, bank["updates_per_teacher"]))
                        training_seconds = bank["summed_training_seconds"]
                        peak_memory_bytes = "unavailable_in_historical_teacher_assets"
                    writer.writerow({
                        "dataset": "REACT2025",
                        "population": "VAL1142_bidir_all_session_gt",
                        "stage": stage, "seed": 1, "arm": arm,
                        "output_kind": kind,
                        "teacher_bank_hash": audits["teacher_banks"]["banks"][arm]["bank_hash"],
                        "cache_hash": cache_hash,
                        "initial_state_hash": initial_state_hash,
                        "source_snapshot_hash": source_snapshot_hash,
                        "observed_source_schedule_hash": schedule_hash,
                        "updates": updates,
                        "training_seconds": training_seconds,
                        "peak_memory_bytes": peak_memory_bytes,
                        "window": 750, "stride": 375, "power": 4.0,
                        "postprocess": "raw" if kind == "raw" else "fixed_query0_transition",
                        "summed_context_seconds": evaluation[stage][arm]["summed_context_seconds"],
                        "evaluation_seconds_basis": evaluation[stage][arm]["seconds_basis"],
                        "FRC": values["FRC"], "exact_FRD": values["exact_FRD"],
                        "FRDiv": values["FRDiv"], "FRVar": values["FRVar"],
                        "FRSyn": values["TLCC"], "status": "complete",
                    })
    print(json.dumps({
        "status": "complete", "rows": 12,
        "audits": {name: value["status"] for name, value in audits.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
