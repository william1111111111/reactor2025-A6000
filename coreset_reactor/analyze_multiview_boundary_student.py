"""Paired session-bootstrap analysis for the multi-view student experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEW = (
    ROOT
    / "coreset_reactor/reports/multiview_boundary_student_seed1_20261018"
    / "eval_forward_val571/comparison.json"
)
DEFAULT_CONTROL = (
    ROOT / "coreset_reactor/reports/window_gap_audit_20261018/comparison.json"
)
METRICS = ("FRC", "exact_FRD", "FRDiv", "FRVar")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def session_name(row) -> str:
    return row["clip_id"].split("/")[1]


def paired_bootstrap(rows, values, rng, replicates: int):
    differences = np.asarray([values(row) for row in rows], dtype=np.float64)
    sessions = sorted({session_name(row) for row in rows})
    session_index = np.asarray(
        [sessions.index(session_name(row)) for row in rows], dtype=np.int64
    )
    session_sums = np.stack(
        [differences[session_index == index].sum(axis=0)
         for index in range(len(sessions))]
    )
    session_counts = np.asarray(
        [(session_index == index).sum() for index in range(len(sessions))],
        dtype=np.int64,
    )
    selections = rng.integers(
        0, len(sessions), size=(replicates, len(sessions)), dtype=np.int64
    )
    samples = (
        session_sums[selections].sum(axis=1)
        / session_counts[selections].sum(axis=1, keepdims=True)
    )
    output = {}
    for metric_index, metric in enumerate(METRICS):
        lower, upper = np.quantile(samples[:, metric_index], (0.025, 0.975))
        delta = differences[:, metric_index]
        favorable = delta < 0 if metric == "exact_FRD" else delta > 0
        bootstrap_favorable = (
            samples[:, metric_index] < 0
            if metric == "exact_FRD"
            else samples[:, metric_index] > 0
        )
        output[metric] = {
            "mean_paired_delta": float(delta.mean()),
            "session_bootstrap_95ci": [float(lower), float(upper)],
            "bootstrap_probability_favorable": float(bootstrap_favorable.mean()),
            "context_fraction_favorable": float(favorable.mean()),
            "direction": "lower_is_better" if metric == "exact_FRD" else "higher_is_better",
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--replicates", type=int, default=20_000)
    args = parser.parse_args()

    new_document = json.loads(args.new.read_text())
    control_document = json.loads(args.control.read_text())
    new_rows = new_document["rows"]
    control_by_id = {row["clip_id"]: row for row in control_document["rows"]}
    if set(control_by_id) != {row["clip_id"] for row in new_rows}:
        raise ValueError("New and control comparisons do not contain identical clips")

    rng = np.random.default_rng(args.seed)
    subsets = {
        "all_forward": lambda row: True,
        "cropped_only": lambda row: row["length"] > 750,
        "short_or_equal_control": lambda row: row["length"] <= 750,
    }
    paths = {
        "center_standalone": ("positions", "center"),
        "full_nonoverlap": ("full_sequence", "nonoverlap"),
        "full_overlap_center_weighted": (
            "full_sequence", "overlap_center_weighted"
        ),
    }
    results = {}
    for subset_name, predicate in subsets.items():
        rows = [row for row in new_rows if predicate(row)]
        training = {}
        for comparison_name, path in paths.items():
            def training_values(row, path=path):
                new_metrics = row[path[0]][path[1]]
                old_metrics = control_by_id[row["clip_id"]][path[0]][path[1]]
                return [new_metrics[metric] - old_metrics[metric]
                        for metric in METRICS]

            training[comparison_name] = paired_bootstrap(
                rows, training_values, rng, args.replicates
            )

        def boundary_values(row):
            overlap = row["full_sequence"]["overlap_center_weighted"]
            nonoverlap = row["full_sequence"]["nonoverlap"]
            return [overlap[metric] - nonoverlap[metric] for metric in METRICS]

        results[subset_name] = {
            "contexts": len(rows),
            "sessions": len({session_name(row) for row in rows}),
            "training_delta_same_inference": training,
            "boundary_inference_delta": paired_bootstrap(
                rows, boundary_values, rng, args.replicates
            ),
        }

    write_json(args.output, {
        "protocol": (
            "Paired clip differences with a session-clustered percentile "
            "bootstrap; session clusters are sampled with replacement and "
            "all clips in each sampled cluster are retained."
        ),
        "seed": args.seed,
        "replicates": args.replicates,
        "inputs": {
            "new": [str(args.new.resolve()), sha256(args.new)],
            "control": [str(args.control.resolve()), sha256(args.control)],
        },
        "groups": results,
    })


if __name__ == "__main__":
    main()
