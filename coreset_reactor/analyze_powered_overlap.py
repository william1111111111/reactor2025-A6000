"""Paired session-bootstrap analysis for powered overlap fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coreset_reactor.analyze_multiview_boundary_student import (
    METRICS,
    paired_bootstrap,
    sha256,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--powered", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--replicates", type=int, default=20_000)
    args = parser.parse_args()

    powered_document = json.loads(args.powered.read_text())
    reference_document = json.loads(args.reference.read_text())
    powered_rows = powered_document["rows"]
    reference = {row["clip_id"]: row for row in reference_document["rows"]}
    if set(reference) != {row["clip_id"] for row in powered_rows}:
        raise ValueError("Powered and reference evaluations contain different clips")

    rng = np.random.default_rng(args.seed)
    groups = {}
    for group_name, predicate in {
        "all_forward": lambda row: True,
        "cropped_only": lambda row: row["length"] > 750,
        "short_or_equal_control": lambda row: row["length"] <= 750,
    }.items():
        rows = [row for row in powered_rows if predicate(row)]
        comparisons = {}
        for reference_name in ("overlap_center_weighted", "nonoverlap"):
            def values(row, reference_name=reference_name):
                new = row["powered_center_overlap"]
                old = reference[row["clip_id"]]["full_sequence"][reference_name]
                return [new[metric] - old[metric] for metric in METRICS]

            comparisons[f"powered_minus_{reference_name}"] = paired_bootstrap(
                rows, values, rng, args.replicates
            )
        groups[group_name] = {
            "contexts": len(rows),
            "comparisons": comparisons,
        }

    write_json(args.output, {
        "protocol": (
            "Paired clip differences with a session-clustered percentile bootstrap"
        ),
        "seed": args.seed,
        "replicates": args.replicates,
        "inputs": {
            "powered": [str(args.powered.resolve()), sha256(args.powered)],
            "reference": [str(args.reference.resolve()), sha256(args.reference)],
        },
        "groups": groups,
    })


if __name__ == "__main__":
    main()
