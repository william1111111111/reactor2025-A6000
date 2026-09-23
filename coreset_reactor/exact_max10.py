"""CLI for the dataset-level exact session Max-10 FRDiv ceiling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coreset_reactor.dataset_gt_oracle import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--role", default="listener")
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--set-size", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--session-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.data_root, args.split, args.frames, args.role, args.set_size,
        args.random_trials, args.seed, args.time_limit, args.session_limit,
    )
    result["entrypoint"] = "coreset_reactor.exact_max10"
    result["interpretation"] = (
        "exact_max10 is reported only when the MILP status is optimal; "
        "otherwise best_known is a lower bound"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
