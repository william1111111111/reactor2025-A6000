"""Run HiGHS warm-start exact solves for selected large GT sessions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coreset_reactor.dataset_gt_oracle import (
    _session_seed,
    exact_max_sum_dispersion_highspy,
    farthest_sum_indices,
    frdiv,
    load_session_reactions,
    official_distance_matrix,
    random_summary,
)
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--role", default="listener")
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--set-size", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--time-limit", type=float, default=900.0)
    args = parser.parse_args()
    sessions = load_session_reactions(args.data_root, args.split, args.frames, args.role)
    records = []
    for session in args.sessions:
        if session not in sessions:
            raise KeyError(f"unknown session: {session}")
        paths, reactions = sessions[session]
        distance = official_distance_matrix(reactions)
        farthest = farthest_sum_indices(torch.from_numpy(distance), args.set_size)
        result = exact_max_sum_dispersion_highspy(
            distance, args.set_size, args.time_limit, farthest
        )
        record = {
            "session_id": session,
            "candidate_count": len(paths),
            "random": random_summary(
                distance, args.set_size, args.random_trials,
                _session_seed(args.seed, session),
            ),
            "farthest_indices": farthest,
            "farthest_FRDiv": frdiv(distance, farthest),
            "exact": result,
            "paths": [str(path.relative_to(args.data_root)) for path in paths],
        }
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"sessions": records}, indent=2) + "\n")
        print(json.dumps({
            "session": session,
            "N": len(paths),
            "farthest": record["farthest_FRDiv"],
            "solver": result["solver_FRDiv"],
            "optimal": result["optimal"],
            "status": result["status"],
            "gap": result["mip_gap"],
            "seconds": result["runtime_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
