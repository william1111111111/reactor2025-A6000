"""Dataset-level GT diversity ceiling for exact select-10 sessions.

For every session, all listener trajectories are converted to one common
750-frame representation.  The script reports the analytic random-10
expectation, a deterministic farthest-sum lower bound, and a MILP solution of
the cardinality-constrained maximum-sum dispersion problem.  A result is
called exact only when SciPy/HiGHS returns an optimal MILP status; time-limited
incumbents are retained as lower bounds and are never mislabeled as optima.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from coreset_reactor.dataset import fixed_reaction


def farthest_sum_indices(distance: torch.Tensor, count: int,
                         max_swaps: int = 20) -> list[int]:
    """Greedy farthest-sum plus one-swap refinement, local and dependency-free."""
    distance = distance.detach().cpu()
    size = distance.shape[0]
    if distance.shape != (size, size) or not 1 < count <= size:
        raise ValueError("invalid distance matrix or selection count")
    upper_mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    upper = torch.where(upper_mask, distance,
                        torch.full_like(distance, -torch.inf))
    first_flat = int(upper.argmax())
    selected = [first_flat // size, first_flat % size]
    while len(selected) < count:
        candidates = [index for index in range(size) if index not in selected]
        gains = torch.tensor([float(distance[index, selected].sum())
                              for index in candidates])
        selected.append(candidates[int(gains.argmax())])

    def score(indices: list[int]) -> float:
        return float(distance[indices][:, indices].sum())

    for _ in range(max_swaps):
        current_score = score(selected)
        best_score = current_score
        best_swap = None
        outside = [index for index in range(size) if index not in selected]
        for position in range(count):
            kept = selected[:position] + selected[position + 1:]
            base = score(kept)
            for candidate in outside:
                candidate_score = base + 2 * float(distance[candidate, kept].sum())
                if candidate_score > best_score + 1e-9:
                    best_score = candidate_score
                    best_swap = position, candidate
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
    return sorted(selected)


def official_distance_matrix(reactions: torch.Tensor) -> np.ndarray:
    """Return the symmetric pairwise terms used by official FRDiv."""
    if reactions.ndim != 3 or reactions.shape[-1] != 25:
        raise ValueError("reactions must have shape [N,T,25]")
    flat = reactions.detach().cpu().numpy().astype(np.float64).reshape(
        reactions.shape[0], -1
    )
    gram = flat @ flat.T
    squared = np.maximum(
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram,
        0.0,
    )
    np.fill_diagonal(squared, 0.0)
    return squared / float(flat.shape[1])


def frdiv(distance: np.ndarray, selected: list[int]) -> float:
    """Compute ordered-pair FRDiv, equivalent to the 1/C(K,2) upper sum."""
    ids = np.asarray(selected, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("selected indices must be distinct and contain >=2 items")
    chosen = distance[np.ix_(ids, ids)]
    k = len(ids)
    return float(chosen.sum() / (k * (k - 1)))


def random_summary(distance: np.ndarray, count: int, trials: int,
                   seed: int) -> dict:
    """Summarize random subsets and include the exact random expectation."""
    n = distance.shape[0]
    if count > n:
        raise ValueError("selection count exceeds session size")
    # Every unordered pair has the same inclusion probability, so the
    # all-pair mean is the exact expectation of random-K FRDiv.
    all_pairs = distance[np.triu_indices(n, k=1)]
    expected = float(all_pairs.mean())
    rng = np.random.default_rng(seed)
    values = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        values[trial] = frdiv(distance, rng.choice(n, count, replace=False).tolist())
    return {
        "trials": int(trials),
        "analytic_expectation": expected,
        "sample_mean": float(values.mean()),
        "sample_std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _milp_constraints(n: int, pairs: list[tuple[int, int]], count: int):
    """Build cardinality and McCormick binary-product constraints."""
    pair_count = len(pairs)
    width = n + pair_count
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower = [float(count)]
    upper = [float(count)]
    for index in range(n):
        rows.append(0)
        cols.append(index)
        values.append(1.0)

    row = 1
    for offset, (left, right) in enumerate(pairs):
        z = n + offset
        # z - x_left <= 0
        rows.extend((row, row))
        cols.extend((z, left))
        values.extend((1.0, -1.0))
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1
        # z - x_right <= 0
        rows.extend((row, row))
        cols.extend((z, right))
        values.extend((1.0, -1.0))
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1
        # x_left + x_right - z <= 1
        rows.extend((row, row, row))
        cols.extend((left, right, z))
        values.extend((1.0, 1.0, -1.0))
        lower.append(-np.inf)
        upper.append(1.0)
        row += 1

    matrix = coo_matrix(
        (values, (rows, cols)), shape=(row, width), dtype=np.float64
    ).tocsc()
    return matrix, np.asarray(lower), np.asarray(upper)


def exact_max_sum_dispersion(distance: np.ndarray, count: int,
                             time_limit: float,
                             lower_bound_selected: list[int] | None = None) -> dict:
    """Solve max-sum dispersion with a binary MILP linearization."""
    n = distance.shape[0]
    if distance.shape != (n, n) or count < 2 or count > n:
        raise ValueError("invalid distance matrix or selection count")
    pairs = list(itertools.combinations(range(n), 2))
    matrix, lower, upper = _milp_constraints(n, pairs, count)
    minimum_sum = None
    if lower_bound_selected is not None:
        minimum_sum = float(distance[np.ix_(lower_bound_selected,
                                            lower_bound_selected)].sum() / 2.0)
        bound_row = np.zeros(n + len(pairs), dtype=np.float64)
        bound_row[n:] = [-distance[left, right] for left, right in pairs]
        # -sum(d_ij z_ij) <= -known_feasible_sum. This is a valid
        # incumbent cut and prevents a time-limited solve from returning a
        # useless low-quality incumbent when a heuristic lower bound exists.
        matrix = vstack((matrix, coo_matrix(bound_row[None, :]))).tocsc()
        lower = np.r_[lower, -np.inf]
        upper = np.r_[upper, -minimum_sum]
    objective = np.zeros(n + len(pairs), dtype=np.float64)
    objective[n:] = [-distance[left, right] for left, right in pairs]
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones_like(objective),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    elapsed = time.perf_counter() - started
    selected = None
    if result.x is not None:
        binary = result.x[:n]
        if np.max(np.abs(binary - np.round(binary))) <= 1e-6:
            selected = np.flatnonzero(binary > 0.5).astype(int).tolist()
    dual_bound = getattr(result, "mip_dual_bound", None)
    # The MILP minimizes -sum(d_ij z_ij). Convert both bounds to a positive
    # maximum-sum objective for an interpretable optimality gap.
    positive_incumbent = float(-result.fun) if result.fun is not None else None
    positive_dual = float(-dual_bound) if dual_bound is not None and np.isfinite(dual_bound) else None
    node_count = getattr(result, "mip_node_count", None)
    return {
        "selected": selected,
        "solver_FRDiv": (frdiv(distance, selected)
                          if selected is not None and len(selected) == count
                          else None),
        "status": int(result.status),
        "message": str(result.message),
        "optimal": bool(result.status == 0 and selected is not None and len(selected) == count),
        "runtime_seconds": elapsed,
        "mip_node_count": int(node_count) if node_count is not None else None,
        "mip_gap": (float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None),
        "objective_sum_lower": positive_incumbent,
        "objective_sum_upper": positive_dual,
        "known_feasible_sum": minimum_sum,
        "time_limit_seconds": float(time_limit),
        "formulation": "binary x_i and z_ij=x_i*x_j; sum_i x_i=K; three linear product constraints per pair",
    }


def exact_max_sum_dispersion_highspy(
    distance: np.ndarray, count: int, time_limit: float,
    lower_bound_selected: list[int] | None = None,
) -> dict:
    """HiGHS-native variant with a sparse farthest warm start.

    ``scipy.optimize.milp`` does not expose an initial solution.  The optional
    highspy backend does, which matters for the three N~110 sessions where
    HiGHS otherwise spent the whole time limit before finding an incumbent.
    """
    import highspy

    n = distance.shape[0]
    pairs = list(itertools.combinations(range(n), 2))
    matrix, lower, upper = _milp_constraints(n, pairs, count)
    minimum_sum = None
    if lower_bound_selected is not None:
        minimum_sum = float(distance[np.ix_(lower_bound_selected,
                                            lower_bound_selected)].sum() / 2.0)
        bound_row = np.zeros(n + len(pairs), dtype=np.float64)
        bound_row[n:] = [-distance[left, right] for left, right in pairs]
        matrix = vstack((matrix, coo_matrix(bound_row[None, :]))).tocsc()
        lower = np.r_[lower, -np.inf]
        upper = np.r_[upper, -minimum_sum]

    width = n + len(pairs)
    objective = np.zeros(width, dtype=np.float64)
    objective[n:] = [-distance[left, right] for left, right in pairs]
    csr = matrix.tocsr()
    solver = highspy.Highs()
    solver.addBinaries(width)
    solver.changeColsCost(width, np.arange(width, dtype=np.int32), objective)
    solver.addRows(
        matrix.shape[0], lower, upper, len(csr.data),
        csr.indptr.astype(np.int32), csr.indices.astype(np.int32),
        csr.data.astype(np.float64),
    )
    solver.setMinimize()
    solver.setOptionValue("time_limit", float(time_limit))
    solver.setOptionValue("mip_rel_gap", 0.0)
    solver.setOptionValue("output_flag", False)
    if lower_bound_selected is not None:
        selected_set = set(lower_bound_selected)
        warm_indices = list(lower_bound_selected)
        warm_values = [1.0] * len(warm_indices)
        pair_offset = {pair: n + offset for offset, pair in enumerate(pairs)}
        for left, right in itertools.combinations(sorted(selected_set), 2):
            warm_indices.append(pair_offset[(left, right)])
            warm_values.append(1.0)
        solver.setSolution(
            len(warm_indices), np.asarray(warm_indices, dtype=np.int32),
            np.asarray(warm_values, dtype=np.float64),
        )
    started = time.perf_counter()
    solver.run()
    elapsed = time.perf_counter() - started
    solution = solver.getSolution()
    values = np.asarray(solution.col_value, dtype=np.float64)
    selected = None
    if len(values) >= n and np.isfinite(values[:n]).all():
        if np.max(np.abs(values[:n] - np.round(values[:n]))) <= 1e-6:
            selected = np.flatnonzero(values[:n] > .5).astype(int).tolist()
    status = solver.getModelStatus()
    status_text = solver.modelStatusToString(status)
    info = solver.getInfo()
    solver_value = (frdiv(distance, selected)
                    if selected is not None and len(selected) == count else None)
    return {
        "selected": selected,
        "solver_FRDiv": solver_value,
        "status": status_text,
        "optimal": bool(status == highspy.HighsModelStatus.kOptimal
                          and selected is not None and len(selected) == count),
        "runtime_seconds": elapsed,
        "mip_node_count": int(info.mip_node_count),
        "mip_gap": float(info.mip_gap),
        "objective_sum_lower": (float(-info.objective_function_value)
                                 if np.isfinite(info.objective_function_value) else None),
        "objective_sum_upper": (float(-info.mip_dual_bound)
                                 if np.isfinite(info.mip_dual_bound) else None),
        "known_feasible_sum": minimum_sum,
        "time_limit_seconds": float(time_limit),
        "formulation": "HiGHS MILP with sparse farthest warm start and valid incumbent cut",
    }


def _session_seed(seed: int, session: str) -> int:
    digest = hashlib.sha256(f"{seed}|{session}|dataset-oracle".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def load_session_reactions(data_root: Path, split: str, frames: int,
                           role: str) -> dict[str, tuple[list[Path], torch.Tensor]]:
    root = data_root / split / "facial-attributes" / role
    if not root.is_dir():
        raise FileNotFoundError(f"missing reaction root: {root}")
    sessions = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = sorted(directory.glob("*.npy"))
        raw = []
        for path in paths:
            value = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
            if value.ndim != 2 or value.shape[1] != 25 or not torch.isfinite(value).all():
                raise ValueError(f"invalid reaction array: {path}")
            raw.append(value)
        if len(raw) >= 10:
            sessions[directory.name] = (paths, torch.stack([fixed_reaction(x, frames) for x in raw]))
    if not sessions:
        raise ValueError("no session contains at least ten reactions")
    return sessions


def run(data_root: Path, split: str, frames: int, role: str, count: int,
        random_trials: int, seed: int, time_limit: float,
        session_limit: int | None = None) -> dict:
    sessions = load_session_reactions(data_root, split, frames, role)
    names = sorted(sessions)
    if session_limit is not None:
        names = names[:session_limit]
    records = []
    started = time.perf_counter()
    for number, session in enumerate(names, start=1):
        paths, reactions = sessions[session]
        distance = official_distance_matrix(reactions)
        farthest = farthest_sum_indices(torch.from_numpy(distance), count)
        random = random_summary(distance, count, random_trials,
                                _session_seed(seed, session))
        exact = exact_max_sum_dispersion(distance, count, time_limit,
                                         lower_bound_selected=farthest)
        farthest_value = frdiv(distance, farthest)
        solver_value = exact["solver_FRDiv"]
        if solver_value is not None and solver_value >= farthest_value:
            best_known = solver_value
            best_known_indices = exact["selected"]
        else:
            best_known = farthest_value
            best_known_indices = farthest
        exact["exact_FRDiv"] = solver_value if exact["optimal"] else None
        exact["best_known_FRDiv"] = best_known
        exact["best_known_selected"] = best_known_indices
        record = {
            "session_id": session,
            "candidate_count": len(paths),
            "frames": frames,
            "distance_dimension": int(reactions.shape[1] * reactions.shape[2]),
            "random": random,
            "farthest_indices": farthest,
            "farthest_FRDiv": farthest_value,
            "exact": exact,
            "farthest_over_exact": (farthest_value / exact["exact_FRDiv"]
                                     if exact["exact_FRDiv"] else None),
            "paths": [str(path.relative_to(data_root)) for path in paths],
            "farthest_paths": [str(paths[index].relative_to(data_root)) for index in farthest],
        }
        records.append(record)
        print(json.dumps({
            "progress": f"{number}/{len(names)}", "session": session,
            "N": len(paths), "random": random["analytic_expectation"],
            "farthest": farthest_value,
            "solver_incumbent": exact["solver_FRDiv"],
            "best_known": exact["best_known_FRDiv"],
            "exact": exact["exact_FRDiv"], "optimal": exact["optimal"],
            "seconds": exact["runtime_seconds"],
        }), flush=True)
    exact_values = [r["exact"]["exact_FRDiv"] for r in records
                    if r["exact"]["exact_FRDiv"] is not None]
    best_known_values = [r["exact"]["best_known_FRDiv"] for r in records]
    farthest_values = [r["farthest_FRDiv"] for r in records]
    random_values = [r["random"]["analytic_expectation"] for r in records]
    exact_optimal_values = [
        r["exact"]["exact_FRDiv"] for r in records if r["exact"]["optimal"]
    ]
    summary = {
        "sessions": len(records),
        "exactly_solved_sessions": len(exact_optimal_values),
        "candidate_count": {
            "mean": float(np.mean([r["candidate_count"] for r in records])),
            "min": min(r["candidate_count"] for r in records),
            "max": max(r["candidate_count"] for r in records),
        },
        "session_balanced": {
            "random10_expectation": float(np.mean(random_values)),
            "farthest10_lower_bound": float(np.mean(farthest_values)),
            "best_known10_lower_bound": float(np.mean(best_known_values)),
            "exact_max10_if_all_sessions_optimal": (
                float(np.mean(exact_values))
                if len(exact_values) == len(records) else None
            ),
            "exact_max10_only_optimal_sessions": (
                float(np.mean(exact_optimal_values)) if exact_optimal_values else None
            ),
            "farthest_over_exact_mean": (
                float(np.mean([
                    r["farthest_over_exact"] for r in records
                    if r["farthest_over_exact"] is not None
                ])) if exact_values else None
            ),
        },
        "distribution_over_sessions": {
            "random10_expectation": {
                "median": float(np.median(random_values)),
                "p10": float(np.quantile(random_values, .1)),
                "p90": float(np.quantile(random_values, .9)),
            },
            "farthest10_lower_bound": {
                "median": float(np.median(farthest_values)),
                "p10": float(np.quantile(farthest_values, .1)),
                "p90": float(np.quantile(farthest_values, .9)),
            },
            "best_known10_lower_bound": {
                "median": float(np.median(best_known_values)),
                "p10": float(np.quantile(best_known_values, .1)),
                "p90": float(np.quantile(best_known_values, .9)),
            },
            "exact_max10_only_optimal_sessions": ({
                "median": float(np.median(exact_values)),
                "p10": float(np.quantile(exact_values, .1)),
                "p90": float(np.quantile(exact_values, .9)),
            } if exact_values else None),
        },
    }
    return {
        "protocol": {
            "split": split,
            "role": role,
            "frames": frames,
            "set_size": count,
            "distance": "float64 squared Euclidean / (25*T), fixed crop or linear stretch to T",
            "session_policy": "one independent GT pool per session; no source-context resampling",
            "random_policy": "analytic all-pair expectation plus seeded Monte Carlo distribution",
            "farthest_policy": "existing deterministic farthest-sum greedy plus one-swap refinement",
            "exact_policy": "SciPy HiGHS MILP; exact only when status=0 (optimal)",
            "seed": seed,
            "random_trials": random_trials,
            "time_limit_seconds_per_session": time_limit,
        },
        "summary": summary,
        "sessions": records,
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--role", default="listener")
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--set-size", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_root, args.split, args.frames, args.role,
                 args.set_size, args.random_trials, args.seed,
                 args.time_limit, args.session_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
