"""Build TRAIN-only random and Max-FRDiv fixed-pair teacher manifests.

The membership objective is the official trajectory-space FRDiv objective after
deterministic relative-time resampling.  Descriptors are used only after
membership is fixed, to give the nine non-anchor targets a stable global slot
identity.  No validation GT is read by this module.

T0 is the existing deterministic random fixed-pair policy.  T1 keeps the true
same-basename pair at rank zero and chooses nine alternatives by trajectory
Max-FRDiv.  T2 is an unconstrained diagnostic and is never a training default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, milp
from scipy.sparse import coo_matrix
from scipy.optimize import Bounds, LinearConstraint
from sklearn.cluster import KMeans

from coreset_reactor.dataset import full_reaction_descriptor
from coreset_reactor.dataset_gt_oracle import (
    _milp_constraints,
    farthest_sum_indices,
    frdiv,
    official_distance_matrix,
)
from coreset_reactor.fixed_pair_manifest import assign_fixed_random


SCHEMA_VERSION = 1
SELECTION_VERSION = "relative-time-resample-v1"


def _sha256_file_set(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _relative_resample(raw: torch.Tensor, frames: int) -> torch.Tensor:
    """Call Mam-Reactor's canonical cross-listener resampler."""
    try:
        from regnn.conditional_data import relative_time_resample_reaction
    except ImportError as exc:  # pragma: no cover - CLI setup error
        raise ImportError(
            "Run with mam_reactor on PYTHONPATH or pass --mam-root"
        ) from exc
    return relative_time_resample_reaction(raw, frames)


def _stable_random(seed: int, sample_id: str) -> random.Random:
    value = hashlib.sha256(
        f"{seed}|{sample_id}|fixed-random-assignment".encode()
    ).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


def _load_raw(path: Path) -> torch.Tensor:
    value = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
    if value.ndim != 2 or value.shape[1] != 25 or len(value) == 0:
        raise ValueError(f"invalid reaction array: {path}: {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite reaction array: {path}")
    return value


def _pairwise_sum(distance: np.ndarray, selected: list[int]) -> float:
    ids = np.asarray(selected, dtype=np.int64)
    return float(distance[np.ix_(ids, ids)].sum() / 2.0)


def paired_farthest_indices(
    distance: np.ndarray,
    paired_index: int,
    count: int = 10,
    max_swaps: int = 20,
) -> list[int]:
    """Approximate paired-constrained max-sum dispersion.

    The paired item is immutable.  Greedy additions and one-swap refinement
    are deterministic and operate on the same official distance matrix as the
    exact MILP implementation.
    """
    n = distance.shape[0]
    if distance.shape != (n, n) or not 0 <= paired_index < n or count < 2:
        raise ValueError("invalid distance matrix or paired index")
    if count > n:
        raise ValueError("selection count exceeds candidate count")
    selected = [paired_index]
    while len(selected) < count:
        candidates = [i for i in range(n) if i not in selected]
        gains = [float(distance[i, selected].sum()) for i in candidates]
        selected.append(candidates[int(np.argmax(gains))])
    for _ in range(max_swaps):
        current = _pairwise_sum(distance, selected)
        best = current
        best_swap = None
        outside = [i for i in range(n) if i not in selected]
        for position in range(1, count):  # rank zero is the paired target
            kept = selected[:position] + selected[position + 1:]
            for candidate in outside:
                value = _pairwise_sum(distance, kept + [candidate])
                if value > best + 1e-10:
                    best = value
                    best_swap = position, candidate
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
    return sorted(selected, key=lambda i: (i != paired_index, i))


def paired_exact_indices(
    distance: np.ndarray,
    paired_index: int,
    count: int = 10,
    time_limit: float = 300.0,
) -> dict:
    """Exact paired-constrained MILP, retaining solver provenance."""
    n = distance.shape[0]
    alternatives = [i for i in range(n) if i != paired_index]
    choose = count - 1
    pairs = [(left, right) for left in range(len(alternatives))
             for right in range(left + 1, len(alternatives))]
    matrix, lower, upper = _milp_constraints(len(alternatives), pairs, choose)
    width = matrix.shape[1]
    objective = np.zeros(width, dtype=np.float64)
    for index, original in enumerate(alternatives):
        objective[index] = -distance[paired_index, original]
    for offset, (left, right) in enumerate(pairs):
        objective[len(alternatives) + offset] = -distance[
            alternatives[left], alternatives[right]
        ]
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones(width, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    selected = None
    if result.x is not None:
        values = result.x[:len(alternatives)]
        if np.max(np.abs(values - np.round(values))) <= 1e-6:
            chosen = np.flatnonzero(values > .5).tolist()
            if len(chosen) == choose:
                selected = [paired_index] + [alternatives[i] for i in chosen]
    return {
        "selected": sorted(selected, key=lambda i: (i != paired_index, i))
        if selected is not None else None,
        "FRDiv": frdiv(distance, selected) if selected is not None else None,
        "status": int(result.status),
        "message": str(result.message),
        "optimal": bool(result.status == 0 and selected is not None),
        "runtime_seconds": time.perf_counter() - started,
        "mip_gap": (float(result.mip_gap)
                     if getattr(result, "mip_gap", None) is not None else None),
        "mip_node_count": (int(result.mip_node_count)
                           if getattr(result, "mip_node_count", None) is not None
                           else None),
        "formulation": "paired fixed x0 + binary alternatives and z_ij products",
    }


def _descriptor_statistics(
    raw_descriptors: np.ndarray,
    selected: list[int],
    reactions: torch.Tensor,
    descriptor_mean: np.ndarray,
    descriptor_std: np.ndarray,
) -> dict:
    values = (raw_descriptors - descriptor_mean) / descriptor_std
    selected_values = values[selected]
    chosen = reactions[selected].numpy()
    expression = np.argmax(chosen[:, :, 17:25].mean(1), axis=1)
    return {
        "FRDiv": frdiv(official_distance_matrix(reactions), selected),
        "descriptor_pair_distance": float(np.mean([
            np.sqrt(np.mean((selected_values[i] - selected_values[j]) ** 2))
            for i in range(len(selected)) for j in range(i + 1, len(selected))
        ])),
        "au_range": float(np.mean(chosen[:, :, :15].max(1).max(0)
                                 - chosen[:, :, :15].min(1).min(0))),
        "va_range": float(np.mean(chosen[:, :, 15:17].max(1).max(0)
                                 - chosen[:, :, 15:17].min(1).min(0))),
        "distinct_expression_modes": int(np.unique(expression).size),
    }


def _fit_global_prototypes(
    data_root: Path, n_modes: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    root = data_root / "train" / "facial-attributes"
    paths = sorted(root.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"no TRAIN facial attributes under {root}")
    raw = np.stack([full_reaction_descriptor(_load_raw(path)).numpy()
                    for path in paths])
    mean = raw.mean(0)
    std = np.maximum(raw.std(0), .03)
    values = (raw - mean) / std
    model = KMeans(n_clusters=n_modes, random_state=1, n_init=20).fit(values)
    nearest = np.mean((values[:, None] - model.cluster_centers_[None]) ** 2, 2)
    order = np.argsort(nearest.argmin(0))
    centers = model.cluster_centers_[order]
    return mean, std, centers, _sha256_file_set(paths, root)


def _assign_canonical_slots(
    selected: list[int], paired_index: int, values: np.ndarray,
    centers: np.ndarray,
) -> list[int]:
    """Keep rank zero paired and Hungarian-match the nine alternatives."""
    if paired_index not in selected or len(selected) != 10:
        raise ValueError("T1 canonical assignment requires paired target")
    alternatives = [index for index in selected if index != paired_index]
    cost = np.mean((values[alternatives, None] - centers[None]) ** 2, axis=-1)
    modes, columns = linear_sum_assignment(cost.T)
    assigned = [alternatives[int(columns[np.where(modes == mode)[0][0]])]
                for mode in range(centers.shape[0])]
    return [paired_index, *assigned]


def _stats_summary(records: list[dict], key: str) -> dict:
    values = np.asarray([row[key]["stats"] for row in records], dtype=object)
    fields = ("FRDiv", "descriptor_pair_distance", "au_range", "va_range",
              "distinct_expression_modes")
    return {field: float(np.mean([item[field] for item in values]))
            for field in fields}


def build_manifest(
    data_root: Path,
    output: Path,
    seed: int = 1,
    frames: int = 750,
    exact_t1: bool = False,
    exact_t2: bool = False,
    exact_time_limit: float = 300.0,
    session_limit: int | None = None,
) -> dict:
    data_root = data_root.resolve()
    facial = data_root / "train" / "facial-attributes"
    source_paths = sorted((facial / "speaker").glob("*/*.npy"))
    if not source_paths:
        raise FileNotFoundError(f"no TRAIN speaker attributes under {facial}")
    target_by_session: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((facial / "listener").glob("*/*.npy")):
        target_by_session[path.parent.name].append(path)
    sources_by_session: dict[str, list[Path]] = defaultdict(list)
    for path in source_paths:
        sources_by_session[path.parent.name].append(path)
    sessions = sorted(sources_by_session)
    if session_limit is not None:
        sessions = sessions[:session_limit]

    descriptor_mean, descriptor_std, centers, data_sha256 = _fit_global_prototypes(
        data_root
    )
    records = []
    summary_rows = []
    exact_counts = {"T1": 0, "T2": 0}
    for session in sessions:
        target_paths = target_by_session[session]
        if len(target_paths) < 10:
            raise ValueError(f"session {session} has fewer than ten listener GTs")
        raw = [_load_raw(path) for path in target_paths]
        reactions = torch.stack([_relative_resample(value, frames) for value in raw])
        raw_descriptors = np.stack([
            full_reaction_descriptor(value).numpy() for value in raw
        ])
        values = (raw_descriptors - descriptor_mean) / descriptor_std
        distance = official_distance_matrix(reactions)
        target_ids = [path.relative_to(facial).as_posix() for path in target_paths]
        target_lookup = {path: index for index, path in enumerate(target_ids)}
        for source in sorted(sources_by_session[session]):
            source_id = source.relative_to(facial).as_posix()
            basename = source.name
            paired_id = f"listener/{session}/{basename}"
            if paired_id not in target_lookup:
                raise ValueError(f"missing paired target {paired_id}")
            paired_index = target_lookup[paired_id]
            candidates = list(range(len(target_paths)))

            t0_ids, _ = assign_fixed_random(
                target_ids, paired_id, source_id, seed, 10, True
            )
            t0_indices = [target_lookup[item] for item in t0_ids]

            t1_solver = None
            if exact_t1:
                t1_solver = paired_exact_indices(
                    distance, paired_index, 10, exact_time_limit
                )
                t1_indices = t1_solver["selected"]
                if t1_indices is None:
                    t1_indices = paired_farthest_indices(distance, paired_index)
            else:
                t1_indices = paired_farthest_indices(distance, paired_index)
            t1_indices = _assign_canonical_slots(
                t1_indices, paired_index, values, centers
            )

            t2_solver = None
            if exact_t2:
                from coreset_reactor.dataset_gt_oracle import exact_max_sum_dispersion
                t2_solver = exact_max_sum_dispersion(
                    distance, 10, exact_time_limit,
                    lower_bound_selected=farthest_sum_indices(
                        torch.from_numpy(distance), 10
                    ),
                )
                t2_indices = t2_solver["selected"]
                if t2_indices is None:
                    t2_indices = farthest_sum_indices(torch.from_numpy(distance), 10)
            else:
                t2_indices = farthest_sum_indices(torch.from_numpy(distance), 10)
            if paired_index in t2_indices:
                t2_indices = _assign_canonical_slots(
                    t2_indices, paired_index, values, centers
                )
            else:
                # T2 is diagnostic only.  Give it a deterministic descriptor
                # ordering; it is never passed to fixed-pair training.
                t2_values = values[t2_indices]
                modes, columns = linear_sum_assignment(
                    np.mean((t2_values[:, None] - centers[None]) ** 2, axis=-1).T
                )
                ordered = [t2_indices[int(columns[np.where(modes == mode)[0][0]])]
                           for mode in range(9)]
                remainder = [index for index in t2_indices if index not in ordered]
                t2_indices = [*ordered, *remainder]

            def pack(indices: list[int], method: str, solver: dict | None):
                return {
                    "target_ids": [target_ids[index] for index in indices],
                    "membership_indices": indices,
                    "selection_method": method,
                    "solver": solver,
                    "stats": _descriptor_statistics(
                        raw_descriptors, indices, reactions,
                        descriptor_mean, descriptor_std,
                    ),
                }

            record = {
                "sample_id": source_id,
                "session_id": session,
                "paired_target_id": paired_id,
                "candidate_count": len(target_ids),
                "T0": pack(t0_indices, "random_fixed_pair", None),
                "T1": pack(
                    t1_indices,
                    "paired_exact_max_frdiv" if exact_t1 else
                    "paired_approximate_farthest_sum_swap",
                    t1_solver,
                ),
                "T2": pack(
                    t2_indices,
                    "unconstrained_exact_max_frdiv" if exact_t2 else
                    "unconstrained_approximate_farthest_sum_swap",
                    t2_solver,
                ),
            }
            records.append(record)
            summary_rows.append(record)
        print(json.dumps({"session": session, "contexts": len(sources_by_session[session]),
                          "completed": len(records), "total": len(source_paths)}), flush=True)

    summary = {name: _stats_summary(records, name) for name in ("T0", "T1", "T2")}
    summary["T1_over_T0_FRDiv"] = summary["T1"]["FRDiv"] / max(summary["T0"]["FRDiv"], 1e-12)
    summary["T2_over_T1_FRDiv"] = summary["T2"]["FRDiv"] / max(summary["T1"]["FRDiv"], 1e-12)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "source_role": "speaker",
        "target_role": "listener",
        "frames": frames,
        "selection_version": SELECTION_VERSION,
        "alignment_policy": "relative_time_masked",
        "membership_distance": "mean squared 25D trajectory distance after relative_time_resample_reaction",
        "seed": seed,
        "data_root": str(data_root),
        "data_sha256": data_sha256,
        "prototype_policy": "TRAIN-only global KMeans(9) descriptor prototypes; Hungarian slot assignment",
        "descriptor_mean": descriptor_mean.tolist(),
        "descriptor_std": descriptor_std.tolist(),
        "prototype_centers": centers.tolist(),
        "contexts": len(records),
        "sessions": len(sessions),
        "exact_t1": exact_t1,
        "exact_t2": exact_t2,
        "exact_time_limit_seconds": exact_time_limit,
        "summary": summary,
        "records": records,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def write_sanitized_report(manifest: dict, output: Path) -> None:
    summary = manifest["summary"]
    lines = [
        "# Max-FRDiv TRAIN teacher target sets",
        "",
        "This report is aggregate-only. TRAIN target IDs and raw trajectories remain local.",
        "",
        f"- contexts: {manifest['contexts']}",
        f"- sessions: {manifest['sessions']}",
        f"- frames: {manifest['frames']}",
        f"- selection: `{manifest['selection_version']}`",
        f"- alignment: `{manifest['alignment_policy']}`",
        f"- manifest SHA256: `{manifest['manifest_sha256']}`",
        "",
        "| teacher set | FRDiv | descriptor pair distance | AU range | VA range | distinct expression modes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"T0": "Random fixed paired+9", "T1": "Paired-constrained Max-FRDiv9",
              "T2": "Unconstrained Max-FRDiv10 diagnostic"}
    for key in ("T0", "T1", "T2"):
        row = summary[key]
        lines.append(
            f"| {labels[key]} | {row['FRDiv']:.6f} | {row['descriptor_pair_distance']:.6f} | "
            f"{row['au_range']:.6f} | {row['va_range']:.6f} | {row['distinct_expression_modes']:.3f} |"
        )
    lines.extend([
        "",
        f"T1/T0 FRDiv ratio: **{summary['T1_over_T0_FRDiv']:.4f}**",
        f"T2/T1 FRDiv ratio: **{summary['T2_over_T1_FRDiv']:.4f}**",
        "",
        "T1 membership is selected in trajectory space; descriptor prototypes only assign q1--q9 slot identities.",
        "T2 is a ceiling diagnostic and is not a teacher-training default.",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mam-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--exact-t1", action="store_true")
    parser.add_argument("--exact-t2", action="store_true")
    parser.add_argument("--exact-time-limit", type=float, default=300.0)
    parser.add_argument("--session-limit", type=int)
    args = parser.parse_args()
    if args.mam_root is not None:
        sys.path.insert(0, str(args.mam_root.resolve()))
    manifest = build_manifest(
        args.data_root, args.output, args.seed, args.frames,
        args.exact_t1, args.exact_t2, args.exact_time_limit,
        args.session_limit,
    )
    if args.report:
        write_sanitized_report(manifest, args.report)
    print(json.dumps({"contexts": manifest["contexts"],
                      "sessions": manifest["sessions"],
                      "summary": manifest["summary"],
                      "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
