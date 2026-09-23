"""Build TRAIN-only source-compatible Max-FRDiv target manifests.

The frozen T0/T1 experiment showed that pure same-session Max-FRDiv can select
extreme reactions that are poor conditional targets.  This module keeps the
paired anchor fixed, filters opposite-role same-session GTs by a TRAIN-only
distance to that paired GT, and then runs the existing paired farthest-sum plus
one-swap selector inside the legal pool.

Compatibility and membership are deliberately separate:

* descriptor/trajectory distance to the paired GT decides eligibility;
* the official trajectory-space distance decides the nine-member Max-FRDiv set;
* TRAIN-only global descriptor prototypes assign stable q1--q9 identities.

No VAL data is read by this module.  The JSON manifest contains target IDs for
local training, while the optional Markdown report is aggregate-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.dataset_gt_oracle import frdiv, official_distance_matrix
from coreset_reactor.fixed_pair_manifest import assign_fixed_random
from coreset_reactor.maxfrdiv_teacher_manifest import (
    _assign_canonical_slots,
    _fit_global_prototypes,
    _load_raw,
    _relative_resample,
    _stats_summary,
    paired_farthest_indices,
    _descriptor_statistics,
)


QUANTILES = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
GATE_NAMES = tuple(
    f"{family}_Q{int(round(quantile * 100)):02d}"
    for family in ("DESC", "TRAJ")
    for quantile in QUANTILES
)
SCHEMA_VERSION = 1
SELECTION_VERSION = "paired-distance-gated-relative-time-resample-v1"


def selection_source_sha256() -> str:
    """Hash this selector implementation for experiment provenance."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _rms_distance(values: np.ndarray, paired_index: int) -> np.ndarray:
    if values.ndim != 2 or not 0 <= paired_index < len(values):
        raise ValueError("values must be [N,D] and paired_index must be legal")
    paired = values[paired_index]
    return np.sqrt(np.mean((values - paired[None]) ** 2, axis=1))


def eligible_indices(
    scores: np.ndarray,
    paired_index: int,
    quantile: float,
    minimum_alternatives: int = 9,
) -> dict:
    """Return deterministic nearest candidates with a minimum-count guard.

    Quantiles are applied only to non-paired candidates.  If a requested
    quantile contains fewer than nine alternatives, the threshold is widened
    to the nine nearest alternatives and the effective quantile is recorded.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not 0 <= paired_index < len(scores):
        raise ValueError("scores must be one-dimensional and paired_index legal")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0,1]")
    alternatives = [i for i in range(len(scores)) if i != paired_index]
    if len(alternatives) < minimum_alternatives:
        raise ValueError("session has fewer than the required alternatives")
    ordered = sorted(alternatives, key=lambda i: (float(scores[i]), i))
    requested_threshold = float(np.quantile(scores[alternatives], quantile))
    legal = [i for i in ordered if scores[i] <= requested_threshold + 1e-12]
    widened = len(legal) < minimum_alternatives
    if widened:
        legal = ordered[:minimum_alternatives]
    effective_threshold = float(max(scores[i] for i in legal))
    return {
        "requested_quantile": float(quantile),
        "requested_threshold": requested_threshold,
        "effective_threshold": effective_threshold,
        "eligible_indices": legal,
        "eligible_count": len(legal),
        "alternative_count": len(alternatives),
        "effective_quantile": float(len(legal) / len(alternatives)),
        "minimum_count_widened": bool(widened),
    }


def _select_gated(
    distance: np.ndarray,
    paired_index: int,
    legal_alternatives: list[int],
    values: np.ndarray,
    centers: np.ndarray,
) -> list[int]:
    allowed = [paired_index, *sorted(set(legal_alternatives))]
    if len(allowed) < 10:
        raise ValueError("gated pool must contain paired target plus 9 alternatives")
    local_distance = distance[np.ix_(allowed, allowed)]
    local_selected = paired_farthest_indices(local_distance, 0, 10)
    selected = [allowed[index] for index in local_selected]
    return _assign_canonical_slots(selected, paired_index, values, centers)


def _pack_stats(
    raw_descriptors: np.ndarray,
    selected: list[int],
    paired_index: int,
    reactions: torch.Tensor,
    descriptor_values: np.ndarray,
    descriptor_scores: np.ndarray,
    trajectory_scores: np.ndarray,
    eligible: dict | None,
) -> dict:
    stats = _descriptor_statistics(
        descriptor_values, selected, reactions,
        np.zeros(descriptor_values.shape[1]), np.ones(descriptor_values.shape[1]),
    )
    # The descriptor values passed above are already TRAIN-scaled.
    alternatives = [index for index in selected if index != paired_index]
    stats["descriptor_pair_distance"] = float(np.mean([
        np.sqrt(np.mean((descriptor_values[index] -
                         descriptor_values[paired_index]) ** 2))
        for index in alternatives
    ]))
    stats["mean_distance_to_paired_descriptor"] = float(
        np.mean([descriptor_scores[index] for index in alternatives])
    )
    stats["mean_distance_to_paired_trajectory"] = float(
        np.mean([trajectory_scores[index] for index in alternatives])
    )
    if eligible is not None:
        stats.update({
            "requested_quantile": eligible["requested_quantile"],
            "requested_threshold": eligible["requested_threshold"],
            "effective_threshold": eligible["effective_threshold"],
            "eligible_count": eligible["eligible_count"],
            "alternative_count": eligible["alternative_count"],
            "effective_quantile": eligible["effective_quantile"],
            "minimum_count_widened": eligible["minimum_count_widened"],
        })
    return stats


def _source_paths(data_root: Path, role: str) -> list[Path]:
    root = data_root / "train" / "facial-attributes" / role
    paths = sorted(root.glob("*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"no TRAIN {role} facial attributes under {root}")
    return paths


def _target_cache(data_root: Path, role: str, frames: int) -> dict[str, dict]:
    cache = {}
    for session_dir in sorted(
        (data_root / "train" / "facial-attributes" / role).glob("*")
    ):
        paths = sorted(session_dir.glob("*.npy"))
        if len(paths) < 10:
            continue
        raw = [_load_raw(path) for path in paths]
        reactions = torch.stack([_relative_resample(value, frames) for value in raw])
        raw_descriptors = np.stack([
            # Keep the descriptor implementation identical to frozen T1.
            __import__("coreset_reactor.dataset", fromlist=["full_reaction_descriptor"])
            .full_reaction_descriptor(value).numpy()
            for value in raw
        ])
        cache[session_dir.name] = {
            "paths": paths,
            "ids": [path.relative_to(data_root / "train" / "facial-attributes").as_posix()
                    for path in paths],
            "reactions": reactions,
            "raw_descriptors": raw_descriptors,
            "distance": official_distance_matrix(reactions),
        }
    if not cache:
        raise ValueError(f"no TRAIN {role} session has at least ten GTs")
    return cache


def _pack_selection(
    arm: str,
    target_ids: list[str],
    selected: list[int],
    paired_index: int,
    raw_descriptors: np.ndarray,
    reactions: torch.Tensor,
    values: np.ndarray,
    descriptor_scores: np.ndarray,
    trajectory_scores: np.ndarray,
    eligible: dict | None,
) -> dict:
    return {
        "target_ids": [target_ids[index] for index in selected],
        "membership_indices": selected,
        "selection_method": arm,
        "stats": _pack_stats(
            raw_descriptors, selected, paired_index, reactions, values,
            descriptor_scores, trajectory_scores, eligible,
        ),
    }


def build_manifest(
    data_root: Path,
    output: Path,
    seed: int = 1,
    frames: int = 750,
    source_roles: tuple[str, ...] = ("speaker",),
    frozen_manifest: Path | None = None,
    session_limit: int | None = None,
) -> dict:
    data_root = data_root.resolve()
    if not source_roles or any(role not in {"speaker", "listener"}
                               for role in source_roles):
        raise ValueError("source_roles must contain speaker and/or listener")
    target_roles = {"speaker": "listener", "listener": "speaker"}
    descriptor_mean, descriptor_std, centers, data_sha256 = _fit_global_prototypes(
        data_root, n_modes=9
    )
    target_cache = {
        role: _target_cache(data_root, role, frames)
        for role in {target_roles[source] for source in source_roles}
    }
    gate_specs = [("DESC", q) for q in QUANTILES]
    gate_specs += [("TRAJ", q) for q in QUANTILES]
    records: list[dict] = []
    source_count_by_role: dict[str, int] = {}
    source_root = data_root / "train" / "facial-attributes"
    for source_role in source_roles:
        target_role = target_roles[source_role]
        sources = _source_paths(data_root, source_role)
        sessions = sorted({path.parent.name for path in sources})
        if session_limit is not None:
            sessions = sessions[:session_limit]
        source_count_by_role[source_role] = 0
        for session in sessions:
            if session not in target_cache[target_role]:
                raise ValueError(f"missing target session {target_role}/{session}")
            target = target_cache[target_role][session]
            target_ids = target["ids"]
            target_lookup = {value: index for index, value in enumerate(target_ids)}
            for source in [p for p in sources if p.parent.name == session]:
                source_id = source.relative_to(source_root).as_posix()
                paired_id = f"{target_role}/{session}/{source.name}"
                if paired_id not in target_lookup:
                    raise ValueError(f"missing paired target {paired_id}")
                paired_index = target_lookup[paired_id]
                distance = target["distance"]
                raw_descriptors = target["raw_descriptors"]
                values = (raw_descriptors - descriptor_mean) / descriptor_std
                descriptor_scores = _rms_distance(values, paired_index)
                trajectory_scores = distance[paired_index].copy()
                t0_ids, _ = assign_fixed_random(
                    target_ids, paired_id, source_id, seed, 10, True
                )
                t0 = [target_lookup[item] for item in t0_ids]
                t1 = paired_farthest_indices(distance, paired_index, 10)
                t1 = _assign_canonical_slots(t1, paired_index, values, centers)
                arms = {
                    "T0": _pack_selection(
                        "random_fixed_pair", target_ids, t0, paired_index,
                        raw_descriptors, target["reactions"], values,
                        descriptor_scores, trajectory_scores, None,
                    ),
                    "T1": _pack_selection(
                        "paired_approximate_farthest_sum_swap", target_ids, t1,
                        paired_index, raw_descriptors, target["reactions"],
                        values, descriptor_scores, trajectory_scores, None,
                    ),
                }
                for family, quantile in gate_specs:
                    scores = (descriptor_scores if family == "DESC"
                              else trajectory_scores)
                    eligible = eligible_indices(scores, paired_index, quantile)
                    selected = _select_gated(
                        distance, paired_index, eligible["eligible_indices"],
                        values, centers,
                    )
                    name = f"{family}_Q{int(round(quantile * 100)):02d}"
                    arms[name] = _pack_selection(
                        name, target_ids, selected, paired_index,
                        raw_descriptors, target["reactions"], values,
                        descriptor_scores, trajectory_scores, eligible,
                    )
                records.append({
                    "sample_id": source_id,
                    "source_role": source_role,
                    "target_role": target_role,
                    "session_id": session,
                    "paired_target_id": paired_id,
                    "candidate_count": len(target_ids),
                    **arms,
                })
                source_count_by_role[source_role] += 1
            print(json.dumps({"source_role": source_role, "session": session,
                              "completed": len(records)}), flush=True)

    summary = {
        arm: _stats_summary(records, arm)
        for arm in ("T0", "T1", *GATE_NAMES)
    }
    for arm in GATE_NAMES:
        summary[arm]["FRDiv_over_T0"] = summary[arm]["FRDiv"] / max(
            summary["T0"]["FRDiv"], 1e-12
        )
        summary[arm]["mean_eligible_pool"] = float(np.mean([
            row[arm]["stats"]["eligible_count"] for row in records
        ]))
        summary[arm]["mean_effective_quantile"] = float(np.mean([
            row[arm]["stats"]["effective_quantile"] for row in records
        ]))
        summary[arm]["widened_fraction"] = float(np.mean([
            row[arm]["stats"]["minimum_count_widened"] for row in records
        ]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "source_roles": list(source_roles),
        "target_roles": sorted({target_roles[role] for role in source_roles}),
        "frames": frames,
        "selection_version": SELECTION_VERSION,
        "alignment_policy": "relative_time_masked",
        "compatibility_metrics": {
            "descriptor": "TRAIN-scaled 602-D descriptor RMS distance to paired GT",
            "trajectory": "mean squared 25-D trajectory distance after relative_time_resample_reaction",
        },
        "membership_distance": "official trajectory-space mean squared 25-D distance",
        "seed": seed,
        "data_sha256": data_sha256,
        "selection_source_sha256": selection_source_sha256(),
        "frozen_manifest_sha256": (
            hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()
            if frozen_manifest is not None else None
        ),
        "prototype_policy": "TRAIN-only global KMeans(9) descriptor prototypes; Hungarian slot assignment",
        "descriptor_mean": descriptor_mean.tolist(),
        "descriptor_std": descriptor_std.tolist(),
        "prototype_centers": centers.tolist(),
        "contexts": len(records),
        "source_count_by_role": source_count_by_role,
        "sessions": len({row["session_id"] for row in records}),
        "gate_quantiles": list(QUANTILES),
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
    for arm in ("T0", "T1", *GATE_NAMES):
        rows = [record[arm]["stats"] for record in manifest["records"]]
        summary[arm].setdefault(
            "mean_distance_to_paired_descriptor",
            float(np.mean([row["mean_distance_to_paired_descriptor"]
                           for row in rows])),
        )
        summary[arm].setdefault(
            "mean_distance_to_paired_trajectory",
            float(np.mean([row["mean_distance_to_paired_trajectory"]
                           for row in rows])),
        )
    lines = [
        "# Quality-gated Max-FRDiv TRAIN target frontier",
        "",
        "This report is aggregate-only. TRAIN target IDs and raw trajectories remain local.",
        "",
        f"- source roles: `{', '.join(manifest['source_roles'])}`",
        f"- contexts: {manifest['contexts']}",
        f"- sessions: {manifest['sessions']}",
        f"- frames: {manifest['frames']}",
        f"- alignment: `{manifest['alignment_policy']}`",
        f"- selection source SHA256: `{manifest['selection_source_sha256']}`",
        f"- frozen T0/T1 source manifest SHA256: `{manifest['frozen_manifest_sha256']}`",
        f"- generated manifest SHA256: `{manifest['manifest_sha256']}`",
        "",
        "Descriptor and trajectory gates are separate. Within each eligible pool,",
        "membership is still chosen by the frozen paired farthest-sum plus one-swap",
        "trajectory-space algorithm; Hungarian matching only assigns q1--q9 identities.",
        "",
        "| Gate | Target FRDiv | vs T0 | Mean desc distance to paired | Mean traj distance to paired | AU range | VA range | Expr modes | Mean eligible pool | Effective quantile | Widened fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("T0 random", summary["T0"], None),
            ("T1 pure Max-FRDiv", summary["T1"], None)]
    rows += [(name, summary[name], name) for name in GATE_NAMES]
    for label, row, name in rows:
        ratio = "1.000x" if name is None and label == "T0 random" else (
            "—" if name is None else f"{row['FRDiv_over_T0']:.3f}x"
        )
        pool = "—" if name is None else f"{row['mean_eligible_pool']:.2f}"
        effective = "—" if name is None else f"{row['mean_effective_quantile']:.3f}"
        widened = "—" if name is None else f"{row['widened_fraction']:.3f}"
        lines.append(
            f"| {label} | {row['FRDiv']:.6f} | {ratio} | "
            f"{row['mean_distance_to_paired_descriptor']:.6f} | "
            f"{row['mean_distance_to_paired_trajectory']:.6f} | "
            f"{row['au_range']:.6f} | {row['va_range']:.6f} | "
            f"{row['distinct_expression_modes']:.3f} | {pool} | "
            f"{effective} | {widened} |"
        )
    lines.extend([
        "",
        "The T0/T1 rows are frozen endpoints for audit comparison. No VAL GT is",
        "used for gate thresholds or target selection. A gate with fewer than nine",
        "legal alternatives is widened to the nine nearest alternatives and marked",
        "in `widened fraction`; it never silently falls back to random selection.",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--source-role", choices=("speaker", "listener", "both"),
                        default="speaker")
    parser.add_argument("--session-limit", type=int)
    args = parser.parse_args()
    source_roles = ("speaker", "listener") if args.source_role == "both" else (args.source_role,)
    manifest = build_manifest(
        args.data_root, args.output, args.seed, args.frames, source_roles,
        args.frozen_manifest, args.session_limit,
    )
    if args.report:
        write_sanitized_report(manifest, args.report)
    print(json.dumps({"contexts": manifest["contexts"],
                      "source_count_by_role": manifest["source_count_by_role"],
                      "summary": manifest["summary"],
                      "manifest_sha256": manifest["manifest_sha256"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
