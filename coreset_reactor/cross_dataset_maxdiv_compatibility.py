"""Diagnose conditional compatibility of frozen Max-FRDiv target sets.

This is a read-only TRAIN diagnostic.  It uses one implementation for
React2024 and React2025:

* all opposite-role GTs in a session are resampled to 750 frames;
* membership distance is mean squared 25-D trajectory distance;
* T0/T1 membership comes from the frozen rank-00 manifests;
* a frozen paired ConditionalREGNN anchor supplies source-conditioned
  compatibility distances.

The report deliberately separates session diversity from conditionality.  It
does not train a model and does not read validation/test GT.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from coreset_reactor.dataset_gt_oracle import frdiv as _official_frdiv
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.maxfrdiv_teacher_manifest import (
    _load_raw,
    _relative_resample,
    paired_farthest_indices,
)


FRD_GROUPS = ((0, 15, 1.0 / 15.0), (15, 17, 1.0), (17, 25, 1.0 / 8.0))
QUANTILES = (0.30, 0.50, 0.70, 0.90)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file_set(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pairwise_distance(reactions: np.ndarray) -> np.ndarray:
    flat = np.asarray(reactions, dtype=np.float64).reshape(len(reactions), -1)
    gram = flat @ flat.T
    squared = np.maximum(
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram,
        0.0,
    )
    np.fill_diagonal(squared, 0.0)
    return squared / float(flat.shape[1])


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: float("nan") for key in
                ("mean", "median", "std", "p10", "p25", "p50", "p75",
                 "p90", "p95", "max")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, .10)),
        "p25": float(np.quantile(values, .25)),
        "p50": float(np.quantile(values, .50)),
        "p75": float(np.quantile(values, .75)),
        "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)),
        "max": float(np.max(values)),
    }


def percentile(value: float, reference: np.ndarray) -> float:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(reference):
        return float("nan")
    return float(np.searchsorted(reference, value, side="right") / len(reference))


def _safe_spearman(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    values = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(targets)
    if int(mask.sum()) < 3:
        return {"rho": None, "pvalue": None, "n": int(mask.sum())}
    result = spearmanr(values[mask], targets[mask])
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue),
            "n": int(mask.sum())}


def _load_model(checkpoint: Path, device: torch.device):
    from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ConditionalREGNN(
        ConditionalREGNNConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    return model, payload


def _source_features(data_root: Path, source_id: str,
                     mean_face: torch.Tensor, std_face: torch.Tensor) -> tuple[torch.Tensor, ...]:
    base = data_root / "train"
    relative = Path(source_id)
    emotion = torch.from_numpy(np.load(
        base / "facial-attributes" / relative, allow_pickle=False
    ).astype(np.float32))
    audio = torch.from_numpy(np.load(
        base / "audio-features" / relative, allow_pickle=False
    ).astype(np.float32))
    face = torch.from_numpy(np.load(
        base / "coefficients" / relative, allow_pickle=False
    ).astype(np.float32))
    if face.ndim == 3 and face.shape[1] == 1:
        face = face[:, 0]
    if emotion.ndim != 2 or emotion.shape[1] != 25:
        raise ValueError(f"invalid emotion shape for {source_id}: {tuple(emotion.shape)}")
    if audio.ndim != 2 or audio.shape[1] != 768:
        raise ValueError(f"invalid audio shape for {source_id}: {tuple(audio.shape)}")
    if face.ndim != 2 or face.shape[1] != 58:
        raise ValueError(f"invalid 3DMM shape for {source_id}: {tuple(face.shape)}")
    face = (face - mean_face) / std_face.clamp_min(1e-8)
    total = min(len(emotion), len(audio), len(face))
    if total < 1:
        raise ValueError(f"empty source {source_id}")
    start = max(0, (total - 750) // 2)
    length = min(750, total - start)

    def padded(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(750, value.shape[1], dtype=torch.float32)
        result[:length] = value[start:start + length]
        return result

    return padded(audio), padded(emotion), padded(face), torch.tensor([length])


def _trajectory_from_path(path: Path) -> np.ndarray:
    value = _load_raw(path)
    return _relative_resample(value, 750).numpy().astype(np.float32)


def _rolling_frd(anchor: np.ndarray, target: np.ndarray) -> float:
    """Official weighted unconstrained DTW used by the repository's FRD."""
    from tools.compute_frd_resumable import rolling_dtw

    total = 0.0
    for start, end, weight in FRD_GROUPS:
        total += weight * float(rolling_dtw(
            np.ascontiguousarray(anchor[:, start:end], dtype=np.float32),
            np.ascontiguousarray(target[:, start:end], dtype=np.float32),
        ))
    return total


def _frdiv(distance: np.ndarray, indices: list[int]) -> float:
    return float(_official_frdiv(distance, indices))


def _opposite(role: str) -> str:
    if role == "speaker":
        return "listener"
    if role == "listener":
        return "speaker"
    raise ValueError(role)


def _manifest_records(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    return payload["mapping"]


def _session_cache(data_root: Path, target_role: str) -> dict[str, dict[str, Any]]:
    root = data_root / "train" / "facial-attributes" / target_role
    cache: dict[str, dict[str, Any]] = {}
    for session_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        paths = sorted(session_dir.glob("*.npy"))
        if len(paths) < 2:
            continue
        reactions = np.stack([_trajectory_from_path(path) for path in paths])
        descriptors = reaction_descriptor(torch.from_numpy(reactions)).numpy()
        cache[session_dir.name] = {
            "paths": paths,
            "ids": [path.relative_to(data_root / "train" / "facial-attributes").as_posix()
                    for path in paths],
            "reactions": reactions,
            "descriptors": descriptors,
            "distance": pairwise_distance(reactions),
        }
    return cache


def _selection_indices(selection: list[str], ids: list[str]) -> list[int]:
    lookup = {value: index for index, value in enumerate(ids)}
    missing = [value for value in selection if value not in lookup]
    if missing:
        raise KeyError(f"manifest targets missing from session pool: {missing[:3]}")
    return [lookup[value] for value in selection]


def _anchor_metrics(anchor: np.ndarray, target_reactions: np.ndarray,
                    selected: list[int], all_pair_values: np.ndarray,
                    dtw_executor: concurrent.futures.Executor | None = None) -> dict:
    anchor_distance = np.mean((target_reactions - anchor[None]) ** 2, axis=(1, 2))
    selected_alt = selected[1:]
    selected_values = anchor_distance[selected_alt]
    tasks = [(anchor, target_reactions[index]) for index in selected_alt]
    if dtw_executor is None:
        rolling = [_rolling_frd(left, right) for left, right in tasks]
    else:
        rolling = list(dtw_executor.map(
            lambda item: _rolling_frd(item[0], item[1]), tasks
        ))
    descriptor_anchor = reaction_descriptor(torch.from_numpy(anchor[None])).numpy()[0]
    target_descriptor = reaction_descriptor(torch.from_numpy(target_reactions)).numpy()
    descriptor_distance = np.sqrt(np.mean(
        (target_descriptor - descriptor_anchor[None]) ** 2, axis=1
    ))
    return {
        "trajectory_mean": float(np.mean(selected_values)),
        "trajectory_median": float(np.median(selected_values)),
        "trajectory_max": float(np.max(selected_values)),
        "trajectory_percentiles": [percentile(float(value), all_pair_values)
                                   for value in selected_values],
        "rolling_frd": float(np.mean(rolling)),
        "rolling_frd_values": [float(value) for value in rolling],
        "descriptor_mean": float(np.mean(descriptor_distance[selected_alt])),
        "descriptor_values": [float(descriptor_distance[index]) for index in selected_alt],
        "trajectory_values": [float(value) for value in selected_values],
    }


def _gated_selection(distance: np.ndarray, paired: int,
                     anchor_distance: np.ndarray, quantile: float) -> dict:
    alternatives = [index for index in range(len(distance)) if index != paired]
    ordered = sorted(alternatives, key=lambda index: (float(anchor_distance[index]), index))
    keep = max(9, int(math.ceil(len(ordered) * quantile)))
    eligible = ordered[:keep]
    allowed = [paired, *eligible]
    local = distance[np.ix_(allowed, allowed)]
    selected_local = paired_farthest_indices(local, 0, 10)
    selected = [allowed[index] for index in selected_local]
    return {"selected": selected, "eligible_count": len(eligible),
            "effective_quantile": len(eligible) / max(1, len(ordered))}


def _one_dataset(name: str, data_root: Path, t0_manifest: Path,
                 t1_manifest: Path, anchor_checkpoint: Path,
                 asset_mam_root: Path, device: torch.device,
                 max_contexts: int | None, dtw_workers: int) -> dict:
    facial_root = data_root / "train" / "facial-attributes"
    t0 = _manifest_records(t0_manifest)
    t1 = _manifest_records(t1_manifest)
    source_ids = sorted(set(t0) & set(t1))
    if max_contexts is not None:
        source_ids = source_ids[:max_contexts]
    if not source_ids:
        raise ValueError(f"no common source records for {name}")

    mean_face = torch.from_numpy(np.load(
        asset_mam_root / "external" / "FaceVerse" / "mean_face.npy",
    ).astype(np.float32))
    std_face = torch.from_numpy(np.load(
        asset_mam_root / "external" / "FaceVerse" / "std_face.npy",
    ).astype(np.float32))
    model, checkpoint = _load_model(anchor_checkpoint, device)
    role_cache = {
        "speaker": _session_cache(data_root, "speaker"),
        "listener": _session_cache(data_root, "listener"),
    }
    target_cache: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        role, session, _ = source_id.split("/", 2)
        target_role = _opposite(role)
        key = f"{target_role}/{session}"
        if key not in target_cache:
            if session not in role_cache[target_role]:
                raise KeyError(f"missing {target_role}/{session} target pool")
            target_cache[key] = role_cache[target_role][session]

    # Descriptor normalization is dataset-local and TRAIN-only.  The raw
    # distance is retained in every row so cross-dataset scaling is explicit.
    all_descriptors = np.concatenate([
        cache["descriptors"] for cache in target_cache.values()
    ], axis=0)
    descriptor_mean = all_descriptors.mean(0)
    descriptor_std = np.maximum(all_descriptors.std(0), .03)
    for cache in target_cache.values():
        cache["descriptors_scaled"] = (cache["descriptors"] - descriptor_mean) / descriptor_std

    rows: list[dict] = []
    executor = (concurrent.futures.ThreadPoolExecutor(max_workers=dtw_workers)
                if dtw_workers > 1 else None)
    try:
      with torch.inference_mode():
        for number, source_id in enumerate(source_ids, 1):
            role, session, _ = source_id.split("/", 2)
            target_role = _opposite(role)
            cache = target_cache[f"{target_role}/{session}"]
            ids = cache["ids"]
            distance = cache["distance"]
            all_pairs = distance[np.triu_indices(len(distance), k=1)]
            session_median = float(np.median(all_pairs))
            session_p90 = float(np.quantile(all_pairs, .90))
            source_audio, source_emotion, source_face, lengths = _source_features(
                data_root, source_id, mean_face, std_face
            )
            output = model(
                source_audio[None].to(device), source_emotion[None].to(device),
                source_face[None].to(device), lengths.to(device),
            )["prediction"].detach().cpu()
            valid = int(lengths.item())
            anchor = _relative_resample(output[0, :valid].float(), 750).numpy()
            anchor = official_prediction(torch.from_numpy(anchor[None])).numpy()[0]
            t0_indices = _selection_indices(t0[source_id], ids)
            t1_indices = _selection_indices(t1[source_id], ids)
            anchor_distance = np.mean((cache["reactions"] - anchor[None]) ** 2, axis=(1, 2))
            t0_metrics = _anchor_metrics(
                anchor, cache["reactions"], t0_indices, all_pairs, executor
            )
            t1_metrics = _anchor_metrics(
                anchor, cache["reactions"], t1_indices, all_pairs, executor
            )
            t1_internal = _frdiv(distance, t1_indices)
            t0_internal = _frdiv(distance, t0_indices)
            row = {
                "dataset": name,
                "source_id": source_id,
                "source_role": role,
                "target_role": target_role,
                "session_id": session,
                "candidate_count": len(ids),
                "session_geometry": {
                    "all_pair": summarize(all_pairs),
                    "session_scale_median": session_median,
                    "session_tail_p90_over_median": session_p90 / max(session_median, 1e-12),
                },
                "T0": {"FRDiv": t0_internal, **t0_metrics},
                "T1": {"FRDiv": t1_internal, **t1_metrics},
                "T1_diversity_efficiency": t1_internal / max(t1_metrics["trajectory_mean"], 1e-12),
                "T0_CI": t0_metrics["trajectory_mean"] / max(session_median, 1e-12),
                "T1_CI": t1_metrics["trajectory_mean"] / max(session_median, 1e-12),
                "T0_TailRate90": float(np.mean(np.asarray(t0_metrics["trajectory_values"]) > session_p90)),
                "T1_TailRate90": float(np.mean(np.asarray(t1_metrics["trajectory_values"]) > session_p90)),
                "ranked": {
                    f"q{rank}": {
                        "trajectory_distance": float(anchor_distance[t1_indices[rank]]),
                        "trajectory_percentile": t1_metrics["trajectory_percentiles"][rank - 1],
                        "rolling_frd": t1_metrics["rolling_frd_values"][rank - 1],
                        "descriptor_distance": t1_metrics["descriptor_values"][rank - 1],
                    }
                    for rank in range(1, 10)
                },
                "gated": {},
            }
            for quantile in QUANTILES:
                gated = _gated_selection(distance, t1_indices[0], anchor_distance, quantile)
                selected = gated["selected"]
                row["gated"][f"Q{int(quantile * 100):02d}"] = {
                    **gated,
                    "FRDiv": _frdiv(distance, selected),
                    "mean_anchor_distance": float(np.mean(anchor_distance[selected[1:]])),
                    "CI": float(np.mean(anchor_distance[selected[1:]]) / max(session_median, 1e-12)),
                    "TailRate90": float(np.mean(anchor_distance[selected[1:]] > session_p90)),
                }
            rows.append(row)
            if number == 1 or number % 250 == 0 or number == len(source_ids):
                print(json.dumps({"dataset": name, "completed": number,
                                  "total": len(source_ids)}), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return {
        "dataset": name,
        "contexts": len(rows),
        "data_root": str(data_root),
        "data_fingerprint": sha256_file_set(
            data_root / "train" / "facial-attributes",
            [path for path in (data_root / "train" / "facial-attributes").glob("*/*/*.npy")],
        ),
        "t0_manifest_sha256": sha256_file(t0_manifest),
        "t1_manifest_sha256": sha256_file(t1_manifest),
        "anchor_checkpoint": str(anchor_checkpoint),
        "anchor_checkpoint_sha256": sha256_file(anchor_checkpoint),
        "anchor_model_config": checkpoint["model_config"],
        "rows": rows,
    }


def _aggregate(dataset: dict) -> dict:
    rows = dataset["rows"]
    def vals(key: str, arm: str | None = None) -> np.ndarray:
        if arm is None:
            return np.asarray([row[key] for row in rows], dtype=np.float64)
        return np.asarray([row[arm][key] for row in rows], dtype=np.float64)

    output = {"contexts": len(rows)}
    for arm in ("T0", "T1"):
        output[arm] = {
            "target_FRDiv": float(np.mean(vals("FRDiv", arm))),
            "anchor_deviation": summarize(vals("trajectory_mean", arm)),
            "anchor_CI": summarize(vals(f"{arm}_CI")),
            "TailRate90": float(np.mean(vals(f"{arm}_TailRate90"))),
            "rolling_FRD": float(np.mean(vals("rolling_frd", arm))),
            "descriptor_compatibility": float(np.mean(vals("descriptor_mean", arm))),
            "diversity_efficiency": float(np.mean(vals("FRDiv", arm) /
                                                  np.maximum(vals("trajectory_mean", arm), 1e-12))),
        }
        output[arm]["session_tail"] = summarize(
            np.asarray([row["session_geometry"]["session_tail_p90_over_median"] for row in rows])
        )
    output["T1_minus_T0"] = {
        "target_FRDiv": output["T1"]["target_FRDiv"] - output["T0"]["target_FRDiv"],
        "anchor_deviation": (output["T1"]["anchor_deviation"]["mean"] -
                              output["T0"]["anchor_deviation"]["mean"]),
        "CI": output["T1"]["anchor_CI"]["mean"] - output["T0"]["anchor_CI"]["mean"],
        "TailRate90": output["T1"]["TailRate90"] - output["T0"]["TailRate90"],
    }
    output["ranked_T1"] = {}
    for rank in range(1, 10):
        prefix = f"q{rank}"
        output["ranked_T1"][prefix] = {
            key: float(np.mean([row["ranked"][prefix][key] for row in rows]))
            for key in ("trajectory_distance", "trajectory_percentile",
                        "rolling_frd", "descriptor_distance")
        }
    output["gated"] = {}
    for quantile in QUANTILES:
        key = f"Q{int(quantile * 100):02d}"
        output["gated"][key] = {
            metric: float(np.mean([row["gated"][key][metric] for row in rows]))
            for metric in ("FRDiv", "mean_anchor_distance", "CI", "TailRate90")
        }
    return output


def _report(result: dict) -> str:
    lines = [
        "# Cross-dataset Max-FRDiv compatibility diagnostic", "",
        "This is a TRAIN-only diagnostic. No model was retrained and no VAL/TEST GT was used.",
        "All cross-listener GT trajectories use `relative_time_resample_reaction(..., 750)`.",
        "Membership distance is mean squared 25-D trajectory distance; compatibility FRD is",
        "the repository's weighted rolling-DTW implementation on the selected q1--q9 targets.", "",
        "## Main comparison", "",
        "| Dataset | Arm | Target FRDiv | Anchor deviation | Normalized CI | TailRate90 | Rolling anchor FRD | Descriptor compatibility |", 
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in result["datasets"]:
        summary = dataset["aggregate"]
        for arm in ("T0", "T1"):
            row = summary[arm]
            lines.append(
                f"| {dataset['dataset']} | {arm} | {row['target_FRDiv']:.6f} | "
                f"{row['anchor_deviation']['mean']:.6f} | {row['anchor_CI']['mean']:.6f} | "
                f"{row['TailRate90']:.4f} | {row['rolling_FRD']:.6f} | "
                f"{row['descriptor_compatibility']:.6f} |"
            )
    lines += ["", "## Session geometry", "",
              "| Dataset | all-pair mean | median | std | P10 | P90 | P95 | max | P90/median |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in result["datasets"]:
        # Geometry is repeated per source; summarize the per-session-tail proxy.
        rows = dataset["rows"]
        values = np.asarray([row["session_geometry"]["all_pair"] for row in rows], dtype=object)
        tail = dataset["aggregate"]["T1"]["session_tail"]
        lines.append(
            f"| {dataset['dataset']} | {np.mean([v['mean'] for v in values]):.6f} | "
            f"{np.mean([v['median'] for v in values]):.6f} | {np.mean([v['std'] for v in values]):.6f} | "
            f"{np.mean([v['p10'] for v in values]):.6f} | {np.mean([v['p90'] for v in values]):.6f} | "
            f"{np.mean([v['p95'] for v in values]):.6f} | {np.mean([v['max'] for v in values]):.6f} | "
            f"{tail['mean']:.4f} |"
        )
    lines += ["", "## T1 rank distribution", "",
              "| Dataset | rank | anchor trajectory distance | session percentile | rolling FRD | descriptor distance |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in result["datasets"]:
        for rank, row in dataset["aggregate"]["ranked_T1"].items():
            lines.append(
                f"| {dataset['dataset']} | {rank} | {row['trajectory_distance']:.6f} | "
                f"{row['trajectory_percentile']:.4f} | {row['rolling_frd']:.6f} | "
                f"{row['descriptor_distance']:.6f} |"
            )
    lines += ["", "## Compatibility-gated target simulation", "",
              "The gate ranks all same-session alternatives by frozen-anchor trajectory distance,",
              "keeps the nearest Q30/Q50/Q70/Q90 pool (with a minimum of nine alternatives),",
              "then runs the same paired Max-FRDiv selection without training.", "",
              "| Dataset | gate | Target FRDiv | mean anchor distance | CI | TailRate90 |",
              "|---|---|---:|---:|---:|---:|"]
    for dataset in result["datasets"]:
        for gate, row in dataset["aggregate"]["gated"].items():
            lines.append(
                f"| {dataset['dataset']} | {gate} | {row['FRDiv']:.6f} | "
                f"{row['mean_anchor_distance']:.6f} | {row['CI']:.6f} | "
                f"{row['TailRate90']:.4f} |"
            )
    lines += ["", "## Teacher-result linkage", "",
              "The frozen teacher result aggregates are recorded separately because the existing",
              "R24/R25 evaluation JSONs do not retain per-context teacher rows. Therefore this",
              "diagnostic reports target-level compatibility and does not claim per-context",
              "Spearman correlations with ΔFRC/ΔFRD. A follow-up per-context evaluator can be",
              "attached without changing this diagnostic or retraining any model.", ""]
    for dataset in result["datasets"]:
        lines.append(
            f"- `{dataset['dataset']}`: {dataset['contexts']} TRAIN sources; "
            f"anchor `{dataset['anchor_checkpoint']}`; T0 manifest `{dataset['t0_manifest_sha256']}`; "
            f"T1 manifest `{dataset['t1_manifest_sha256']}`."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("react2024", "react2025"):
        parser.add_argument(f"--{name}-root", type=Path, required=True)
        parser.add_argument(f"--{name}-t0-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-t1-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-anchor", type=Path, required=True)
    parser.add_argument("--asset-mam-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--dtw-workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    result = {
        "protocol": {
            "frames": 750,
            "membership_distance": "mean squared 25-D trajectory distance after relative_time_resample_reaction",
            "compatibility_frd": "weighted rolling unconstrained DTW on q1-q9",
            "descriptor": "602-D reaction_descriptor with TRAIN-only dataset-local scaling",
            "max_contexts": args.max_contexts,
        },
        "datasets": [],
    }
    for name in ("react2024", "react2025"):
        dataset = _one_dataset(
            name=name,
            data_root=getattr(args, f"{name}_root").resolve(),
            t0_manifest=getattr(args, f"{name}_t0_manifest").resolve(),
            t1_manifest=getattr(args, f"{name}_t1_manifest").resolve(),
            anchor_checkpoint=getattr(args, f"{name}_anchor").resolve(),
            asset_mam_root=args.asset_mam_root.resolve(),
            device=device,
            max_contexts=args.max_contexts,
            dtw_workers=args.dtw_workers,
        )
        dataset["aggregate"] = _aggregate(dataset)
        result["datasets"].append(dataset)
    result["report"] = {"note": "per-context teacher rows unavailable in frozen aggregate evaluation JSONs"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report(result))
    print(json.dumps({
        "output": str(args.output), "report": str(args.report),
        "datasets": {item["dataset"]: item["contexts"] for item in result["datasets"]},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
