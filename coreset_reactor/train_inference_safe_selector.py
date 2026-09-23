"""Train a GT-free candidate selector on TRAIN labels and audit on VAL571.

The selector is trained from all-session TRAIN quality labels, but inference
features contain only the speaker, candidate trajectory summaries, source
length, and frozen-generator provenance.  VAL GT is used only for the final
audit.  The selector makes at most one replacement of the ten Basic-KD
baseline candidates, using prediction-only diversity as its objective.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from coreset_reactor.certified_select import geometry, selection_value
from coreset_reactor.inference_safe_one_swap import pair_features
from coreset_reactor.inference_safe_selector import (
    atomic_json, balanced_session_folds, trajectory_summary,
)
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.run_certified_oracle import RUNS, SOURCES, build_pool


LABELS = ["basic", "distance", "direction"]
DEFAULT_TRAIN_ROOT = RUNS / "train_selector_labels_e60_center750"


def load_train_shards(paths: list[Path]):
    """Load contiguous shard outputs and verify their provenance/order."""
    if not paths:
        paths = sorted(RUNS.glob("train_selector_labels_e60_center750_gpu*"))
    if not paths:
        raise FileNotFoundError("No TRAIN selector-label shards found")
    chunks = []
    for root in paths:
        manifest = json.loads((root / "manifest.json").read_text())
        payload = torch.load(root / "train_predictions.pt", map_location="cpu",
                             weights_only=False)
        start, stop = int(manifest["context_start"]), int(manifest["context_stop"])
        quality = sorted(root.glob("quality_*.npz"))
        if len(quality) != stop - start or len(payload["clip_ids"]) != stop - start:
            raise ValueError(f"Incomplete TRAIN shard: {root}")
        chunks.append((start, stop, payload, quality, manifest))
    chunks.sort(key=lambda item: item[0])
    expected = chunks[0][0]
    soft_parts = [[] for _ in LABELS]
    quality_files: list[Path] = []
    clip_ids: list[str] = []
    manifests = []
    for start, stop, payload, quality, manifest in chunks:
        if start != expected:
            raise ValueError(f"TRAIN shards are not contiguous at {start}, expected {expected}")
        expected = stop
        if payload.get("split") != "train" or payload.get("crop_mode") != "center":
            raise ValueError("TRAIN shard protocol mismatch")
        for model, value in enumerate(payload["soft_predictions"]):
            # Shards store predictions as float16 to keep the cache small.
            # Residual expansion uses a 1e-12 expression normalizer, which
            # underflows in float16 and can turn an all-zero expression frame
            # into NaN.  Re-enter the numerical metric path in float32.
            soft_parts[model].append(value.float())
        quality_files.extend(quality)
        clip_ids.extend(payload["clip_ids"])
        manifests.append(manifest)
    return [torch.cat(parts, dim=0) for parts in soft_parts], quality_files, clip_ids, manifests


def metadata_from_origins(origins, source_length: int) -> np.ndarray:
    metadata = np.zeros((60, 15), dtype=np.float32)
    for row, origin_list in enumerate(origins):
        origin = origin_list[0]
        metadata[row, LABELS.index(origin["model"])] = 1.0
        metadata[row, 3 + int(origin["slot"])] = 1.0
        metadata[row, 13] = float(origin["scale"] == 1.5)
        metadata[row, 14] = source_length / 750.0
    return metadata


def make_pairs_with_length(soft, speaker: np.ndarray, source_length: int):
    pool, origins, raw_count, unique = build_pool(soft, LABELS)
    if len(pool) != 60 or raw_count != 30 or unique != 60:
        raise ValueError(f"Expected unique E60 pool, got raw={raw_count}, unique={unique}")
    candidate = np.stack([trajectory_summary(value.numpy()) for value in pool])
    metadata = metadata_from_origins(origins, source_length)
    points = pool.flatten(1).double().numpy() / np.sqrt(pool.shape[1] * pool.shape[2])
    distance = geometry(points)[2]
    features, pairs = pair_features(candidate, speaker, metadata, distance)
    return pool, origins, distance, features, pairs


def load_val_sources():
    values, soft = [], []
    for _, path in SOURCES:
        report = json.loads(path.read_text())
        payload = torch.load(path.with_suffix(".predictions.pt"), map_location="cpu",
                             weights_only=False)
        if payload["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("VAL prediction provenance mismatch")
        values.append(report)
        soft.append([value.float() for value in payload["soft_predictions"]])
    clip_ids = [row["clip_id"] for row in values[0]["rows"]]
    if len(clip_ids) != 571 or any([row["clip_id"] for row in value["rows"]] != clip_ids
                                    for value in values[1:]):
        raise ValueError("VAL source contexts do not match")
    return soft, clip_ids


def load_val_quality():
    oracle_root = RUNS / "certified_select10_val571_seed1234"
    quality = {}
    for shard in oracle_root.glob("shard_*"):
        for path in shard.glob("quality_*.npz"):
            quality[int(path.stem.split("_")[1])] = path
    if set(quality) != set(range(571)):
        raise ValueError("VAL quality cache incomplete")
    return quality


def fit_extra_trees(features, targets, contexts, folds, trees, jobs, seed):
    train_contexts = np.flatnonzero(folds != 0)
    calibration_contexts = np.flatnonzero(folds == 0)
    rng = np.random.default_rng(seed)
    sampled = []
    for index in train_contexts:
        chosen = rng.choice(features[index].shape[0], 150, replace=False)
        sampled.append((features[index][chosen], targets[index][chosen]))
    train_x = np.concatenate([x for x, _ in sampled], axis=0)
    train_y = np.concatenate([y for _, y in sampled], axis=0)
    mean, std = train_y.mean(0), train_y.std(0).clip(min=1e-6)
    model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=12,
                                max_features=.75, max_depth=24, n_jobs=jobs,
                                random_state=seed)
    model.fit(train_x, (train_y - mean) / std)
    cal_x = np.concatenate([features[index] for index in calibration_contexts], axis=0)
    cal_y = np.concatenate([targets[index] for index in calibration_contexts], axis=0)
    cal_pred = model.predict(cal_x) * std + mean
    residual = np.abs(cal_y - cal_pred)
    eps = {str(q): np.quantile(residual, q, axis=0).tolist()
           for q in (0.0, .5, .75, .9, .95, .99)}
    return model, mean, std, eps, {
        "train_contexts": int(len(train_contexts)),
        "calibration_contexts": int(len(calibration_contexts)),
        "calibration_sessions": sorted(set(contexts[index] for index in calibration_contexts)),
        "train_pairs": int(len(train_y)),
    }


def aggregate(records, policy):
    selected = [row["policies"][policy] for row in records]
    baseline = [row["baseline"] for row in records]
    keys = ("FRC", "exact_FRD", "FRDiv")
    base = {key: float(np.mean([row[key] for row in baseline])) for key in keys}
    out = {key: float(np.mean([row[key] for row in selected])) for key in keys}
    sources = Counter()
    for row in records:
        for index in row["policies"][policy]["selected"]:
            sources.update([row["origins"][index][0]["model"]])
    return {
        "baseline": base, "selected": out,
        "delta": {key: out[key] - base[key] for key in keys},
        "strict_quality_pass_contexts": int(sum(row["policies"][policy]["strict_quality_pass"]
                                                for row in records)),
        "positive_FRDiv_contexts": int(sum(row["policies"][policy]["FRDiv"] > row["baseline"]["FRDiv"] + 1e-12
                                           for row in records)),
        "changed_contexts": int(sum(row["policies"][policy]["new"] is not None for row in records)),
        "selected_source_counts": dict(sources),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-shard", type=Path, action="append")
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()

    train_soft, train_quality_files, train_clip_ids, manifests = load_train_shards(args.train_shard or [])
    train_data = PairedReactionDataset(
        Path("/public/zhaowenjie/react2025_new/data"), "train", 750, crop_mode="center",
        normalization_dir=Path("/home/zhaowenjie/react2025_new/mam_reactor/external/FaceVerse"))
    if len(train_data) != len(train_clip_ids):
        raise ValueError("TRAIN label count does not match dataset")
    train_features, train_targets, train_sessions = [], [], []
    for index in range(len(train_data)):
        sample = train_data[index]
        if sample["clip_id"] != train_clip_ids[index]:
            raise ValueError(f"TRAIN dataset order mismatch at {index}")
        length = int(sample["source_lengths"])
        speaker = trajectory_summary(sample["speaker_emotion"][:length].numpy())
        _, _, _, x, pairs = make_pairs_with_length([value[index] for value in train_soft], speaker, length)
        with np.load(train_quality_files[index], allow_pickle=False) as quality:
            y = np.stack((quality["frc"], quality["frd"]), axis=1)
        dy = np.asarray([[y[new, 0] - y[old, 0], y[new, 1] - y[old, 1]]
                         for old, new, _ in pairs], dtype=np.float32)
        train_features.append(x)
        train_targets.append(dy)
        train_sessions.append(sample["session_id"])
        if (index + 1) % 100 == 0:
            print(json.dumps({"phase": "train_features", "completed": index + 1,
                              "total": len(train_data)}), flush=True)
    train_features = np.stack(train_features)
    train_targets = np.stack(train_targets)
    session_counts = Counter(train_sessions)
    fold_sessions = balanced_session_folds(sorted(session_counts), session_counts, 5)
    train_folds = np.array([next(k for k, group in enumerate(fold_sessions) if session in group)
                            for session in train_sessions])
    model, mean, std, calibration_eps, fit_info = fit_extra_trees(
        train_features, train_targets, train_sessions, train_folds,
        args.trees, args.jobs, args.seed)

    # Fit the deployable model on all TRAIN contexts after obtaining an
    # honest session-held-out calibration margin.
    rng = np.random.default_rng(args.seed)
    sampled = []
    for index in range(len(train_features)):
        chosen = rng.choice(500, 150, replace=False)
        sampled.append((train_features[index][chosen], train_targets[index][chosen]))
    final_x = np.concatenate([x for x, _ in sampled], axis=0)
    final_y = np.concatenate([y for _, y in sampled], axis=0)
    final_mean, final_std = final_y.mean(0), final_y.std(0).clip(min=1e-6)
    final_model = ExtraTreesRegressor(n_estimators=args.trees, min_samples_leaf=12,
                                      max_features=.75, max_depth=24, n_jobs=args.jobs,
                                      random_state=args.seed)
    final_model.fit(final_x, (final_y - final_mean) / final_std)

    val_soft, val_clip_ids = load_val_sources()
    val_quality = load_val_quality()
    val_data = PairedReactionDataset(
        Path("/public/zhaowenjie/react2025_new/data"), "val", 750, crop_mode="center",
        normalization_dir=Path("/home/zhaowenjie/react2025_new/mam_reactor/external/FaceVerse"))
    records = []
    val_deltas, val_predictions = [], []
    for index in range(571):
        sample = val_data[index]
        if sample["clip_id"] != val_clip_ids[index]:
            raise ValueError(f"VAL dataset order mismatch at {index}")
        length = int(sample["source_lengths"])
        speaker = trajectory_summary(sample["speaker_emotion"][:length].numpy())
        pool, origins, distance, x, pairs = make_pairs_with_length(
            [value[index] for value in val_soft], speaker, length)
        prediction = final_model.predict(x) * final_std + final_mean
        with np.load(val_quality[index], allow_pickle=False) as quality:
            y = np.stack((quality["frc"], quality["frd"]), axis=1)
        base = list(range(10))
        baseline = {"FRC": float(y[:10, 0].sum()),
                    "exact_FRD": float(y[:10, 1].sum()),
                    "FRDiv": float(selection_value(distance, base))}
        item = {"clip_id": sample["clip_id"], "session": sample["session_id"],
                "baseline": baseline, "origins": origins, "policies": {}}
        for name, quantile in (("margin_0", 0.0), ("margin_q50", .5),
                               ("margin_q75", .75), ("margin_q90", .9),
                               ("margin_q95", .95), ("margin_q99", .99)):
            eps = np.zeros(2) if quantile == 0.0 else np.asarray(calibration_eps[str(quantile)])
            feasible = np.flatnonzero((prediction[:, 0] - eps[0] >= 0) &
                                      (prediction[:, 1] + eps[1] <= 0) &
                                      (x[:, -1] > 1e-12))
            if len(feasible):
                chosen = int(feasible[np.argmax(x[feasible, -1])])
                old, new, _ = pairs[chosen]
                selected = [slot for slot in base if slot != old] + [new]
            else:
                old = new = None
                selected = base
            actual = {"selected": selected, "old": old, "new": new,
                      "FRC": float(y[selected, 0].sum()),
                      "exact_FRD": float(y[selected, 1].sum()),
                      "FRDiv": float(selection_value(distance, selected)),
                      "strict_quality_pass": bool(y[selected, 0].sum() >= y[:10, 0].sum() - 1e-10 and
                                                   y[selected, 1].sum() <= y[:10, 1].sum() + 1e-8),
                      "eps_FRC": float(eps[0]), "eps_FRD": float(eps[1])}
            item["policies"][name] = actual
        records.append(item)
        val_deltas.append(np.asarray([[y[new, 0] - y[old, 0], y[new, 1] - y[old, 1]]
                                      for old, new, _ in pairs]))
        val_predictions.append(prediction)
        if (index + 1) % 25 == 0:
            atomic_json(args.output / "progress.json", {"completed": index + 1, "total": 571})
            print(json.dumps({"phase": "val_audit", "completed": index + 1, "total": 571}), flush=True)

    val_deltas = np.concatenate(val_deltas)
    val_predictions = np.concatenate(val_predictions)
    summary = {
        "scope": "VAL571 center750 all-session GT audit; selector trained only on TRAIN labels; not official TEST",
        "selector_inputs": "speaker emotion summary, candidate trajectory summary, source length, model/slot/scale provenance, prediction-only diversity gain",
        "gt_policy": "all same-session listener GTs for TRAIN labels and VAL audit; no GT enters selector inference",
        "candidate_pool": "E60 frozen Basic/Distance/Direction KD, raw+scale1.5; one replacement maximum",
        "training": {"contexts": len(train_data), "sessions": sorted(session_counts),
                     "folds": fold_sessions, "fit": fit_info,
                     "sampled_pairs_per_context": 150, "trees": args.trees,
                     "calibration_eps": calibration_eps,
                     "train_shard_manifests": manifests},
        "quality_prediction": {
            "delta_FRC_spearman": float(spearmanr(val_deltas[:, 0], val_predictions[:, 0]).statistic),
            "delta_FRD_spearman": float(spearmanr(val_deltas[:, 1], val_predictions[:, 1]).statistic),
            "delta_FRC_MAE": float(np.abs(val_deltas[:, 0] - val_predictions[:, 0]).mean()),
            "delta_FRD_MAE": float(np.abs(val_deltas[:, 1] - val_predictions[:, 1]).mean())},
        "policies": {name: aggregate(records, name)
                     for name in ("margin_0", "margin_q50", "margin_q75", "margin_q90", "margin_q95", "margin_q99")},
        "elapsed_seconds": time.monotonic() - started,
        "limitations": ["single held-out TRAIN session fold for calibration",
                        "one replacement maximum", "empirical margins are not formal guarantees",
                        "matched center750/all-session research protocol"]}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "records.json", {"records": records})
    torch.save({"model": final_model, "mean": final_mean, "std": final_std,
                "calibration_eps": calibration_eps, "selector_inputs": summary["selector_inputs"]},
               args.output / "selector.pt")
    lines = ["# TRAIN-supervised inference-safe selector", "", summary["scope"], "",
             "The selector never receives GT, session ID, clip ID, or GT-derived quality at selection time.", "",
             "| Policy | FRC | exact FRD | FRDiv | strict pass | changed |", "|---|---:|---:|---:|---:|---:|"]
    baseline = summary["policies"]["margin_0"]["baseline"]
    lines.append(f"| baseline | {baseline['FRC']:.6f} | {baseline['exact_FRD']:.6f} | {baseline['FRDiv']:.6f} | 571/571 | 0 |")
    for name, value in summary["policies"].items():
        selected = value["selected"]
        lines.append(f"| {name} | {selected['FRC']:.6f} | {selected['exact_FRD']:.6f} | {selected['FRDiv']:.6f} | "
                     f"{value['strict_quality_pass_contexts']}/571 | {value['changed_contexts']} |")
    lines += ["", "## Interpretation", "",
              "This is a held-out selector audit, not a GT-oracle score. Strict pass is measured with VAL GT only after selection.",
              "A positive FRDiv change with strict pass below 571 is not a zero-slack result; use the calibrated-margin rows for the quality-safe tradeoff."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
