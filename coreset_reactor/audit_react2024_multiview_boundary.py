"""Audit the REACT2025 multi-view/boundary route on fixed-length REACT2024.

The transferred route has two operations:

1. use one crop for sources of at most 750 frames and three stratified crops
   for longer sources;
2. fuse 750-frame inference windows with powered triangular weights (p=4).

REACT2024 was converted into 750-frame clips.  This program verifies that
fact on the actual train/validation inputs and evaluates the real p=4 code on
the frozen full-VAL1124 prediction cache.  It optionally recomputes exact
rolling-DTW FRD against the frozen all-session GT cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/react2024_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from framework.metrics import compute_FRVar, compute_s_mse  # noqa: E402
from tools.compute_frd_resumable import rolling_dtw  # noqa: E402

from coreset_reactor.eval_overlap_save_student import (  # noqa: E402
    powered_center_overlap,
)
from coreset_reactor.eval_student_window_gap import window_starts  # noqa: E402
from coreset_reactor.evaluate import official_prediction  # noqa: E402
from coreset_reactor.fast_frc import candidate_frc  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_length(data_root: Path, split: str, record: Path) -> int:
    relative = record.relative_to(data_root / split / "facial-attributes")
    paths = (
        record,
        data_root / split / "audio-features" / relative,
        data_root / split / "coefficients" / relative,
    )
    return min(int(np.load(path, mmap_mode="r").shape[0]) for path in paths)


def length_audit(data_root: Path, train_manifest: dict) -> dict:
    allowlist = set(train_manifest.get("session_allowlist", []))
    result = {}
    for split, role in (("train", None), ("val", "speaker")):
        root = data_root / split / "facial-attributes"
        records = sorted(root.glob("*/*/*.npy"))
        if role is not None:
            records = [path for path in records if path.relative_to(root).parts[0] == role]
        if split == "train" and allowlist:
            records = [
                path for path in records
                if path.relative_to(root).parts[1] in allowlist
            ]
        lengths = [source_length(data_root, split, path) for path in records]
        if not lengths:
            raise ValueError(f"No {split} records found")
        result[split] = {
            "contexts": len(lengths),
            "minimum": min(lengths),
            "maximum": max(lengths),
            "equal_750": sum(length == 750 for length in lengths),
            "longer_than_750": sum(length > 750 for length in lengths),
            "effective_multiview_contexts": sum(
                3 if length > 750 else 1 for length in lengths
            ),
        }
    return result


def exact_frd(prediction: np.ndarray, target: np.ndarray,
              pool: ThreadPoolExecutor) -> float:
    groups = ((0, 15, 1 / 15), (15, 17, 1.0), (17, 25, 1 / 8))
    targets = [np.ascontiguousarray(target[..., start:stop])
               for start, stop, _ in groups]

    def candidate_score(candidate: np.ndarray) -> float:
        candidates = [np.ascontiguousarray(candidate[:, start:stop])
                      for start, stop, _ in groups]
        best = math.inf
        for target_index in range(target.shape[0]):
            total = 0.0
            for group_index, (_, _, weight) in enumerate(groups):
                total += weight * rolling_dtw(
                    candidates[group_index], targets[group_index][target_index]
                )
                if total >= best:
                    break
            best = min(best, total)
        return best

    return float(sum(pool.map(candidate_score, prediction)))


def metric_delta(current: dict, reference: dict) -> dict:
    return {
        key: {
            "absolute": float(current[key] - reference[key]),
            "relative_percent": float(
                100 * (current[key] - reference[key]) / reference[key]
            ),
        }
        for key in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-exact-frd", action="store_true")
    args = parser.parse_args()
    if args.power <= 0 or not np.isfinite(args.power):
        raise ValueError("power must be finite and positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    reference_json = args.reference_dir / "val1124_full.json"
    prediction_path = args.reference_dir / "val1124_full.predictions.pt"
    train_manifest_path = args.reference_dir / "train_cache/manifest.json"
    reference = json.loads(reference_json.read_text())
    train_manifest = json.loads(train_manifest_path.read_text())
    lengths = length_audit(args.data_root, train_manifest)
    if lengths["train"]["contexts"] != train_manifest["contexts"]:
        raise ValueError("Training length audit does not match frozen cache")
    if lengths["val"]["contexts"] != reference["metrics"]["contexts"]:
        raise ValueError("Validation length audit does not match frozen evaluation")

    payload = torch.load(
        str(prediction_path), map_location="cpu", weights_only=False, mmap=True
    )
    if payload["clip_ids"] != [row["clip_id"] for row in reference["rows"]]:
        raise ValueError("Prediction and metric cache orders differ")

    started = time.time()
    soft_predictions = []
    predictions = []
    soft_exact_contexts = official_exact_contexts = 0
    soft_different_elements = official_different_elements = 0
    max_abs_soft = max_abs_official = 0.0
    for soft, archived_official in zip(
        payload["soft_predictions"], payload["official_predictions"]
    ):
        length = int(soft.shape[1])
        starts = window_starts(length)
        if starts != [0]:
            raise ValueError(f"Expected a single REACT2024 window, got {starts}")
        fused = powered_center_overlap(
            soft.unsqueeze(0), starts, length, args.power
        )
        official = official_prediction(fused)
        soft_difference = (fused - soft).abs()
        official_difference = (official - archived_official).abs()
        soft_exact_contexts += int(torch.equal(fused, soft))
        official_exact_contexts += int(torch.equal(official, archived_official))
        soft_different_elements += int(torch.count_nonzero(soft_difference))
        official_different_elements += int(torch.count_nonzero(official_difference))
        max_abs_soft = max(max_abs_soft, float(soft_difference.max()))
        max_abs_official = max(
            max_abs_official, float(official_difference.max())
        )
        soft_predictions.append(fused)
        predictions.append(official)

    frc_rows = []
    targets = []
    for number, (row, prediction) in enumerate(
        zip(reference["rows"], predictions), 1
    ):
        cache_path = args.target_cache / f"{row['target_cache_key']}.pt"
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["fingerprint"] != row["target_cache_key"]:
            raise ValueError("GT cache fingerprint mismatch")
        target = cached["targets"]
        targets.append(target)
        frc_rows.append(candidate_frc(target.numpy(), prediction.numpy()))
        if number % 100 == 0 or number == len(predictions):
            print(f"[react2024-p4-fast] {number}/{len(predictions)}", flush=True)

    frc_per_expert = np.mean(frc_rows, axis=0)
    metrics = {
        "contexts": len(predictions),
        "FRC": float(frc_per_expert.sum()),
        "FRC_per_expert": frc_per_expert.tolist(),
        "FRDiv": float(compute_s_mse(predictions)),
        "FRVar": float(compute_FRVar(predictions)),
        "soft_FRDiv": float(compute_s_mse(soft_predictions)),
        "exact_FRD": None,
    }

    exact_rows_path = args.output_dir / "exact_frd_rows.jsonl"
    exact_rows = []
    if exact_rows_path.exists():
        exact_rows = [json.loads(line) for line in exact_rows_path.read_text().splitlines()]
    if not args.skip_exact_frd:
        # Trigger Numba compilation before timing the resumable loop.
        rolling_dtw(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool, exact_rows_path.open("a") as stream:
            for index in range(len(exact_rows), len(predictions)):
                value = exact_frd(
                    predictions[index].numpy(), targets[index].numpy(), pool
                )
                item = {
                    "index": index,
                    "clip_id": reference["rows"][index]["clip_id"],
                    "powered_exact_FRD": value,
                    "reference_exact_FRD": reference["rows"][index]["exact_FRD"],
                }
                exact_rows.append(item)
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                if len(exact_rows) % 10 == 0 or len(exact_rows) == len(predictions):
                    write_json(args.output_dir / "progress.json", {
                        "completed": len(exact_rows),
                        "total": len(predictions),
                        "elapsed_seconds": time.time() - started,
                        "partial_exact_FRD": float(np.mean([
                            item["powered_exact_FRD"] for item in exact_rows
                        ])),
                    })
                    print(
                        f"[react2024-p4-exact] {len(exact_rows)}/{len(predictions)}",
                        flush=True,
                    )
        metrics["exact_FRD"] = float(np.mean([
            item["powered_exact_FRD"] for item in exact_rows
        ]))

    exact_row_delta = None
    if exact_rows:
        values = np.asarray([
            item["powered_exact_FRD"] - item["reference_exact_FRD"]
            for item in exact_rows
        ])
        exact_row_delta = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "maximum_absolute": float(np.abs(values).max()),
            "within_1e-7": int(np.sum(np.abs(values) <= 1e-7)),
            "contexts": int(values.size),
        }
    numerical = {
        "single_window_starts": [0],
        "soft_exact_contexts": soft_exact_contexts,
        "official_exact_contexts": official_exact_contexts,
        "soft_different_elements": soft_different_elements,
        "official_different_elements": official_different_elements,
        "max_abs_soft": max_abs_soft,
        "max_abs_official": max_abs_official,
        "exact_frd_row_delta": exact_row_delta,
        "interpretation": (
            "A lone weighted window is mathematically the identity. Nonzero "
            "differences are float32 multiply/divide roundoff."
        ),
    }
    result = {
        "protocol": (
            "REACT2024 full VAL1124, frozen count4 student predictions, "
            f"single-window powered triangular fusion p={args.power:g}, "
            "all-session opposite-role GT, AU rounded"
        ),
        "route": {
            "training": "three views only when source length >750",
            "inference": f"750-frame overlap-add with center weight power {args.power:g}",
        },
        "length_audit": lengths,
        "metrics": metrics,
        "reference_metrics": reference["metrics"],
        "delta_vs_reference": (
            metric_delta(metrics, reference["metrics"])
            if metrics["exact_FRD"] is not None else None
        ),
        "numerical_identity_audit": numerical,
        "artifacts": {
            "reference_json": [str(reference_json.resolve()), sha256(reference_json)],
            "prediction_cache": [str(prediction_path.resolve()), sha256(prediction_path)],
            "train_manifest": [str(train_manifest_path.resolve()), sha256(train_manifest_path)],
        },
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Validation only; this is not a hidden TEST result.",
            "No retraining is needed to test the multi-view mechanism because every eligible training source has one effective view at start 0.",
        ],
    }
    write_json(args.output_dir / "results.json", result)
    print(json.dumps({
        "length_audit": lengths,
        "metrics": metrics,
        "delta_vs_reference": result["delta_vs_reference"],
        "numerical_identity_audit": numerical,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
