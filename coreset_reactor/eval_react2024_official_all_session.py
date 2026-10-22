"""Evaluate REACT2024 with unmodified official metrics and all-session GT.

The only protocol extension is the target pool: every prediction is matched
against every processed opposite-role recording from the same session. FRC,
FRD, and TLCC are computed by the official Mam-Reactor ``_func`` bodies. The
public wrappers only parallel-map these functions and take their mean; this
runner adds resumability and records the same per-context outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/react2024_official_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"framework\.metrics\.FRC"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MAM_ROOT = PROJECT_ROOT / "mam_reactor"
sys.path.insert(0, str(OFFICIAL_MAM_ROOT))

from framework.metrics.FRC import _func as official_frc_context  # noqa: E402
from framework.metrics.FRD import _func as official_frd_context  # noqa: E402
from framework.metrics.FRVar import compute_FRVar  # noqa: E402
from framework.metrics.S_MSE import compute_s_mse  # noqa: E402
from framework.metrics.TLCC import _func as official_tlcc_context  # noqa: E402
from tslearn.metrics import dtw  # noqa: E402

from coreset_reactor.eval_overlap_save_student import (  # noqa: E402
    powered_center_overlap,
)
from coreset_reactor.eval_student_window_gap import window_starts  # noqa: E402
from coreset_reactor.evaluate import official_prediction  # noqa: E402
from coreset_reactor.query0_au_transition import (  # noqa: E402
    stabilize_query0_au_transitions,
)
from coreset_reactor.student_spread_probe import expand_predictions  # noqa: E402


_TARGETS: list[torch.Tensor] = []
_SOURCES: list[torch.Tensor] = []
_PREDICTIONS: dict[str, list[torch.Tensor]] = {}


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


def evaluate_task(task: tuple[int, str]) -> dict:
    index, arm = task
    target = _TARGETS[index]
    source = _SOURCES[index]
    prediction = _PREDICTIONS[arm][index]
    previous = np.seterr(divide="ignore", invalid="ignore")
    try:
        tlcc = float(official_tlcc_context(prediction, source))
    finally:
        np.seterr(**previous)
    if not np.isfinite(tlcc):
        raise ValueError("Official TLCC returned a non-finite value")
    return {
        "index": index,
        "arm": arm,
        "FRC": float(official_frc_context(target, prediction)),
        "FRD": float(official_frd_context(target, prediction)),
        "TLCC": tlcc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("powered", "spread"), default="powered",
        help="Comparison transform applied to the cached soft predictions",
    )
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--spread-alpha", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-contexts", type=int, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.power) or args.power <= 0:
        raise ValueError("power must be finite and positive")
    if not np.isfinite(args.spread_alpha) or args.spread_alpha < 1:
        raise ValueError("spread alpha must be finite and at least one")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    reference_json = args.reference_dir / "val1124_full.json"
    prediction_path = args.reference_dir / "val1124_full.predictions.pt"
    reference = json.loads(reference_json.read_text())
    payload = torch.load(
        str(prediction_path), map_location="cpu", weights_only=False, mmap=True
    )
    total = len(reference["rows"])
    if args.max_contexts > 0:
        total = min(total, args.max_contexts)
    if payload["clip_ids"][:total] != [
        row["clip_id"] for row in reference["rows"][:total]
    ]:
        raise ValueError("Prediction and reference orders differ")

    raw_baseline = payload["official_predictions"][:total]
    raw_transformed = []
    targets = []
    sources = []
    gt_counts = []
    for index in range(total):
        soft = payload["soft_predictions"][index]
        if args.arm == "powered":
            starts = window_starts(int(soft.shape[1]))
            changed = powered_center_overlap(
                soft.unsqueeze(0), starts, int(soft.shape[1]), args.power
            )
            arm_name = f"powered_p{args.power:g}"
        else:
            changed = expand_predictions(soft, args.spread_alpha)
            arm_name = f"spread_alpha{args.spread_alpha:g}"
        raw_transformed.append(official_prediction(changed))
        row = reference["rows"][index]
        source_path = (
            args.data_root / "val" / "facial-attributes" / f"{row['clip_id']}.npy"
        )
        source = torch.from_numpy(
            np.load(source_path, allow_pickle=False).astype(np.float32)
        )
        if source.shape != raw_transformed[-1].shape[1:]:
            raise ValueError("Source face and prediction shapes differ")
        sources.append(source)
        cached = torch.load(
            args.target_cache / f"{row['target_cache_key']}.pt",
            map_location="cpu", weights_only=False,
        )
        if cached["fingerprint"] != row["target_cache_key"]:
            raise ValueError("GT cache fingerprint mismatch")
        target = cached["targets"]
        if target.shape[0] != row["gt_count"]:
            raise ValueError("GT count differs from frozen all-session record")
        targets.append(target)
        gt_counts.append(int(target.shape[0]))

    baseline = [
        stabilize_query0_au_transitions(prediction, source)
        for prediction, source in zip(raw_baseline, sources)
    ]
    transformed = [
        stabilize_query0_au_transitions(prediction, source)
        for prediction, source in zip(raw_transformed, sources)
    ]
    changed_contexts = {
        "baseline": sum(
            not torch.equal(before, after)
            for before, after in zip(raw_baseline, baseline)
        ),
        arm_name: sum(
            not torch.equal(before, after)
            for before, after in zip(raw_transformed, transformed)
        ),
    }

    global _TARGETS, _SOURCES, _PREDICTIONS
    _TARGETS = targets
    _SOURCES = sources
    _PREDICTIONS = {"baseline": baseline, arm_name: transformed}
    arms = tuple(_PREDICTIONS)

    # Compile tslearn's DTW kernel before forking workers.
    dtw(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    rows_path = args.output_dir / "official_rows.jsonl"
    completed = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            item = json.loads(line)
            if {"FRC", "FRD", "TLCC"}.issubset(item):
                completed[(int(item["index"]), item["arm"])] = item
    tasks = [
        (index, arm)
        for index in range(total)
        for arm in arms
        if (index, arm) not in completed
    ]
    started = time.time()
    context = mp.get_context("fork")
    if tasks:
        with context.Pool(processes=args.workers) as pool, rows_path.open("a") as stream:
            for item in pool.imap_unordered(evaluate_task, tasks, chunksize=1):
                key = (int(item["index"]), item["arm"])
                completed[key] = item
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                if len(completed) % 20 == 0 or len(completed) == total * 2:
                    print(
                        f"[react2024-official-all-session] "
                        f"{len(completed)}/{total * 2}", flush=True,
                    )
                    write_json(args.output_dir / "progress.json", {
                        "completed": len(completed),
                        "total": total * 2,
                        "elapsed_seconds_this_run": time.time() - started,
                    })

    ordered = {
        arm: [completed[(index, arm)] for index in range(total)]
        for arm in arms
    }
    metrics = {}
    for arm, predictions in _PREDICTIONS.items():
        metrics[arm] = {
            "contexts": total,
            "FRC": float(np.mean([row["FRC"] for row in ordered[arm]])),
            "exact_FRD": float(np.mean([row["FRD"] for row in ordered[arm]])),
            "FRDiv": float(compute_s_mse(predictions)),
            "FRVar": float(compute_FRVar(predictions)),
            "TLCC": float(np.mean([row["TLCC"] for row in ordered[arm]])),
        }
    delta = {
        key: {
            "absolute": metrics[arm_name][key] - metrics["baseline"][key],
            "relative_percent": 100 * (
                metrics[arm_name][key] - metrics["baseline"][key]
            ) / metrics["baseline"][key],
        }
        for key in ("FRC", "exact_FRD", "FRDiv", "FRVar", "TLCC")
    }
    official_files = [
        OFFICIAL_MAM_ROOT / "framework/metrics/FRC.py",
        OFFICIAL_MAM_ROOT / "framework/metrics/FRD.py",
        OFFICIAL_MAM_ROOT / "framework/metrics/S_MSE.py",
        OFFICIAL_MAM_ROOT / "framework/metrics/FRVar.py",
        OFFICIAL_MAM_ROOT / "framework/metrics/TLCC.py",
        OFFICIAL_MAM_ROOT / "framework/modules/post_processor.py",
    ]
    result = {
        "protocol": (
            "Official Mam-Reactor metric functions; full REACT2024 VAL1124; "
            "every processed opposite-role GT in the same session; AU-rounded "
            "predictions; query-0 sparse constant-AU transition postprocess; "
            + (
                f"powered single-window fusion p={args.power:g}"
                if args.arm == "powered"
                else f"slot-centered spread scaling alpha={args.spread_alpha:g}"
            )
        ),
        "official_metric_contract": {
            "FRC": "mean of framework.metrics.FRC._func over contexts",
            "exact_FRD": "mean of framework.metrics.FRD._func over contexts",
            "FRDiv": "framework.metrics.S_MSE.compute_s_mse",
            "FRVar": "framework.metrics.FRVar.compute_FRVar",
            "TLCC": "framework.metrics.TLCC._func per context",
            "orchestration_note": (
                "The official public FRC/FRD wrappers perform the same starmap "
                "and mean; this runner records rows for resumability."
            ),
        },
        "target_pool": {
            "rule": "all same-session opposite-role GT, paired target first",
            "minimum_gt_count": min(gt_counts),
            "maximum_gt_count": max(gt_counts),
            "mean_gt_count": float(np.mean(gt_counts)),
        },
        "postprocess": {
            "rule": (
                "one source-aligned transition frame in every constant "
                "query-0 AU"
            ),
            "source_constant_policy": "leave prediction unchanged",
            "margin_frames": 49,
            "changed_contexts": changed_contexts,
        },
        "metrics": metrics,
        f"delta_{arm_name}_minus_baseline": delta,
        "provenance": {
            "reference_json": [str(reference_json.resolve()), sha256(reference_json)],
            "prediction_cache": [str(prediction_path.resolve()), sha256(prediction_path)],
            "target_cache": str(args.target_cache.resolve()),
            "official_sources": [
                [str(path.resolve()), sha256(path)] for path in official_files
            ],
        },
        "limitations": [
            "The official metric code is unchanged; the GT pool is intentionally extended from sampled/neighbour targets to all same-session targets.",
            "Validation only; this is not a hidden TEST claim.",
        ],
    }
    write_json(args.output_dir / "results.json", result)
    print(json.dumps({
        "target_pool": result["target_pool"],
        "metrics": metrics,
        "delta": delta,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
