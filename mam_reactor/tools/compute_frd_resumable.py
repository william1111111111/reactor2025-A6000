#!/usr/bin/env python3
"""Compute the existing FRD metric exactly, with resume and balanced CPU work.

The metric is unchanged from ``framework/metrics/FRD.py``.  The implementation
uses the same unconstrained DTW recurrence while retaining only two dynamic-
programming rows, and schedules one prediction at a time to avoid long-sample
tail imbalance.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from numba import njit


ALGORITHM_VERSION = 1
GROUPS = ((0, 15, 1.0 / 15.0, "au"),
          (15, 17, 1.0, "va"),
          (17, 25, 1.0 / 8.0, "emotion"))

_PREDICTION_GROUPS = None
_TARGET_GROUPS = None
_ENGINE = "rolling"
_PRUNE = True


@njit(nogil=True, cache=True)
def rolling_dtw(series_a: np.ndarray, series_b: np.ndarray) -> float:
    """Return tslearn-equivalent unconstrained DTW using O(min(T1,T2)) memory."""
    if series_b.shape[0] > series_a.shape[0]:
        series_a, series_b = series_b, series_a

    width = series_b.shape[0]
    previous = np.full(width + 1, np.inf, dtype=np.float64)
    current = np.full(width + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for row in range(series_a.shape[0]):
        current[0] = np.inf
        for column in range(width):
            local_cost = 0.0
            for feature in range(series_a.shape[1]):
                # tslearn converts inputs to float64 before its recurrence.
                difference = (
                    np.float64(series_a[row, feature])
                    - np.float64(series_b[column, feature])
                )
                local_cost += difference * difference

            best_previous = previous[column + 1]
            if current[column] < best_previous:
                best_previous = current[column]
            if previous[column] < best_previous:
                best_previous = previous[column]
            current[column + 1] = local_cost + best_previous

        previous, current = current, previous

    return math.sqrt(previous[width])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--engine", choices=("rolling", "tslearn"), default="rolling")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument(
        "--sample-indices",
        help="Comma-separated original sample indices",
    )
    parser.add_argument(
        "--stratified-count",
        type=int,
        help="Select this many length-quantile samples for a benchmark",
    )
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Evaluate all target/group pairs instead of exact nonnegative pruning",
    )
    parser.add_argument(
        "--stop-after-tasks",
        type=int,
        help="Graceful interruption hook for resume validation",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard an existing checkpoint in output-dir",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_numpy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.numpy()


def split_groups(tensor: torch.Tensor) -> tuple[np.ndarray, ...]:
    array = tensor_to_numpy(tensor)
    return tuple(
        np.ascontiguousarray(array[..., start:end], dtype=np.float32)
        for start, end, _, _ in GROUPS
    )


def select_indices(
    predictions: list[torch.Tensor], args: argparse.Namespace
) -> list[int]:
    selectors = sum(
        value is not None
        for value in (args.indices_file, args.sample_indices, args.stratified_count)
    )
    if selectors > 1:
        raise ValueError(
            "Use only one of --indices-file, --sample-indices, or --stratified-count"
        )

    if args.indices_file is not None:
        tokens = args.indices_file.read_text(encoding="utf-8").replace(",", " ").split()
        indices = [int(token) for token in tokens]
    elif args.sample_indices:
        indices = [int(token) for token in args.sample_indices.split(",") if token]
    elif args.stratified_count is not None:
        count = args.stratified_count
        if count <= 0 or count > len(predictions):
            raise ValueError(
                f"stratified-count must be in [1, {len(predictions)}], got {count}"
            )
        ranked = sorted(
            range(len(predictions)), key=lambda index: predictions[index].shape[1]
        )
        positions = np.linspace(0, len(ranked) - 1, num=count)
        indices = [ranked[int(round(position))] for position in positions]
    else:
        indices = list(range(len(predictions)))

    if not indices:
        raise ValueError("No samples selected")
    if len(set(indices)) != len(indices):
        raise ValueError("Selected sample indices contain duplicates")
    invalid = [index for index in indices if not 0 <= index < len(predictions)]
    if invalid:
        raise IndexError(f"Sample indices out of range: {invalid}")
    return indices


def source_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_metadata(
    args: argparse.Namespace,
    selected_indices: list[int],
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict:
    for index in selected_indices:
        if predictions[index].ndim != 3 or targets[index].ndim != 3:
            raise ValueError(
                f"Sample {index} must have rank-3 PRED/GT tensors, got "
                f"{predictions[index].shape} and {targets[index].shape}"
            )
        if predictions[index].shape[-1] < 25 or targets[index].shape[-1] < 25:
            raise ValueError(
                f"Sample {index} must contain at least 25 features"
            )
    prediction_counts = [int(predictions[index].shape[0]) for index in selected_indices]
    target_counts = [int(targets[index].shape[0]) for index in selected_indices]
    sequence_lengths = [int(predictions[index].shape[1]) for index in selected_indices]
    return {
        "algorithm": "react2025_exact_frd",
        "algorithm_version": ALGORITHM_VERSION,
        "engine": args.engine,
        "pruning": not args.disable_pruning,
        "source": source_signature(args.results),
        "selected_indices": selected_indices,
        "prediction_counts": prediction_counts,
        "target_counts": target_counts,
        "sequence_lengths": sequence_lengths,
        "groups": [
            {"start": start, "end": end, "weight": weight, "name": name}
            for start, end, weight, name in GROUPS
        ],
    }


def metadata_identity(metadata: dict) -> dict:
    return {
        key: metadata[key]
        for key in (
            "algorithm",
            "algorithm_version",
            "engine",
            "pruning",
            "source",
            "selected_indices",
            "prediction_counts",
            "target_counts",
            "sequence_lengths",
            "groups",
        )
    }


def prepare_arrays(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    selected_indices: list[int],
) -> tuple[list[tuple[np.ndarray, ...]], list[tuple[np.ndarray, ...]]]:
    prediction_groups = []
    target_groups = []
    for index in selected_indices:
        prediction_groups.append(split_groups(predictions[index]))
        target_groups.append(split_groups(targets[index]))
    return prediction_groups, target_groups


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    if _ENGINE == "rolling":
        return float(rolling_dtw(first, second))
    from tslearn.metrics import dtw as tslearn_dtw

    return float(tslearn_dtw(first, second))


def compute_prediction_task(task: tuple[int, int, int]) -> tuple[int, int, float, int]:
    row, prediction_index, work_units = task
    prediction_groups = _PREDICTION_GROUPS[row]
    target_groups = _TARGET_GROUPS[row]

    best = math.inf
    target_count = target_groups[0].shape[0]
    for target_index in range(target_count):
        total = 0.0
        for group_index, (_, _, weight, _) in enumerate(GROUPS):
            distance = dtw_distance(
                prediction_groups[group_index][prediction_index],
                target_groups[group_index][target_index],
            )
            total += weight * distance
            # Every remaining contribution is nonnegative, so this cannot
            # change the minimum or the final metric.
            if _PRUNE and total >= best:
                break
        if total < best:
            best = total

    return row, prediction_index, best, work_units


def calculate_task_work(
    row: int,
    prediction_index: int,
    prediction_groups: list[tuple[np.ndarray, ...]],
    target_groups: list[tuple[np.ndarray, ...]],
) -> int:
    prediction_length = prediction_groups[row][0][prediction_index].shape[0]
    target_lengths = [target.shape[0] for target in target_groups[row][0]]
    return int(prediction_length * sum(target_lengths) * len(GROUPS))


def build_tasks(
    values: np.ndarray,
    prediction_counts: list[int],
    prediction_groups: list[tuple[np.ndarray, ...]],
    target_groups: list[tuple[np.ndarray, ...]],
) -> list[tuple[int, int, int]]:
    tasks = []
    for row, prediction_count in enumerate(prediction_counts):
        for prediction_index in range(prediction_count):
            if np.isnan(values[row, prediction_index]):
                work_units = calculate_task_work(
                    row,
                    prediction_index,
                    prediction_groups,
                    target_groups,
                )
                tasks.append((row, prediction_index, work_units))
    # Start the most expensive predictions first to minimize the final tail.
    tasks.sort(key=lambda task: task[2], reverse=True)
    return tasks


def aggregate_complete(values: np.ndarray, prediction_counts: list[int]) -> tuple[float | None, int]:
    sample_sums = []
    for row, prediction_count in enumerate(prediction_counts):
        row_values = values[row, :prediction_count]
        if not np.isnan(row_values).any():
            # Match the legacy Python loop's left-to-right accumulation.
            sample_sums.append(sum(float(value) for value in row_values))
    if not sample_sums:
        return None, 0
    return float(np.mean(sample_sums, dtype=np.float64)), len(sample_sums)


def count_completed(values: np.ndarray, prediction_counts: list[int]) -> int:
    return sum(
        int(np.count_nonzero(~np.isnan(values[row, :prediction_count])))
        for row, prediction_count in enumerate(prediction_counts)
    )


def initialize_worker(lock_fd: int) -> None:
    # Pool workers inherit file descriptors under fork. They must not keep the
    # single-run lock alive if the parent exits unexpectedly.
    try:
        os.close(lock_fd)
    except OSError:
        pass
    # The parent installs graceful handlers. Pool.terminate() relies on the
    # default SIGTERM behavior in workers, otherwise an interrupted run hangs.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def save_state(
    output_dir: Path,
    metadata: dict,
    values: np.ndarray,
    prediction_counts: list[int],
    status: str,
    workers: int,
    started_at: str,
    run_start: float,
    completed_work: int,
    total_work: int,
    message: str | None = None,
) -> None:
    partial_path = output_dir / "frd_partial.npy"
    progress_path = output_dir / "frd_progress.json"
    atomic_numpy(partial_path, values)
    partial_frd, complete_samples = aggregate_complete(values, prediction_counts)
    completed_tasks = count_completed(values, prediction_counts)
    total_tasks = sum(prediction_counts)
    payload = {
        "status": status,
        "pid": os.getpid(),
        "workers": workers,
        "started_at": started_at,
        "updated_at": utc_now(),
        "elapsed_s_this_run": time.perf_counter() - run_start,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "completed_samples": complete_samples,
        "total_samples": len(prediction_counts),
        "completed_work": completed_work,
        "total_work": total_work,
        "work_fraction": completed_work / total_work if total_work else 1.0,
        "partial_frd_complete_samples": partial_frd,
        "metadata_file": "frd_metadata.json",
        "checkpoint_file": "frd_partial.npy",
    }
    if message:
        payload["message"] = message
    atomic_json(progress_path, payload)


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.checkpoint_every <= 0 or args.progress_every <= 0:
        raise ValueError("checkpoint/progress intervals must be positive")
    if args.stop_after_tasks is not None and args.stop_after_tasks <= 0:
        raise ValueError("stop-after-tasks must be positive")

    args.results = args.results.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lock_handle = (args.output_dir / "frd.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"Another FRD runner holds {args.output_dir / 'frd.lock'}"
        ) from error

    print(f"[{utc_now()}] loading {args.results}", flush=True)
    load_start = time.perf_counter()
    saved = torch.load(args.results, map_location="cpu", weights_only=False)
    predictions = saved["PRED"]
    targets = saved["GT"]
    if len(predictions) != len(targets):
        raise ValueError(f"PRED/GT sample count differs: {len(predictions)} != {len(targets)}")
    selected_indices = select_indices(predictions, args)
    load_elapsed = time.perf_counter() - load_start
    metadata = build_metadata(args, selected_indices, predictions, targets)
    metadata_path = args.output_dir / "frd_metadata.json"
    partial_path = args.output_dir / "frd_partial.npy"

    if args.reset:
        for path in (
            metadata_path,
            partial_path,
            args.output_dir / "frd_progress.json",
            args.output_dir / "frd_result.json",
        ):
            path.unlink(missing_ok=True)

    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_identity(existing_metadata) != metadata_identity(metadata):
            raise RuntimeError(
                "Existing checkpoint metadata does not match this run. "
                "Use a new output directory or pass --reset intentionally."
            )
    else:
        atomic_json(metadata_path, metadata)

    prediction_counts = metadata["prediction_counts"]
    max_predictions = max(prediction_counts)
    if partial_path.exists():
        values = np.load(partial_path)
        expected_shape = (len(selected_indices), max_predictions)
        if values.shape != expected_shape:
            raise ValueError(
                f"Checkpoint shape {values.shape} != expected {expected_shape}"
            )
    else:
        values = np.full(
            (len(selected_indices), max_predictions), np.nan, dtype=np.float64
        )

    print(
        f"[{utc_now()}] loaded {len(predictions)} samples; selected "
        f"{len(selected_indices)} in {load_elapsed:.2f}s; "
        f"GT count range={min(metadata['target_counts'])}..{max(metadata['target_counts'])}",
        flush=True,
    )
    if max(metadata["target_counts"]) != 10 or min(metadata["target_counts"]) != 10:
        print(
            "WARNING: preserving the exact GT count stored in results.pt; "
            "this run is not forcing 10 GTs.",
            flush=True,
        )

    preprocess_start = time.perf_counter()
    prediction_groups, target_groups = prepare_arrays(
        predictions, targets, selected_indices
    )
    preprocess_elapsed = time.perf_counter() - preprocess_start
    del saved, predictions, targets
    print(
        f"[{utc_now()}] contiguous feature groups prepared in "
        f"{preprocess_elapsed:.2f}s",
        flush=True,
    )

    global _PREDICTION_GROUPS, _TARGET_GROUPS, _ENGINE, _PRUNE
    _PREDICTION_GROUPS = prediction_groups
    _TARGET_GROUPS = target_groups
    _ENGINE = args.engine
    _PRUNE = not args.disable_pruning

    # Compile once before fork so workers inherit the machine code.
    if args.engine == "rolling":
        rolling_dtw(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )

    all_task_work = {}
    for row, prediction_count in enumerate(prediction_counts):
        for prediction_index in range(prediction_count):
            all_task_work[(row, prediction_index)] = calculate_task_work(
                row, prediction_index, prediction_groups, target_groups
            )
    total_work = sum(all_task_work.values())
    completed_work = sum(
        work
        for (row, prediction_index), work in all_task_work.items()
        if not np.isnan(values[row, prediction_index])
    )
    tasks = build_tasks(
        values, prediction_counts, prediction_groups, target_groups
    )
    initial_completed = count_completed(values, prediction_counts)
    total_tasks = sum(prediction_counts)
    print(
        f"[{utc_now()}] engine={args.engine} workers={args.workers} "
        f"remaining_tasks={len(tasks)}/{total_tasks} "
        f"resume_completed={initial_completed}",
        flush=True,
    )

    started_at = utc_now()
    run_start = time.perf_counter()
    save_state(
        args.output_dir,
        metadata,
        values,
        prediction_counts,
        "running",
        args.workers,
        started_at,
        run_start,
        completed_work,
        total_work,
    )

    if not tasks:
        final_frd, complete_samples = aggregate_complete(values, prediction_counts)
        result = {
            "status": "complete",
            "FRD": final_frd,
            "samples": complete_samples,
            "tasks": total_tasks,
            "engine": args.engine,
            "pruning": not args.disable_pruning,
            "workers": args.workers,
            "compute_elapsed_s": 0.0,
            "wall_elapsed_s_this_run": time.perf_counter() - run_start,
            "load_elapsed_s": load_elapsed,
            "preprocess_elapsed_s": preprocess_elapsed,
            "total_elapsed_s": time.perf_counter() - process_start,
            "completed_at": utc_now(),
            "source": metadata["source"],
            "selected_indices": selected_indices,
            "target_count_min": min(metadata["target_counts"]),
            "target_count_max": max(metadata["target_counts"]),
        }
        atomic_json(args.output_dir / "frd_result.json", result)
        save_state(
            args.output_dir,
            metadata,
            values,
            prediction_counts,
            "complete",
            args.workers,
            started_at,
            run_start,
            total_work,
            total_work,
        )
        print(json.dumps(result), flush=True)
        return 0

    stop_requested = False

    def request_stop(signum, frame):
        del frame
        nonlocal stop_requested
        stop_requested = True
        print(f"[{utc_now()}] received signal {signum}; checkpointing", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    completed_this_run = 0
    compute_start = time.perf_counter()
    context = mp.get_context("fork")
    pool = context.Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(lock_handle.fileno(),),
    )
    status = "running"
    message = None
    try:
        for row, prediction_index, value, work_units in pool.imap_unordered(
            compute_prediction_task, tasks, chunksize=1
        ):
            values[row, prediction_index] = value
            completed_work += work_units
            completed_this_run += 1

            if (
                completed_this_run % args.progress_every == 0
                or completed_this_run == len(tasks)
            ):
                elapsed = time.perf_counter() - compute_start
                print(
                    f"[{utc_now()}] completed "
                    f"{initial_completed + completed_this_run}/{total_tasks} tasks; "
                    f"work={completed_work / total_work:.2%}; "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if completed_this_run % args.checkpoint_every == 0:
                save_state(
                    args.output_dir,
                    metadata,
                    values,
                    prediction_counts,
                    "running",
                    args.workers,
                    started_at,
                    run_start,
                    completed_work,
                    total_work,
                )

            if stop_requested:
                status = "interrupted"
                message = "Stopped by signal after a resumable checkpoint"
                break
            if (
                args.stop_after_tasks is not None
                and completed_this_run >= args.stop_after_tasks
            ):
                status = "interrupted"
                message = "Planned stop-after-tasks reached"
                break

        if status == "running":
            pool.close()
            status = "complete"
        else:
            pool.terminate()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        status = "failed"
        message = "Runner failed; completed values were checkpointed"
        save_state(
            args.output_dir,
            metadata,
            values,
            prediction_counts,
            status,
            args.workers,
            started_at,
            run_start,
            completed_work,
            total_work,
            message,
        )
        raise

    compute_elapsed = time.perf_counter() - compute_start
    save_state(
        args.output_dir,
        metadata,
        values,
        prediction_counts,
        status,
        args.workers,
        started_at,
        run_start,
        completed_work,
        total_work,
        message,
    )

    if status == "complete":
        final_frd, complete_samples = aggregate_complete(values, prediction_counts)
        if complete_samples != len(prediction_counts):
            raise RuntimeError(
                f"Only {complete_samples}/{len(prediction_counts)} samples complete"
            )
        result = {
            "status": "complete",
            "FRD": final_frd,
            "samples": complete_samples,
            "tasks": total_tasks,
            "engine": args.engine,
            "pruning": not args.disable_pruning,
            "workers": args.workers,
            "compute_elapsed_s": compute_elapsed,
            "wall_elapsed_s_this_run": time.perf_counter() - run_start,
            "load_elapsed_s": load_elapsed,
            "preprocess_elapsed_s": preprocess_elapsed,
            "total_elapsed_s": time.perf_counter() - process_start,
            "completed_at": utc_now(),
            "source": metadata["source"],
            "selected_indices": selected_indices,
            "target_count_min": min(metadata["target_counts"]),
            "target_count_max": max(metadata["target_counts"]),
        }
        atomic_json(args.output_dir / "frd_result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return 0

    print(
        f"[{utc_now()}] {status}: {message}; rerun the same command to resume",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
