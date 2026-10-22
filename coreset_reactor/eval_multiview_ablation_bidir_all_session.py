"""Evaluate every active REACT2025 method ablation under one fair protocol.

The main ablation population is the complete 1,142-context validation set:
571 speaker-to-listener contexts and 571 listener-to-speaker contexts.  Every
context uses the full source duration and every opposite-role target from the
same session.  The five repository metrics are FRC, exact FRD, FRDiv, FRVar,
and the official TLCC/FRSyn implementation.

Single-direction results are rejected.  Only the frozen old-student baseline
is imported from a complete bidirectional evaluation; every multi-view arm is
computed over the complete population by this evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from coreset_reactor.eval_full_sequence_all_session import full_student
from coreset_reactor.eval_multiview_boundary_student import load_model
from coreset_reactor.eval_overlap_save_student import (
    overlap_save,
    powered_center_overlap,
)
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    boundary_jump,
    generate_windows,
    score,
    window_starts,
)
from coreset_reactor.evaluate import official_prediction


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "coreset_reactor/reports"
BASE = REPORTS / "full_sequence_bidir_allgt_20261018"
POWERS = (1.0, 2.0, 4.0, 8.0)
METHODS = (
    "single_view_nonoverlap",
    "multiview_nonoverlap",
    "multiview_power_1",
    "multiview_power_2",
    "multiview_power_4",
    "multiview_power_8",
)
METRICS = ("FRC", "exact_FRD", "FRDiv", "FRVar", "TLCC")
OFFICIAL_SCALARS = ("FRC", "exact_FRD", "FRDiv", "FRVar")


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


def frozen(path: Path) -> list[str]:
    return [str(path.resolve()), sha256(path)]


def validate_bidirectional_population(entries: list[dict]) -> None:
    """Reject a forward-only population before official metrics are reported."""
    if len(entries) != 1142:
        raise ValueError(
            f"Official ablations require VAL1142, found {len(entries)} contexts"
        )
    counts = {
        role: sum(entry.get("role") == role for entry in entries)
        for role in ("speaker", "listener")
    }
    if counts != {"speaker": 571, "listener": 571}:
        raise ValueError(f"Official ablations require 571 contexts per direction: {counts}")
    if any(int(entry.get("length", 0)) <= 0 for entry in entries):
        raise ValueError("Every official ablation context must retain its full duration")


def official_tlcc(prediction: torch.Tensor, source_face: torch.Tensor) -> float:
    """Call the repository's official per-context TLCC implementation exactly."""
    from framework.metrics.TLCC import _func

    if prediction.ndim != 3 or source_face.ndim != 2:
        raise ValueError("TLCC expects prediction [K,T,25] and source [T,25]")
    if prediction.shape[1:] != source_face.shape:
        raise ValueError(
            f"TLCC shape mismatch: {tuple(prediction.shape)} / {tuple(source_face.shape)}"
        )
    previous = np.seterr(divide="ignore", invalid="ignore")
    try:
        value = float(_func(prediction, source_face))
    finally:
        np.seterr(**previous)
    if not np.isfinite(value):
        raise ValueError("Official TLCC returned a non-finite value")
    return value


def scalar_metrics(value: dict) -> dict:
    return {metric: float(value[metric]) for metric in OFFICIAL_SCALARS}


def prepare(output: Path, checkpoint: Path) -> None:
    if (output / "manifest.json").exists():
        raise FileExistsError("Use the existing manifest or a fresh output directory")
    base_manifest_path = BASE / "manifest.json"
    base_comparison_path = BASE / "comparison.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    entries = base_manifest["entries"]
    validate_bidirectional_population(entries)

    base = json.loads(base_comparison_path.read_text())
    base_rows = {row["index"]: row for row in base["rows"]}
    if set(base_rows) != {
        entry["index"] for entry in entries
    }:
        raise ValueError("Frozen baseline population differs from VAL1142")

    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("eval_full_sequence_all_session.py").resolve(),
        Path(__file__).with_name("eval_multiview_boundary_student.py").resolve(),
        Path(__file__).with_name("eval_overlap_save_student.py").resolve(),
        Path(__file__).with_name("eval_student_window_gap.py").resolve(),
        Path(__file__).with_name("exact_dtw_cuda.py").resolve(),
        Path(__file__).with_name("fast_frc.py").resolve(),
    ]
    official = ROOT / "mam_reactor/framework/metrics"
    official_paths = [
        official / "FRC.py",
        official / "FRD.py",
        official / "S_MSE.py",
        official / "FRVar.py",
        official / "TLCC.py",
        ROOT / "mam_reactor/framework/modules/post_processor.py",
    ]
    manifest = {
        "schema_version": 1,
        "protocol": (
            "REACT2025 complete VAL1142, both directions, full source duration, "
            "K10, all opposite-role same-session GT, official Processor and AU "
            "rounding, official FRC/exact-FRD/FRDiv/FRVar/TLCC definitions"
        ),
        "main_table_requirement": {
            "contexts": 1142,
            "speaker_to_listener": 571,
            "listener_to_speaker": 571,
            "target_policy": "all opposite-role same-session GT",
            "sequence_policy": "full duration",
            "single_direction_policy": "reject",
        },
        "only_official_protocol_extension": "all same-session GT",
        "base_manifest": frozen(base_manifest_path),
        "base_comparison": frozen(base_comparison_path),
        "multiview_checkpoint": frozen(checkpoint),
        "implementations": [frozen(path) for path in implementation_paths],
        "official_sources": [frozen(path) for path in official_paths],
        "metric_execution": {
            "FRC": "vectorized scalar-equivalent official CCC/best-GT formula",
            "exact_FRD": "unbanded FP64 CUDA DP equivalent to official tslearn.dtw",
            "FRDiv": "official framework.metrics.S_MSE.compute_s_mse",
            "FRVar": "official framework.metrics.FRVar.compute_FRVar",
            "TLCC": "official framework.metrics.TLCC._func per context",
        },
        "powers": list(POWERS),
        "window": WINDOW,
        "stride": 375,
        "tf32": False,
        "entries": entries,
    }
    write_json(output / "manifest.json", manifest)
    (output / "rows").mkdir(exist_ok=True)
    for entry in entries:
        base_row = base_rows[entry["index"]]
        if not (
            base_row["clip_id"] == entry["clip_id"]
            and base_row["role"] == entry["role"]
        ):
            raise ValueError(f"Frozen row mismatch at index {entry['index']}")
        row = {
            "index": entry["index"],
            "clip_id": entry["clip_id"],
            "role": entry["role"],
            "length": entry["length"],
            "gt_count": base_row["gt_count"],
            "methods": {
                "single_view_nonoverlap": {
                    "metrics": scalar_metrics(base_row["models"]["student"]),
                    "source": "frozen_complete_bidirectional_baseline",
                },
            },
        }
        write_json(output / "rows" / f"{entry['index']:04d}.json", row)
    print("Prepared VAL1142 ablation manifest with one bidirectional baseline", flush=True)


def metric_means(rows: list[dict], method: str) -> dict:
    return {
        metric: float(np.mean([
            row["methods"][method]["metrics"][metric] for row in rows
        ]))
        for metric in METRICS
    }


def boundary_means(rows: list[dict], method: str) -> dict:
    records = [
        row["methods"][method]["boundary_jump"]
        for row in rows
        if row["methods"][method]["boundary_jump"]["ratio"] is not None
    ]
    return {
        key: float(np.mean([record[key] for record in records])) if records else None
        for key in (
            "mean_abs_jump_at_nonoverlap_boundaries",
            "mean_abs_jump_elsewhere",
            "ratio",
        )
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "contexts": len(rows),
        "methods": {
            method: {
                **metric_means(rows, method),
                "boundary_jump": boundary_means(rows, method),
            }
            for method in METHODS
        },
    }


def delta(new: dict, old: dict) -> dict:
    return {
        metric: {
            "absolute": new[metric] - old[metric],
            "relative_percent": 100.0 * (new[metric] - old[metric]) / old[metric]
            if old[metric] != 0 else None,
        }
        for metric in METRICS
    }


def aggregate(output: Path, manifest: dict) -> None:
    validate_bidirectional_population(manifest["entries"])
    rows = [
        json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
        for entry in manifest["entries"]
    ]
    if [row["clip_id"] for row in rows] != [
        entry["clip_id"] for entry in manifest["entries"]
    ]:
        raise ValueError("Row order differs from the frozen manifest")
    for row in rows:
        if set(row["methods"]) != set(METHODS):
            raise ValueError(f"Incomplete methods for {row['clip_id']}")
        for method in METHODS:
            if set(row["methods"][method]["metrics"]) != set(METRICS):
                raise ValueError(f"Incomplete five-metric result: {row['clip_id']}/{method}")
            if "boundary_jump" not in row["methods"][method]:
                raise ValueError(f"Missing boundary diagnostic: {row['clip_id']}/{method}")

    groups = {
        "speaker_to_listener": summarize([
            row for row in rows if row["role"] == "speaker"
        ]),
        "listener_to_speaker": summarize([
            row for row in rows if row["role"] == "listener"
        ]),
        "both_directions": summarize(rows),
    }
    both = groups["both_directions"]["methods"]
    comparisons = {
        "multiview_nonoverlap_minus_single_view_nonoverlap": delta(
            both["multiview_nonoverlap"], both["single_view_nonoverlap"]
        ),
        **{
            f"multiview_power_{power:g}_minus_multiview_nonoverlap": delta(
                both[f"multiview_power_{power:g}"],
                both["multiview_nonoverlap"],
            )
            for power in POWERS
        },
    }
    result = {
        "protocol": manifest["protocol"],
        "only_official_protocol_extension": manifest[
            "only_official_protocol_extension"
        ],
        "manifest_sha256": sha256(output / "manifest.json"),
        "metrics": groups,
        "deltas": comparisons,
        "selection": "p=4 is the quality/diversity compromise on VAL1142.",
        "limitations": [
            "Validation data, not hidden TEST.",
            "The target pool deliberately retains every same-session target.",
            "Official TLCC returns after query zero and can select the boundary lag when channels are constant; it is reproduced without redefining it.",
            "Single-direction and cropped-window result tables are rejected.",
        ],
        "provenance": {
            key: manifest[key]
            for key in (
                "base_manifest",
                "base_comparison",
                "multiview_checkpoint",
                "implementations",
                "official_sources",
                "metric_execution",
            )
        },
        "rows": rows,
    }
    write_json(output / "comparison.json", result)
    write_report(output / "REPORT.md", result)
    print(json.dumps({"metrics": groups, "deltas": comparisons}, indent=2))


def write_report(path: Path, result: dict) -> None:
    groups = result["metrics"]
    both = groups["both_directions"]["methods"]
    labels = {
        "single_view_nonoverlap": "单视图学生 + 非重叠",
        "multiview_nonoverlap": "多视图学生 + 非重叠",
        "multiview_power_1": "多视图学生 + p=1",
        "multiview_power_2": "多视图学生 + p=2",
        "multiview_power_4": "多视图学生 + p=4",
        "multiview_power_8": "多视图学生 + p=8",
    }
    lines = [
        "# REACT2025 双方向统一消融",
        "",
        result["protocol"] + "。",
        "",
        "所有论文消融主结果均为双方向等权 VAL1142；不发布单方向或中心裁剪结果表。",
        "",
        "| 方法 | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC/FRSyn ↓ | 边界跳变比 ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = both[method]
        lines.append(
            f"| {labels[method]} | {metrics['FRC']:.9f} | "
            f"{metrics['exact_FRD']:.9f} | {metrics['FRDiv']:.9f} | "
            f"{metrics['FRVar']:.9f} | {metrics['TLCC']:.9f} | "
            f"{metrics['boundary_jump']['ratio']:.9f} |"
        )
    lines.extend([
        "",
        "方向拆分及逐上下文结果见 `comparison.json`。`p=4` 按完整 VAL1142 的质量/多样性曲线确定。",
        "",
        "TLCC 按仓库官方实现原样复现。该实现只返回 query 0，并且常量通道可能使其选择边界 lag，因此报告该值是为了协议完整性，不把它解释成十个输出的稳健同步性。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --prepare")
        prepare(args.output, args.checkpoint)
        return

    manifest = json.loads((args.output / "manifest.json").read_text())
    validate_bidirectional_population(manifest["entries"])
    if args.aggregate:
        aggregate(args.output, manifest)
        return
    for key in (
        "base_manifest",
        "base_comparison",
        "multiview_checkpoint",
    ):
        if sha256(Path(manifest[key][0])) != manifest[key][1]:
            raise ValueError(f"Frozen {key} changed")
    for key in ("implementations", "official_sources"):
        for path, digest in manifest[key]:
            if sha256(Path(path)) != digest:
                raise ValueError(f"Frozen source changed: {path}")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(manifest["multiview_checkpoint"][0])
    model = load_model(checkpoint, device)
    mean = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/mean_face.npy"
    ).astype("float32")).reshape(58)
    std = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/std_face.npy"
    ).astype("float32")).reshape(58).clamp_min(1e-8)

    provenance = {
        "manifest_sha256": sha256(args.output / "manifest.json"),
        "shard": args.shard,
        "shards": args.shards,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "seed": 1234,
        "tf32": False,
    }
    provenance_path = args.output / f"provenance_{args.shard}.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Provenance mismatch")
    else:
        write_json(provenance_path, provenance)

    entries = sorted(
        manifest["entries"], key=lambda entry: (-entry["length"], entry["clip_id"])
    )
    schedule = entries[args.shard::args.shards]
    completed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for entry in schedule:
            row_path = args.output / "rows" / f"{entry['index']:04d}.json"
            row = json.loads(row_path.read_text())
            if set(row["methods"]) == set(METHODS) and all(
                set(row["methods"][method].get("metrics", {})) == set(METRICS)
                and "boundary_jump" in row["methods"][method]
                for method in METHODS
            ):
                continue
            tick = time.monotonic()
            stored = torch.load(
                BASE / "arrays" / f"{entry['index']:04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if stored["clip_id"] != entry["clip_id"]:
                raise ValueError("Base tensor clip mismatch")
            target = stored["targets"].float().contiguous()
            baseline_prediction = stored["predictions"]["student"].float().contiguous()
            length = entry["length"]
            raw_streams = [
                torch.from_numpy(
                    np.load(path, allow_pickle=False).astype("float32")
                ).reshape(length, -1)
                for path, _ in entry["source_files"]
            ]
            source_face = raw_streams[1]
            streams = list(raw_streams)
            streams[2] = (streams[2] - mean) / std

            baseline = row["methods"]["single_view_nonoverlap"]
            baseline["metrics"]["TLCC"] = official_tlcc(
                baseline_prediction, source_face
            )
            baseline["boundary_jump"] = boundary_jump(
                baseline_prediction, list(range(WINDOW, length, WINDOW))
            )

            nonoverlap_prediction = full_student(model, streams, device)
            nonoverlap_scored = score(nonoverlap_prediction, target)
            nonoverlap_metrics = {
                metric: nonoverlap_scored[metric]
                for metric in OFFICIAL_SCALARS
            }
            nonoverlap_metrics["TLCC"] = official_tlcc(
                nonoverlap_prediction, source_face
            )
            row["methods"]["multiview_nonoverlap"] = {
                "metrics": {metric: nonoverlap_metrics[metric] for metric in METRICS},
                "boundary_jump": boundary_jump(
                    nonoverlap_prediction, list(range(WINDOW, length, WINDOW))
                ),
                "source": "computed_complete_bidirectional_ablation",
            }

            starts = window_starts(length)
            valid_lengths = [min(WINDOW, length - start) for start in starts]
            windows = generate_windows(model, streams, starts, valid_lengths, device)
            _, _, switch_boundaries = overlap_save(windows, starts, length)
            for power in POWERS:
                method = f"multiview_power_{power:g}"
                prediction = official_prediction(
                    powered_center_overlap(windows, starts, length, power)
                )
                scored = score(prediction, target)
                metrics = {
                    metric: scored[metric] for metric in OFFICIAL_SCALARS
                }
                metrics["TLCC"] = official_tlcc(prediction, source_face)
                row["methods"][method] = {
                    "metrics": {metric: metrics[metric] for metric in METRICS},
                    "boundary_jump": boundary_jump(prediction, switch_boundaries),
                    "source": "computed_complete_bidirectional_ablation",
                }
            row["seconds"] = time.monotonic() - tick
            write_json(row_path, row)
            completed += 1
            print(json.dumps({
                "index": entry["index"],
                "role": entry["role"],
                "frames": length,
                "completed": completed,
                "total_shard": len(schedule),
                "seconds": row["seconds"],
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
            if args.max_contexts and completed >= args.max_contexts:
                break
    write_json(output_path := args.output / f"shard_status_{args.shard}.json", {
        "newly_completed": completed,
        "all_done": all(
            set(json.loads((args.output / "rows" / f"{entry['index']:04d}.json").read_text())["methods"])
            == set(METHODS)
            for entry in schedule
        ),
        "seconds": time.monotonic() - started,
    })
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
