"""Evaluate the final multi-view student on the official REACT2025 geometry.

The official test population, full sequence lengths, both directions, target
Processor output, AU rounding, and metric definitions are retained.  The only
protocol change is replacing the sampled ten targets with every opposite-role
recording from the same session.
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

from coreset_reactor.eval_multiview_boundary_student import load_model
from coreset_reactor.eval_overlap_save_student import powered_center_overlap
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    generate_windows,
    score,
    window_starts,
)
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.query0_au_transition import stabilize_query0_au_transitions


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "coreset_reactor/reports/full_sequence_bidir_allgt_20261018"
QUALITY_METRICS = ("FRC", "exact_FRD", "FRDiv", "FRVar")
METRICS = (*QUALITY_METRICS, "TLCC")


def official_tlcc(prediction: torch.Tensor, source_face: torch.Tensor) -> float:
    from framework.metrics.TLCC import _func

    previous = np.seterr(divide="ignore", invalid="ignore")
    try:
        value = float(_func(prediction, source_face))
    finally:
        np.seterr(**previous)
    if not np.isfinite(value):
        raise ValueError("Official TLCC returned a non-finite value")
    return value


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


def prepare(output: Path, checkpoint: Path, power: float) -> None:
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    if (output / "manifest.json").exists():
        raise FileExistsError("Use the existing manifest or a fresh output")
    base_manifest_path = BASE / "manifest.json"
    base_comparison_path = BASE / "comparison.json"
    base = json.loads(base_manifest_path.read_text())
    entries = base["entries"]
    if len(entries) != 1142:
        raise ValueError("Expected the official 1,142-context population")
    if sum(entry["role"] == "speaker" for entry in entries) != 571:
        raise ValueError("Expected 571 contexts in each direction")
    implementations = [
        Path(__file__).resolve(),
        Path(__file__).with_name("eval_overlap_save_student.py").resolve(),
        Path(__file__).with_name("query0_au_transition.py").resolve(),
        Path(__file__).with_name("eval_student_window_gap.py").resolve(),
        Path(__file__).with_name("eval_multiview_boundary_student.py").resolve(),
        Path(__file__).with_name("exact_dtw_cuda.py").resolve(),
        Path(__file__).with_name("fast_frc.py").resolve(),
    ]
    official = ROOT / "mam_reactor/framework/metrics"
    official_sources = [
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
            "REACT2025 full VAL1142, both directions, full sequences, K10, "
            "all opposite-role same-session GT, official Processor and AU "
            f"rounding; multi-view student with powered overlap p={power:g}; "
            "query-0 constant-AU single-transition postprocess"
        ),
        "only_official_protocol_extension": "all same-session GT",
        "base_manifest": frozen(base_manifest_path),
        "base_comparison": frozen(base_comparison_path),
        "checkpoint": frozen(checkpoint),
        "implementations": [frozen(path) for path in implementations],
        "official_sources": [frozen(path) for path in official_sources],
        "metric_execution": {
            "FRC": "vectorized scalar-equivalent official CCC/best-GT formula",
            "exact_FRD": "unbanded FP64 CUDA DP equivalent to official tslearn.dtw",
            "FRDiv": "official framework.metrics.S_MSE.compute_s_mse",
            "FRVar": "official framework.metrics.FRVar.compute_FRVar",
            "TLCC": "official framework.metrics.TLCC._func per context",
        },
        "postprocess": {
            "scope": "query 0 constant AU channels only",
            "rule": "one source-aligned binary transition per constant AU",
            "source_constant_policy": "leave prediction unchanged",
            "margin_frames": 49,
        },
        "window": WINDOW,
        "stride": 375,
        "power": power,
        "tf32": False,
        "entries": entries,
    }
    write_json(output / "manifest.json", manifest)

    (output / "rows").mkdir(exist_ok=True)
    print("Prepared VAL1142 manifest for both-direction evaluation", flush=True)


def group_metrics(rows: list[dict]) -> dict:
    return {
        "contexts": len(rows),
        **{
            metric: float(np.mean([row["metrics"][metric] for row in rows]))
            for metric in METRICS
        },
    }


def delta(new: dict, old: dict) -> dict:
    return {
        metric: {
            "absolute": new[metric] - old[metric],
            "relative_percent": 100 * (new[metric] - old[metric]) / old[metric],
        }
        for metric in QUALITY_METRICS
    }


def aggregate(output: Path, manifest: dict) -> None:
    rows = [
        json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
        for entry in manifest["entries"]
    ]
    if [row["clip_id"] for row in rows] != [
        entry["clip_id"] for entry in manifest["entries"]
    ]:
        raise ValueError("Row order differs from manifest")
    groups = {
        "speaker_to_listener": group_metrics(
            [row for row in rows if row["role"] == "speaker"]
        ),
        "listener_to_speaker": group_metrics(
            [row for row in rows if row["role"] == "listener"]
        ),
        "both_directions": group_metrics(rows),
    }
    base = json.loads(Path(manifest["base_comparison"][0]).read_text())
    comparisons = {}
    for group, base_group in (
        ("speaker_to_listener", "speaker"),
        ("listener_to_speaker", "listener"),
        ("both_directions", "both"),
    ):
        comparisons[group] = {
            name: delta(groups[group], base["metrics"][base_group]["models"][name])
            for name in ("student", "set_matching")
        }
    result = {
        "protocol": manifest["protocol"],
        "only_official_protocol_extension": manifest[
            "only_official_protocol_extension"
        ],
        "manifest_sha256": sha256(output / "manifest.json"),
        "metrics": groups,
        "delta_vs_previous_models": comparisons,
        "provenance": {
            key: manifest[key]
            for key in (
                "base_manifest",
                "base_comparison",
                "checkpoint",
                "implementations",
                "official_sources",
                "metric_execution",
                "postprocess",
            )
        },
        "limitations": [
            "Validation data, not hidden TEST.",
            "The target pool deliberately retains every same-session target.",
            "FRC and FRD use exact-equivalent accelerated implementations; parity is audited separately against the official functions.",
        ],
        "rows": rows,
    }
    write_json(output / "comparison.json", result)
    print(json.dumps({"metrics": groups, "deltas": comparisons}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--balanced-listener-shards",
        action="store_true",
        help="Partition only the missing reverse direction for balanced resume",
    )
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --prepare")
        prepare(args.output, args.checkpoint, args.power)
        return
    manifest = json.loads((args.output / "manifest.json").read_text())
    if args.aggregate:
        aggregate(args.output, manifest)
        return
    for key in (
        "base_manifest",
        "base_comparison",
        "checkpoint",
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
    checkpoint = Path(manifest["checkpoint"][0])
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
        "balanced_listener_shards": args.balanced_listener_shards,
    }
    provenance_prefix = (
        "balanced_provenance" if args.balanced_listener_shards else "provenance"
    )
    provenance_path = args.output / f"{provenance_prefix}_{args.shard}.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Provenance mismatch")
    else:
        write_json(provenance_path, provenance)

    entries = sorted(
        manifest["entries"], key=lambda item: (-item["length"], item["clip_id"])
    )
    if args.balanced_listener_shards:
        entries = [entry for entry in entries if entry["role"] == "listener"]
    schedule = entries[args.shard::args.shards]
    completed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for entry in schedule:
            row_path = args.output / "rows" / f"{entry['index']:04d}.json"
            if row_path.exists():
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
            length = entry["length"]
            streams = [
                torch.from_numpy(np.load(path).astype("float32")).reshape(length, -1)
                for path, _ in entry["source_files"]
            ]
            streams[2] = (streams[2] - mean) / std
            starts = window_starts(length)
            valid_lengths = [min(WINDOW, length - start) for start in starts]
            windows = generate_windows(model, streams, starts, valid_lengths, device)
            prediction = official_prediction(powered_center_overlap(
                windows, starts, length, manifest["power"]
            ))
            prediction = stabilize_query0_au_transitions(prediction, streams[1])
            metrics = score(prediction, target)
            metrics["TLCC"] = official_tlcc(prediction, streams[1])
            row = {
                "index": entry["index"],
                "clip_id": entry["clip_id"],
                "role": entry["role"],
                "length": length,
                "gt_count": int(target.shape[0]),
                "metrics": metrics,
                "source": "computed_bidir_p4_with_query0_au_transitions",
                "seconds": time.monotonic() - tick,
            }
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
    write_json(args.output / f"shard_status_{args.shard}.json", {
        "newly_completed": completed,
        "all_done": all(
            (args.output / "rows" / f"{entry['index']:04d}.json").exists()
            for entry in schedule
        ),
        "seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    main()
