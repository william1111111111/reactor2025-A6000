"""Evaluate sharpened center-weighted overlap inference for the multi-view student."""
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
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    boundary_jump,
    center_weights,
    generate_windows,
    overlap_add,
    score,
    window_starts,
)
from coreset_reactor.evaluate import official_prediction


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "coreset_reactor/reports/multiview_boundary_student_seed1_20261018"
    / "eval_forward_val571"
)
BASE = ROOT / "coreset_reactor/reports/full_sequence_bidir_allgt_20261018"


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


def overlap_save(window_predictions, starts, length):
    """Select the prediction with the largest center weight at every frame."""
    if window_predictions.shape[0] != len(starts):
        raise ValueError("Window count and start count differ")
    result = torch.empty((10, length, 25), dtype=window_predictions.dtype)
    best_weight = torch.full((length,), -1.0, dtype=torch.float32)
    selected_window = torch.full((length,), -1, dtype=torch.long)
    weights = center_weights(WINDOW)
    for window_index, (prediction, start) in enumerate(
        zip(window_predictions, starts)
    ):
        valid = min(WINDOW, length - start)
        local_weight = weights[:valid]
        destination = slice(start, start + valid)
        replace = local_weight > best_weight[destination]
        if replace.any():
            frame_ids = torch.arange(start, start + valid)[replace]
            result[:, frame_ids] = prediction[:, :valid][:, replace]
            best_weight[frame_ids] = local_weight[replace]
            selected_window[frame_ids] = window_index
    if (selected_window < 0).any():
        raise ValueError("Overlap-save left uncovered frames")
    switches = (selected_window[1:] != selected_window[:-1]).nonzero().flatten()
    switch_boundaries = (switches + 1).tolist()
    return result, selected_window, switch_boundaries


def powered_center_overlap(window_predictions, starts, length, power):
    """Smooth overlap-add with a sharper, powered triangular center window."""
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    accumulator = torch.zeros((10, length, 25), dtype=window_predictions.dtype)
    denominator = torch.zeros(length, dtype=torch.float32)
    weights = center_weights(WINDOW).pow(power)
    for prediction, start in zip(window_predictions, starts):
        valid = min(WINDOW, length - start)
        weight = weights[:valid]
        accumulator[:, start:start + valid] += (
            prediction[:, :valid] * weight[None, :, None]
        )
        denominator[start:start + valid] += weight
    if not torch.all(denominator > 0):
        raise ValueError("Powered overlap-add left uncovered frames")
    return accumulator / denominator[None, :, None]


def delta(new, old):
    return {
        metric: {
            "absolute": new[metric] - old[metric],
            "relative_percent": 100.0 * (new[metric] - old[metric]) / old[metric],
        }
        for metric in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }


def metric_means(rows):
    return {
        metric: float(np.mean([
            row["powered_center_overlap"][metric] for row in rows
        ]))
        for metric in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }


def seam_means(rows, name):
    valid = [row["switch_seams"][name] for row in rows
             if row["switch_seams"][name]["ratio"] is not None]
    return {
        key: float(np.mean([row[key] for row in valid])) if valid else None
        for key in (
            "mean_abs_jump_at_nonoverlap_boundaries",
            "mean_abs_jump_elsewhere",
            "ratio",
        )
    }


def summarize(rows, reference_rows):
    metrics = metric_means(rows)
    reference = {
        metric: float(np.mean([
            reference_rows[row["clip_id"]]["full_sequence"]
            ["overlap_center_weighted"][metric]
            for row in rows
        ]))
        for metric in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }
    nonoverlap = {
        metric: float(np.mean([
            reference_rows[row["clip_id"]]["full_sequence"]["nonoverlap"][metric]
            for row in rows
        ]))
        for metric in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }
    return {
        "contexts": len(rows),
        "powered_center_overlap": metrics,
        "reference_overlap_center_weighted": reference,
        "reference_nonoverlap": nonoverlap,
        "delta_vs_overlap_center_weighted": delta(metrics, reference),
        "delta_vs_nonoverlap": delta(metrics, nonoverlap),
        "switch_seams": {
            name: seam_means(rows, name)
            for name in ("powered_center_overlap", "overlap_center_weighted")
        },
    }


def prepare(output: Path, checkpoint: Path, power: float) -> None:
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    reference_manifest_path = REFERENCE / "manifest.json"
    reference_comparison_path = REFERENCE / "comparison.json"
    pilot_path = ROOT / "coreset_reactor/reports/overlap_power_probe120_20261018.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    implementations = [
        Path(__file__).resolve(),
        Path(__file__).with_name("eval_student_window_gap.py").resolve(),
        Path(__file__).with_name("eval_multiview_boundary_student.py").resolve(),
    ]
    manifest = {
        "schema_version": 1,
        "protocol": (
            f"Powered center overlap with fixed triangular-weight exponent "
            f"p={power:g}; forward VAL571 with all same-session opposite-role GT"
        ),
        "reference_manifest": [
            str(reference_manifest_path.resolve()), sha256(reference_manifest_path)
        ],
        "reference_comparison": [
            str(reference_comparison_path.resolve()), sha256(reference_comparison_path)
        ],
        "checkpoint": [str(checkpoint.resolve()), sha256(checkpoint)],
        "selection_pilot": [str(pilot_path.resolve()), sha256(pilot_path)],
        "implementations": [[str(path), sha256(path)] for path in implementations],
        "window": WINDOW,
        "stride": 375,
        "power": power,
        "tf32": False,
        "entries": reference_manifest["entries"],
    }
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    write_json(output / "manifest.json", manifest)


def aggregate(output: Path, manifest) -> None:
    rows = [
        json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
        for entry in manifest["entries"]
    ]
    reference_document = json.loads(
        Path(manifest["reference_comparison"][0]).read_text()
    )
    reference_rows = {row["clip_id"]: row for row in reference_document["rows"]}
    result = {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha256(output / "manifest.json"),
        "groups": {
            "all_forward": summarize(rows, reference_rows),
            "cropped_only": summarize(
                [row for row in rows if row["length"] > WINDOW], reference_rows
            ),
            "short_or_equal_control": summarize(
                [row for row in rows if row["length"] <= WINDOW], reference_rows
            ),
        },
        "limitations": [
            "Forward validation only, not hidden TEST.",
            "The fixed exponent was selected from a four-point VAL120 pilot and must remain frozen for hidden TEST.",
            "The fusion rule has no learned component and uses no GT per context.",
        ],
        "rows": rows,
    }
    write_json(output / "comparison.json", result)
    print(json.dumps(result["groups"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--power", type=float, default=4.0)
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
        "reference_manifest", "reference_comparison", "checkpoint", "selection_pilot"
    ):
        if sha256(Path(manifest[key][0])) != manifest[key][1]:
            raise ValueError(f"Frozen {key} changed")
    for path, digest in manifest["implementations"]:
        if sha256(Path(path)) != digest:
            raise ValueError(f"Implementation changed: {path}")
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
    (args.output / "rows").mkdir(exist_ok=True)

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
        manifest["entries"], key=lambda item: (-item["length"], item["clip_id"])
    )
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
                BASE / "arrays" / f"{entry['index']:04d}.pt", map_location="cpu"
            )
            target = stored["targets"].float().contiguous()
            length = entry["length"]
            streams = [
                torch.from_numpy(np.load(path).astype("float32")).reshape(length, -1)
                for path, _ in entry["source_files"]
            ]
            streams[2] = (streams[2] - mean) / std
            starts = window_starts(length)
            valid_lengths = [min(WINDOW, length - start) for start in starts]
            windows = generate_windows(
                model, streams, starts, valid_lengths, device
            )
            _, selected, switches = overlap_save(windows, starts, length)
            powered = official_prediction(powered_center_overlap(
                windows, starts, length, manifest["power"]
            ))
            soft = official_prediction(overlap_add(windows, starts, length, "center"))
            row = {
                "index": entry["index"],
                "clip_id": entry["clip_id"],
                "role": entry["role"],
                "length": length,
                "gt_count": int(target.shape[0]),
                "starts": starts,
                "switch_boundaries": switches,
                "selected_window_counts": torch.bincount(
                    selected, minlength=len(starts)
                ).tolist(),
                "powered_center_overlap": score(powered, target),
                "switch_seams": {
                    "powered_center_overlap": boundary_jump(powered, switches),
                    "overlap_center_weighted": boundary_jump(soft, switches),
                },
                "seconds": time.monotonic() - tick,
            }
            write_json(row_path, row)
            completed += 1
            print(json.dumps({
                "index": entry["index"],
                "frames": length,
                "completed": completed,
                "total_shard": len(schedule),
                "seconds": row["seconds"],
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
