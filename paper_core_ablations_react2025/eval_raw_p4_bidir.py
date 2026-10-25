"""Evaluate raw and fixed-postprocess p=4 output in one VAL1142 pass."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

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
from coreset_reactor.eval_multiview_p4_bidir_all_session import (
    BASE,
    METRICS,
    frozen,
    official_tlcc,
    sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def prepare(output: Path, checkpoint: Path) -> None:
    if (output / "manifest.json").exists():
        raise FileExistsError("Use the existing manifest or a fresh output")
    base_manifest_path = BASE / "manifest.json"
    base = json.loads(base_manifest_path.read_text())
    entries = base["entries"]
    if len(entries) != 1142:
        raise ValueError("Expected VAL1142")
    if sum(entry["role"] == "speaker" for entry in entries) != 571:
        raise ValueError("Expected 571 contexts in each direction")
    implementations = [
        Path(__file__).resolve(),
        ROOT / "coreset_reactor/eval_multiview_p4_bidir_all_session.py",
        ROOT / "coreset_reactor/eval_overlap_save_student.py",
        ROOT / "coreset_reactor/eval_student_window_gap.py",
        ROOT / "coreset_reactor/eval_multiview_boundary_student.py",
        ROOT / "coreset_reactor/query0_au_transition.py",
        ROOT / "coreset_reactor/exact_dtw_cuda.py",
        ROOT / "coreset_reactor/fast_frc.py",
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
            "REACT2025 full VAL1142, both directions, full sequences, K10, all "
            "opposite-role same-session GT, official Processor/AU rounding/five "
            "metrics; raw and fixed query-0-transition multi-view student output "
            "with overlap p=4"
        ),
        "output_kind": "raw_and_fixed_query0_transition_postprocess",
        "only_official_protocol_extension": "all same-session GT",
        "base_manifest": frozen(base_manifest_path),
        "checkpoint": frozen(checkpoint),
        "implementations": [frozen(path) for path in implementations],
        "official_sources": [frozen(path) for path in official_sources],
        "window": WINDOW,
        "stride": 375,
        "power": 4.0,
        "tf32": False,
        "entries": entries,
    }
    write_json(output / "manifest.json", manifest)
    (output / "rows").mkdir(exist_ok=True)


def group_metrics(rows: list[dict], key: str) -> dict:
    return {
        "contexts": len(rows),
        **{
            metric: float(np.mean([row[key][metric] for row in rows]))
            for metric in METRICS
        },
    }


def aggregate(output: Path, manifest: dict) -> None:
    rows = [
        json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
        for entry in manifest["entries"]
    ]
    if [row["clip_id"] for row in rows] != [entry["clip_id"] for entry in manifest["entries"]]:
        raise ValueError("Row order differs from manifest")
    for row in rows:
        for key in ("raw_metrics", "postprocessed_metrics"):
            if not set(METRICS).issubset(row[key]):
                raise ValueError(f"Incomplete {key} for {row['clip_id']}")

    def grouped(key: str) -> dict:
        return {
            "speaker_to_listener": group_metrics(
                [row for row in rows if row["role"] == "speaker"], key),
            "listener_to_speaker": group_metrics(
                [row for row in rows if row["role"] == "listener"], key),
            "both_directions": group_metrics(rows, key),
        }

    raw_groups = grouped("raw_metrics")
    post_groups = grouped("postprocessed_metrics")
    common = {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha256(output / "manifest.json"),
        "provenance": {key: manifest[key] for key in (
            "base_manifest", "checkpoint", "implementations", "official_sources"
        )},
    }
    raw_rows = [
        {**{key: value for key, value in row.items()
            if key not in ("raw_metrics", "postprocessed_metrics")},
         "metrics": row["raw_metrics"], "source": "computed_bidir_raw_p4"}
        for row in rows
    ]
    post_rows = [
        {**{key: value for key, value in row.items()
            if key not in ("raw_metrics", "postprocessed_metrics")},
         "metrics": row["postprocessed_metrics"],
         "source": "computed_bidir_p4_with_query0_au_transitions"}
        for row in rows
    ]
    write_json(output / "comparison_raw.json", {
        **common, "output_kind": "raw_before_query0_transition_postprocess",
        "metrics": raw_groups, "rows": raw_rows,
    })
    write_json(output / "comparison_postprocessed.json", {
        **common, "output_kind": "fixed_query0_transition_postprocess",
        "metrics": post_groups, "rows": post_rows,
    })
    write_json(output / "comparison.json", {
        **common,
        "output_kind": manifest["output_kind"],
        "changed_contexts": sum(row["postprocess_changed"] for row in rows),
        "metrics": {"raw": raw_groups, "postprocessed": post_groups},
    })
    print(json.dumps({
        "changed_contexts": sum(row["postprocess_changed"] for row in rows),
        "raw": raw_groups, "postprocessed": post_groups,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --prepare")
        prepare(args.output, args.checkpoint)
        return
    manifest = json.loads((args.output / "manifest.json").read_text())
    if args.aggregate:
        aggregate(args.output, manifest)
        return
    for key in ("base_manifest", "checkpoint"):
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
    model = load_model(Path(manifest["checkpoint"][0]), device)
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

    entries = sorted(manifest["entries"], key=lambda item: (-item["length"], item["clip_id"]))
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
            source_face = streams[1]
            streams[2] = (streams[2] - mean) / std
            starts = window_starts(length)
            valid_lengths = [min(WINDOW, length - start) for start in starts]
            windows = generate_windows(model, streams, starts, valid_lengths, device)
            prediction = official_prediction(powered_center_overlap(
                windows, starts, length, manifest["power"]
            ))
            raw_metrics = score(prediction, target)
            raw_metrics["TLCC"] = official_tlcc(prediction, source_face)
            raw_metrics = {metric: raw_metrics[metric] for metric in METRICS}
            postprocessed = stabilize_query0_au_transitions(prediction, source_face)
            changed = not torch.equal(postprocessed, prediction)
            if changed:
                postprocessed_metrics = score(postprocessed, target)
                postprocessed_metrics["TLCC"] = official_tlcc(
                    postprocessed, source_face)
                postprocessed_metrics = {
                    metric: postprocessed_metrics[metric] for metric in METRICS
                }
            else:
                postprocessed_metrics = dict(raw_metrics)
            row = {
                "index": entry["index"],
                "clip_id": entry["clip_id"],
                "role": entry["role"],
                "length": length,
                "gt_count": int(target.shape[0]),
                "raw_metrics": raw_metrics,
                "postprocessed_metrics": postprocessed_metrics,
                "postprocess_changed": changed,
                "seconds": time.monotonic() - tick,
            }
            write_json(row_path, row)
            completed += 1
            print(json.dumps({
                "index": entry["index"], "role": entry["role"], "frames": length,
                "completed": completed, "total_shard": len(schedule),
                "seconds": row["seconds"], "elapsed_s": time.monotonic() - started,
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
