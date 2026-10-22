"""Evaluate a multi-view distilled student with boundary-aware inference."""
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
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    STRIDE,
    boundary_jump,
    generate_windows,
    overlap_add,
    position_starts,
    score,
    window_starts,
)
from coreset_reactor.evaluate import official_prediction
from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "coreset_reactor/reports/full_sequence_bidir_allgt_20261018"
CONTROL = ROOT / "coreset_reactor/reports/window_gap_audit_20261018/comparison.json"
METHODS = ("nonoverlap", "overlap_center_weighted")
POSITIONS = ("head", "center", "tail")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu")
    config = state["student_config"]
    model = ExpertConditionalREGNN(
        ExpertStudentConfig(
            backbone=ConditionalREGNNConfig(**config["backbone"]),
            num_experts=config["num_experts"],
            expert_embedding_dim=config["expert_embedding_dim"],
        )
    ).to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    return model


def prepare(output: Path, checkpoint: Path) -> None:
    base_manifest_path = BASE / "manifest.json"
    base = json.loads(base_manifest_path.read_text())
    entries = [entry for entry in base["entries"] if entry["role"] == "speaker"]
    if len(entries) != 571:
        raise ValueError("Expected forward VAL571")
    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("eval_student_window_gap.py").resolve(),
        Path(__file__).with_name("eval_full_sequence_all_session.py").resolve(),
    ]
    manifest = {
        "schema_version": 1,
        "protocol": (
            "Multi-view student forward VAL571, all-session opposite-role GT; "
            "standalone head/center/tail and full nonoverlap/stride375 "
            "center-weighted overlap inference"
        ),
        "base_manifest": [str(base_manifest_path.resolve()), sha(base_manifest_path)],
        "control_comparison": [str(CONTROL.resolve()), sha(CONTROL)],
        "student_checkpoint": [str(checkpoint.resolve()), sha(checkpoint.resolve())],
        "implementations": [[str(path), sha(path)] for path in implementation_paths],
        "window": WINDOW,
        "stride": STRIDE,
        "tf32": False,
        "entries": entries,
    }
    if (output / "manifest.json").exists():
        raise FileExistsError("Use an existing manifest or a fresh output directory")
    write(output / "manifest.json", manifest)


def metric_means(rows, group, name):
    return {
        metric: float(np.mean([row[group][name][metric] for row in rows]))
        for metric in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }


def delta(new, old):
    return {
        metric: {
            "absolute": new[metric] - old[metric],
            "relative_percent": 100.0 * (new[metric] - old[metric]) / old[metric],
        }
        for metric in old
    }


def summarize(rows):
    positions = {name: metric_means(rows, "positions", name) for name in POSITIONS}
    full = {name: metric_means(rows, "full_sequence", name) for name in METHODS}
    seams = {}
    for method in METHODS:
        valid = [row["seams"][method] for row in rows
                 if row["seams"][method]["ratio"] is not None]
        seams[method] = {
            key: float(np.mean([item[key] for item in valid])) if valid else None
            for key in ("mean_abs_jump_at_nonoverlap_boundaries",
                        "mean_abs_jump_elsewhere", "ratio")
        }
    return {
        "contexts": len(rows),
        "positions": positions,
        "full_sequence": full,
        "boundary_inference_delta": delta(
            full["overlap_center_weighted"], full["nonoverlap"]
        ),
        "seams": seams,
    }


def add_control_comparison(summary, control):
    mapping = {
        "all_forward": "all_forward",
        "cropped_only": "cropped_only",
        "short_or_equal_control": "short_or_equal_control",
    }
    for group, control_group in mapping.items():
        old = control["groups"][control_group]
        new = summary[group]
        new["training_delta_same_inference"] = {
            "center_standalone": delta(
                new["positions"]["center"], old["positions"]["center"]
            ),
            "full_nonoverlap": delta(
                new["full_sequence"]["nonoverlap"],
                old["full_sequence"]["nonoverlap"],
            ),
            "full_overlap_center_weighted": delta(
                new["full_sequence"]["overlap_center_weighted"],
                old["full_sequence"]["overlap_center_weighted"],
            ),
        }


def aggregate(output: Path, manifest) -> None:
    rows = [json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
            for entry in manifest["entries"]]
    if [row["clip_id"] for row in rows] != [entry["clip_id"] for entry in manifest["entries"]]:
        raise ValueError("Row order differs from manifest")
    groups = {
        "all_forward": summarize(rows),
        "cropped_only": summarize([row for row in rows if row["length"] > WINDOW]),
        "short_or_equal_control": summarize([row for row in rows if row["length"] <= WINDOW]),
    }
    add_control_comparison(groups, json.loads(CONTROL.read_text()))
    result = {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha(output / "manifest.json"),
        "groups": groups,
        "limitations": [
            "Forward validation only, not hidden TEST.",
            "All same-session GT retained deliberately.",
            "Overlap-add changes local context and fusion but supplies no recurrent cross-window state.",
        ],
        "rows": rows,
    }
    write(output / "comparison.json", result)
    print(json.dumps(groups, indent=2))


def main():
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
    if args.aggregate:
        aggregate(args.output, manifest)
        return
    if sha(Path(manifest["base_manifest"][0])) != manifest["base_manifest"][1]:
        raise ValueError("Base manifest changed")
    if sha(Path(manifest["control_comparison"][0])) != manifest["control_comparison"][1]:
        raise ValueError("Control comparison changed")
    checkpoint = Path(manifest["student_checkpoint"][0])
    if sha(checkpoint) != manifest["student_checkpoint"][1]:
        raise ValueError("Student checkpoint changed")
    for path, digest in manifest["implementations"]:
        if sha(Path(path)) != digest:
            raise ValueError(f"Evaluation implementation changed: {path}")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(checkpoint, device)
    mean = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/mean_face.npy"
    ).astype("float32")).reshape(58)
    std = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/std_face.npy"
    ).astype("float32")).reshape(58).clamp_min(1e-8)
    (args.output / "rows").mkdir(exist_ok=True)
    provenance = {
        "manifest_sha256": sha(args.output / "manifest.json"),
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
        write(provenance_path, provenance)

    entries = sorted(manifest["entries"], key=lambda item: (-item["length"], item["clip_id"]))
    schedule = entries[args.shard::args.shards]
    started = time.monotonic()
    completed = 0
    with torch.inference_mode():
        for entry in schedule:
            row_path = args.output / "rows" / f"{entry['index']:04d}.json"
            if row_path.exists():
                continue
            tick = time.monotonic()
            stored = torch.load(BASE / "arrays" / f"{entry['index']:04d}.pt", map_location="cpu")
            if stored["clip_id"] != entry["clip_id"]:
                raise ValueError("Base tensor clip mismatch")
            target = stored["targets"].float().contiguous()
            length = entry["length"]
            streams = [torch.from_numpy(np.load(path).astype("float32")).reshape(length, -1)
                       for path, _ in entry["source_files"]]
            streams[2] = (streams[2] - mean) / std

            starts_by_position = position_starts(length)
            valid = min(length, WINDOW)
            unique_starts = list(dict.fromkeys(starts_by_position.values()))
            standalone_raw = generate_windows(
                model, streams, unique_starts, [valid] * len(unique_starts), device
            )
            standalone = {
                name: official_prediction(standalone_raw[unique_starts.index(start), :, :valid])
                for name, start in starts_by_position.items()
            }
            position_scores = {
                name: score(
                    standalone[name],
                    target[:, starts_by_position[name]:starts_by_position[name] + valid],
                )
                for name in POSITIONS
            }

            nonoverlap = full_student(model, streams, device)
            overlap_starts = window_starts(length)
            overlap_valid = [min(WINDOW, length - start) for start in overlap_starts]
            overlap_windows = generate_windows(
                model, streams, overlap_starts, overlap_valid, device
            )
            centered = official_prediction(overlap_add(
                overlap_windows, overlap_starts, length, "center"
            ))
            predictions = {
                "nonoverlap": nonoverlap,
                "overlap_center_weighted": centered,
            }
            full_scores = {name: score(prediction, target)
                           for name, prediction in predictions.items()}
            boundaries = list(range(WINDOW, length, WINDOW))
            row = {
                "index": entry["index"],
                "clip_id": entry["clip_id"],
                "role": entry["role"],
                "length": length,
                "gt_count": int(target.shape[0]),
                "position_starts": starts_by_position,
                "overlap_starts": overlap_starts,
                "positions": position_scores,
                "full_sequence": full_scores,
                "seams": {name: boundary_jump(prediction, boundaries)
                          for name, prediction in predictions.items()},
                "seconds": time.monotonic() - tick,
            }
            write(row_path, row)
            completed += 1
            print(json.dumps({
                "index": entry["index"], "frames": length, "completed": completed,
                "total_shard": len(schedule), "seconds": row["seconds"],
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
            if args.max_contexts and completed >= args.max_contexts:
                break
    write(args.output / f"shard_status_{args.shard}.json", {
        "newly_completed": completed,
        "all_done": all((args.output / "rows" / f"{entry['index']:04d}.json").exists()
                        for entry in schedule),
        "seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    main()
