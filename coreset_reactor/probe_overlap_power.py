"""Fast FRC/diversity probe for powered center-window overlap fusion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from coreset_reactor.eval_multiview_boundary_student import load_model
from coreset_reactor.eval_overlap_save_student import (
    BASE,
    REFERENCE,
    overlap_save,
    powered_center_overlap,
    sha256,
)
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    boundary_jump,
    generate_windows,
    window_starts,
)
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.fast_frc import candidate_frc
from framework.metrics import compute_FRVar, compute_s_mse


ROOT = Path(__file__).resolve().parents[1]


def fast_score(prediction, target):
    return {
        "FRC": float(candidate_frc(target.numpy(), prediction.numpy()).sum()),
        "FRDiv": float(compute_s_mse([prediction])),
        "FRVar": float(compute_FRVar([prediction])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=120)
    parser.add_argument("--powers", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.powers)) != len(args.powers):
        raise ValueError("powers must be unique")

    reference_manifest_path = REFERENCE / "manifest.json"
    reference_comparison_path = REFERENCE / "comparison.json"
    manifest = json.loads(reference_manifest_path.read_text())
    long_entries = [entry for entry in manifest["entries"] if entry["length"] > WINDOW]
    if not 0 < args.contexts <= len(long_entries):
        raise ValueError("contexts out of range")
    rng = np.random.default_rng(args.seed)
    selected_indices = sorted(rng.choice(
        len(long_entries), size=args.contexts, replace=False
    ).tolist())
    entries = [long_entries[index] for index in selected_indices]

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.checkpoint, device)
    mean = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/mean_face.npy"
    ).astype("float32")).reshape(58)
    std = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/std_face.npy"
    ).astype("float32")).reshape(58).clamp_min(1e-8)

    methods = [f"power_{power:g}" for power in args.powers]
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for completed, entry in enumerate(entries, 1):
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
            valid = [min(WINDOW, length - start) for start in starts]
            windows = generate_windows(model, streams, starts, valid, device)
            _, _, switches = overlap_save(windows, starts, length)
            row = {
                "index": entry["index"],
                "clip_id": entry["clip_id"],
                "length": length,
                "methods": {},
            }
            for name, power in zip(methods, args.powers):
                prediction = official_prediction(
                    powered_center_overlap(windows, starts, length, power)
                )
                row["methods"][name] = {
                    **fast_score(prediction, target),
                    "switch_seam": boundary_jump(prediction, switches),
                }
            records.append(row)
            print(json.dumps({
                "completed": completed,
                "total": len(entries),
                "frames": length,
                "elapsed_s": time.monotonic() - started,
            }), flush=True)

    summary = {}
    for name in methods:
        summary[name] = {
            **{
                metric: float(np.mean([
                    row["methods"][name][metric] for row in records
                ]))
                for metric in ("FRC", "FRDiv", "FRVar")
            },
            "switch_seam": {
                key: float(np.mean([
                    row["methods"][name]["switch_seam"][key]
                    for row in records
                ]))
                for key in (
                    "mean_abs_jump_at_nonoverlap_boundaries",
                    "mean_abs_jump_elsewhere",
                    "ratio",
                )
            },
        }
    payload = {
        "protocol": (
            "Deterministic random subset of long forward VAL contexts; no exact "
            "FRD; fixed powers selected before evaluation"
        ),
        "contexts": len(entries),
        "seed": args.seed,
        "powers": args.powers,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "inputs": {
            "checkpoint": [str(args.checkpoint.resolve()), sha256(args.checkpoint)],
            "reference_manifest": [
                str(reference_manifest_path.resolve()), sha256(reference_manifest_path)
            ],
            "reference_comparison": [
                str(reference_comparison_path.resolve()), sha256(reference_comparison_path)
            ],
            "implementation": [str(Path(__file__).resolve()), sha256(Path(__file__))],
        },
        "summary": summary,
        "rows": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
