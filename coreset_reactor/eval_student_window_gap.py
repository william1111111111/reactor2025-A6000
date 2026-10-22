"""Diagnose the 750-frame training/inference gap for the frozen student.

The forward VAL population and full-duration all-session GT tensors are reused
from the frozen matched evaluation.  Three controlled comparisons are made:

1. standalone head/center/tail windows of the same length;
2. standalone center inference versus the same center interval cut from the
   non-overlapping full-sequence prediction;
3. full-sequence non-overlap versus 50%-overlap inference, with uniform and
   center-weighted overlap-add.
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

from coreset_reactor.evaluate import official_prediction
from coreset_reactor.exact_dtw_cuda import pairwise_frd
from coreset_reactor.fast_frc import candidate_frc
from framework.metrics import compute_FRVar, compute_s_mse
from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "coreset_reactor/reports/full_sequence_bidir_allgt_20261018"
STUDENT = ROOT / "coreset_reactor/reports/mode_reference_student_seed1/basic/student.pth"
WINDOW = 750
STRIDE = 375
METHODS = ("nonoverlap", "overlap_uniform", "overlap_center_weighted")
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


def window_starts(length: int, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    """Cover every frame with full windows, aligning the final window to tail."""
    if length < 1 or window < 1 or stride < 1:
        raise ValueError("length, window and stride must be positive")
    if length <= window:
        return [0]
    final = length - window
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def position_starts(length: int, window: int = WINDOW) -> dict[str, int]:
    valid = min(length, window)
    return {
        "head": 0,
        "center": (length - valid) // 2,
        "tail": length - valid,
    }


def center_weights(length: int) -> torch.Tensor:
    """Positive triangular weights; division exactly restores a lone window."""
    positions = torch.arange(length, dtype=torch.float32)
    midpoint = (length - 1) / 2.0
    return 1.0 - (positions - midpoint).abs() / ((length + 1) / 2.0)


def load_model(device: torch.device):
    state = torch.load(STUDENT, map_location="cpu")
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


def generate_windows(model, streams, starts, valid_lengths, device, batch_size=8):
    """Return continuous predictions as [N,K,W,25]."""
    padded = []
    for stream in streams:
        items = []
        for start, valid in zip(starts, valid_lengths):
            item = stream[start:start + valid]
            if valid < WINDOW:
                item = torch.nn.functional.pad(item, (0, 0, 0, WINDOW - valid))
            items.append(item)
        padded.append(torch.stack(items))
    lengths = torch.tensor(valid_lengths, dtype=torch.long)
    outputs = []
    for begin in range(0, len(starts), batch_size):
        batch = [item[begin:begin + batch_size].to(device) for item in padded]
        prediction = model.generate(
            *batch, lengths[begin:begin + batch_size].to(device)
        )["prediction"]
        outputs.append(prediction.detach().float().cpu())
    result = torch.cat(outputs)
    if result.shape != (len(starts), 10, WINDOW, 25):
        raise ValueError(f"Unexpected prediction shape {tuple(result.shape)}")
    return result


def overlap_add(window_predictions, starts, length, weighting):
    accumulator = torch.zeros((10, length, 25), dtype=torch.float32)
    denominator = torch.zeros(length, dtype=torch.float32)
    for prediction, start in zip(window_predictions, starts):
        valid = min(WINDOW, length - start)
        if weighting == "uniform":
            weight = torch.ones(valid, dtype=torch.float32)
        elif weighting == "center":
            weight = center_weights(WINDOW)[:valid]
        else:
            raise ValueError(weighting)
        accumulator[:, start:start + valid] += prediction[:, :valid] * weight[None, :, None]
        denominator[start:start + valid] += weight
    if not torch.all(denominator > 0):
        raise ValueError("Overlap-add left uncovered frames")
    return accumulator / denominator[None, :, None]


def score(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    pred_np = prediction.numpy()
    target_np = target.numpy()
    frc = candidate_frc(target_np, pred_np)
    frd = pairwise_frd(pred_np, target_np).min(axis=1)
    return {
        "FRC": float(frc.sum()),
        "exact_FRD": float(frd.sum()),
        "FRDiv": float(compute_s_mse([prediction])),
        "FRVar": float(compute_FRVar([prediction])),
        "FRC_per_prediction": frc.tolist(),
        "FRD_per_prediction": frd.tolist(),
    }


def boundary_jump(prediction: torch.Tensor, boundaries: list[int]) -> dict:
    differences = (prediction[:, 1:] - prediction[:, :-1]).abs().mean(dim=(0, 2))
    indices = [value - 1 for value in boundaries if 0 < value < prediction.shape[1]]
    mask = torch.ones_like(differences, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices)] = False
        at_boundary = float(differences[torch.tensor(indices)].mean())
    else:
        at_boundary = None
    elsewhere = float(differences[mask].mean()) if mask.any() else None
    return {
        "mean_abs_jump_at_nonoverlap_boundaries": at_boundary,
        "mean_abs_jump_elsewhere": elsewhere,
        "ratio": at_boundary / elsewhere if at_boundary is not None and elsewhere else None,
    }


def prepare(output: Path) -> None:
    manifest_path = BASE / "manifest.json"
    base = json.loads(manifest_path.read_text())
    entries = [entry for entry in base["entries"] if entry["role"] == "speaker"]
    if len(entries) != 571:
        raise ValueError(f"Expected 571 forward entries, found {len(entries)}")
    manifest = {
        "schema_version": 1,
        "protocol": (
            "Frozen student forward VAL571; all opposite-role same-session GT; "
            "standalone head/center/tail 750; center standalone vs nonoverlap slice; "
            "full nonoverlap vs stride375 overlap-add"
        ),
        "base_manifest": [str(manifest_path.resolve()), sha(manifest_path)],
        "student_checkpoint": [str(STUDENT.resolve()), sha(STUDENT)],
        "implementation": [str(Path(__file__).resolve()), sha(Path(__file__).resolve())],
        "window": WINDOW,
        "overlap_stride": STRIDE,
        "tf32": False,
        "entries": entries,
    }
    if (output / "manifest.json").exists():
        raise FileExistsError("Use the existing manifest or a fresh output directory")
    write(output / "manifest.json", manifest)


def metric_means(rows, path):
    return {
        metric: float(np.mean([row[path[0]][path[1]][metric] for row in rows]))
        for metric in ("FRC", "exact_FRD", "FRDiv", "FRVar")
    }


def deltas(new, reference):
    return {
        metric: {
            "absolute": new[metric] - reference[metric],
            "relative_percent": 100.0 * (new[metric] - reference[metric]) / reference[metric],
        }
        for metric in reference
    }


def summarize(rows):
    positions = {name: metric_means(rows, ("positions", name)) for name in POSITIONS}
    center_stitched = metric_means(rows, ("center_comparison", "stitched_slice"))
    full = {name: metric_means(rows, ("full_sequence", name)) for name in METHODS}
    seam = {}
    for method in METHODS:
        valid = [row["seams"][method] for row in rows
                 if row["seams"][method]["ratio"] is not None]
        seam[method] = {
            key: float(np.mean([item[key] for item in valid])) if valid else None
            for key in ("mean_abs_jump_at_nonoverlap_boundaries",
                        "mean_abs_jump_elsewhere", "ratio")
        }
    return {
        "contexts": len(rows),
        "positions": positions,
        "position_deltas_vs_center": {
            name: deltas(positions[name], positions["center"])
            for name in ("head", "tail")
        },
        "center_comparison": {
            "standalone": positions["center"],
            "stitched_slice": center_stitched,
            "stitched_minus_standalone": deltas(center_stitched, positions["center"]),
        },
        "full_sequence": full,
        "overlap_deltas_vs_nonoverlap": {
            name: deltas(full[name], full["nonoverlap"])
            for name in ("overlap_uniform", "overlap_center_weighted")
        },
        "seams": seam,
    }


def aggregate(output: Path, manifest) -> None:
    rows = [json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
            for entry in manifest["entries"]]
    if [row["clip_id"] for row in rows] != [entry["clip_id"] for entry in manifest["entries"]]:
        raise ValueError("Row order does not match manifest")
    result = {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha(output / "manifest.json"),
        "groups": {
            "all_forward": summarize(rows),
            "cropped_only": summarize([row for row in rows if row["length"] > WINDOW]),
            "short_or_equal_control": summarize([row for row in rows if row["length"] <= WINDOW]),
        },
        "interpretation_boundary": [
            "Position tests use identical window lengths and source-aligned full-duration GT crops.",
            "Center stitched slices may cross one or more independently inferred nonoverlap chunks.",
            "Overlap-add changes inference context and fusion but does not provide recurrent state across windows.",
            "This diagnosis uses forward VAL only and does not retrain the model.",
        ],
        "rows": rows,
    }
    write(output / "comparison.json", result)
    print(json.dumps(result["groups"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.output)
        return
    manifest = json.loads((args.output / "manifest.json").read_text())
    if args.aggregate:
        aggregate(args.output, manifest)
        return
    if sha(Path(manifest["base_manifest"][0])) != manifest["base_manifest"][1]:
        raise ValueError("Base manifest changed")
    if sha(Path(manifest["student_checkpoint"][0])) != manifest["student_checkpoint"][1]:
        raise ValueError("Student checkpoint changed")
    if sha(Path(manifest["implementation"][0])) != manifest["implementation"][1]:
        raise ValueError("Diagnostic implementation changed after manifest freeze")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(device)
    mean = torch.from_numpy(np.load(ROOT / "mam_reactor/external/FaceVerse/mean_face.npy").astype("float32")).reshape(58)
    std = torch.from_numpy(np.load(ROOT / "mam_reactor/external/FaceVerse/std_face.npy").astype("float32")).reshape(58).clamp_min(1e-8)
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
                raise ValueError("Base array clip mismatch")
            target = stored["targets"].float().contiguous()
            nonoverlap = stored["predictions"]["student"].float().contiguous()
            length = entry["length"]
            if target.shape[1] != length or nonoverlap.shape != (10, length, 25):
                raise ValueError("Base tensor shape mismatch")
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
            position_scores = {}
            for name in POSITIONS:
                start = starts_by_position[name]
                position_scores[name] = score(
                    standalone[name], target[:, start:start + valid]
                )
            center_start = starts_by_position["center"]
            stitched_center = nonoverlap[:, center_start:center_start + valid]
            stitched_center_score = score(
                stitched_center, target[:, center_start:center_start + valid]
            )

            overlap_starts = window_starts(length)
            overlap_valid = [min(WINDOW, length - start) for start in overlap_starts]
            overlap_windows = generate_windows(
                model, streams, overlap_starts, overlap_valid, device
            )
            uniform = official_prediction(overlap_add(
                overlap_windows, overlap_starts, length, "uniform"
            ))
            centered = official_prediction(overlap_add(
                overlap_windows, overlap_starts, length, "center"
            ))
            full_predictions = {
                "nonoverlap": nonoverlap,
                "overlap_uniform": uniform,
                "overlap_center_weighted": centered,
            }
            # The base score is copied to preserve an exact link to the frozen run.
            base_row = json.loads((BASE / "rows" / f"{entry['index']:04d}.json").read_text())
            full_scores = {
                "nonoverlap": base_row["models"]["student"],
                "overlap_uniform": score(uniform, target),
                "overlap_center_weighted": score(centered, target),
            }
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
                "center_comparison": {
                    "standalone": position_scores["center"],
                    "stitched_slice": stitched_center_score,
                },
                "full_sequence": full_scores,
                "seams": {
                    name: boundary_jump(prediction, boundaries)
                    for name, prediction in full_predictions.items()
                },
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
