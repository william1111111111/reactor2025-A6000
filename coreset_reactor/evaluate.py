"""Held-out VAL pilot using existing official metric and GT-alignment code."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.dataset import fixed_reaction
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.losses import descriptor_distance
from coreset_reactor.model import ParallelTCN
from coreset_reactor.paired_data import PairedReactionDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = PROJECT_ROOT / "mam_reactor"
if str(MAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MAM_ROOT))


def _stable_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _cluster_coverage(pred: torch.Tensor, gt: torch.Tensor,
                      count: int = 12) -> float:
    """Diagnostic: fraction of GT medoid neighborhoods reached by predictions."""
    n = gt.shape[0]
    count = min(count, n)
    gg = torch.cdist(gt[None], gt[None])[0] / gt.shape[1] ** .5
    medoids = [int(gg.sum(1).argmin())]
    nearest = gg[medoids[0]].clone()
    for _ in range(1, count):
        next_id = int(nearest.argmax())
        medoids.append(next_id)
        nearest = torch.minimum(nearest, gg[next_id])
    cluster = gg[:, medoids].argmin(1)
    distance = descriptor_distance(pred[None], gt[None])[0]
    covered = []
    for cluster_id, medoid in enumerate(medoids):
        members = torch.where(cluster == cluster_id)[0]
        radius = gg[medoid, members].quantile(.75).clamp_min(.2)
        covered.append(bool(distance[:, members].amin() <= radius))
    return sum(covered) / len(covered)


@torch.no_grad()
def evaluate_model(model: ParallelTCN, data_root: Path,
                   scaler: torch.nn.Module, device: torch.device,
                   max_examples: int, seed: int = 1234,
                   metric_workers: int = 1) -> tuple[dict, dict]:
    """Call official formulas on paired+9 same-session aligned VAL GTs.

    This is a deterministic VAL pilot, *not* the official full TEST protocol.
    No GT enters model.forward; GT is read only after inference.
    """
    from framework.metrics import compute_FRC, compute_FRVar, compute_s_mse
    from framework.metrics.FRD import compute_FRD
    from framework.modules.post_processor import Processor

    data = PairedReactionDataset(
        data_root, "val", 750, crop_mode="center",
        normalization_dir=MAM_ROOT / "external" / "FaceVerse")
    selected = sorted(random.Random(seed).sample(range(len(data)),
                                                  min(max_examples, len(data))))
    facial = data_root / "val" / "facial-attributes"
    pool = {}
    for path in sorted((facial / "listener").glob("*/*.npy")):
        pool.setdefault(path.parent.name, []).append(path)
    processor = Processor(cfg_dir=str(MAM_ROOT),
                          ckpt_dir=str(MAM_ROOT / "pretrained_models" / "post_processor"),
                          clip_len_test=1000, device=device, num_preds=model.num_predictions)
    predictions, targets, context_metrics, latencies = [], [], [], []
    full_gt_cache: dict[str, torch.Tensor] = {}
    model.eval()
    scaler = scaler.to(device)
    for index in selected:
        sample = data[index]
        inputs = (sample["speaker_audio"][None].to(device),
                  sample["speaker_emotion"][None].to(device),
                  sample["speaker_3dmm"][None].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred = model(*inputs)[0].detach().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        predictions.append(pred)
        session = sample["session_id"]
        paired = facial / sample["clip_id"].replace("speaker/", "listener/", 1)
        paired = paired.with_suffix(".npy")
        alternatives = [p for p in pool[session] if p != paired]
        rng = _stable_rng(seed, sample["clip_id"])
        selected_gt = [paired] + (rng.sample(alternatives, 9) if len(alternatives) >= 9
                                  else rng.choices(alternatives or [paired], k=9))
        raw_targets = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                       for path in selected_gt]
        aligned = processor.forward([pred], [raw_targets])[0]
        if aligned.shape != pred.shape:
            raise ValueError(f"official processor alignment failed: {aligned.shape} vs {pred.shape}")
        targets.append(aligned)
        if session not in full_gt_cache:
            whole = torch.stack([fixed_reaction(torch.from_numpy(
                np.load(path, allow_pickle=False).astype(np.float32)), 750)
                                 for path in pool[session]])
            full_gt_cache[session] = scaler(reaction_descriptor(whole.to(device))).cpu()
        pred_desc = scaler(reaction_descriptor(pred.to(device))).cpu()
        gt_desc = full_gt_cache[session]
        distance = descriptor_distance(pred_desc[None], gt_desc[None])[0]
        context_metrics.append({"clip_id": sample["clip_id"],
                                "same_session_gt_count": len(pool[session]),
                                "gt_descriptor_coverage": float(distance.amin(0).mean()),
                                "prediction_validity": float(distance.amin(1).mean()),
                                "candidate_utilization": float(distance.argmin(0).unique().numel()
                                                               / model.num_predictions),
                                "cluster_coverage": _cluster_coverage(pred_desc, gt_desc)})
    metrics = {"split": "val", "scope": "deterministic VAL pilot; official metric formulas and target processor, not full TEST",
               "contexts": len(predictions), "gt_policy": "paired + 9 deterministic same-session GTs",
               "FRC": float(compute_FRC(predictions, targets, p=metric_workers)),
               "exact_FRD": float(compute_FRD(predictions, targets, p=metric_workers)),
               "FRDiv": float(compute_s_mse(predictions)),
               "FRVar": float(compute_FRVar(predictions)),
               "gt_descriptor_coverage": float(np.mean([r["gt_descriptor_coverage"] for r in context_metrics])),
               "prediction_validity": float(np.mean([r["prediction_validity"] for r in context_metrics])),
               "candidate_utilization": float(np.mean([r["candidate_utilization"] for r in context_metrics])),
               "cluster_coverage": float(np.mean([r["cluster_coverage"] for r in context_metrics])),
               "per_context": context_metrics}
    speed = {"parameter_count": sum(p.numel() for p in model.parameters()),
             "inference_latency_s_mean": float(np.mean(latencies)),
             "inference_latency_s_median": float(np.median(latencies)),
             "inference_set_fps": float(750 / np.mean(latencies)),
             "inference_output_fps": float(750 * model.num_predictions / np.mean(latencies))}
    return metrics, speed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "coreset_reactor" / "cache" / "train_descriptors_v1.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    from coreset_reactor.dataset import TrainSetData
    cache = TrainSetData(args.data_root, args.cache)
    state = torch.load(args.checkpoint, map_location="cpu")
    model = ParallelTCN(**state["model_config"])
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    model.to(device)
    metrics, speed = evaluate_model(model, args.data_root, cache.scaler,
                                    device, args.max_examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "speed": speed}, indent=2))


if __name__ == "__main__":
    main()
