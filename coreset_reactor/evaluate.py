"""Held-out VAL pilot using existing official metric and GT-alignment code."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.dataset import full_reaction_descriptor
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.losses import descriptor_distance
from coreset_reactor.model import ParallelTCN
from coreset_reactor.paired_data import PairedReactionDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = PROJECT_ROOT / "mam_reactor"
if str(MAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MAM_ROOT))


def _quality_numpy_worker(item: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """Evaluate one context after IPC-safe NumPy serialization."""
    from framework.metrics.FRC import _func as frc_one
    from framework.metrics.FRD import _func as frd_one
    target_array, prediction_array = item
    target = torch.from_numpy(target_array)
    prediction = torch.from_numpy(prediction_array)
    return float(frc_one(target, prediction)), float(frd_one(target, prediction))


def exact_quality_metrics(predictions: list[torch.Tensor],
                          targets: list[torch.Tensor],
                          workers: int = 1) -> tuple[float, float]:
    """Compute the official FRC/FRD formulas without unbounded tensor IPC."""
    if workers < 1:
        raise ValueError("metric workers must be positive")
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be nonempty and aligned")
    if workers == 1:
        from framework.metrics.FRC import _func as frc_one
        from framework.metrics.FRD import _func as frd_one
        frc = np.mean([frc_one(target, pred)
                       for pred, target in zip(predictions, targets)])
        frd = np.mean([frd_one(target, pred)
                       for pred, target in zip(predictions, targets)])
        return float(frc), float(frd)
    # The official wrappers send Torch tensors through multiprocessing, which
    # opens one shared-memory file descriptor per tensor and exhausts a normal
    # 1024-FD process on full VAL. NumPy payloads preserve the exact private
    # formulas while using ordinary pipe serialization and bounded workers.
    payloads = ((target.numpy(), pred.numpy())
                for pred, target in zip(predictions, targets))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        scores = list(pool.imap(_quality_numpy_worker, payloads, chunksize=1))
    return (float(np.mean([score[0] for score in scores])),
            float(np.mean([score[1] for score in scores])))


def official_prediction(pred: torch.Tensor) -> torch.Tensor:
    """Use binary AUs for official metrics, without changing diagnostics."""
    if pred.ndim != 3 or pred.shape[-1] != 25:
        raise ValueError("prediction must be [K,T,25]")
    result = pred.clone()
    result[..., :15] = torch.round(result[..., :15])
    return result


def _stable_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _stable_seed(seed: int, name: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{purpose}|{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def _deterministic_postprocess(processor, pred: torch.Tensor,
                               raw_targets: list[torch.Tensor],
                               device: torch.device, seed: int,
                               clip_id: str) -> torch.Tensor:
    """Run the stochastic official target processor reproducibly.

    EmotionVAE samples a latent even in eval mode. Fork and restore every RNG
    used by the evaluation stack so this call is stable without perturbing a
    training process that invokes evaluate_model between optimization steps.
    """
    context_seed = _stable_seed(seed, clip_id, "target-postprocessor")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None
                        else torch.cuda.current_device()]
    try:
        random.seed(context_seed)
        np.random.seed(context_seed)
        with torch.random.fork_rng(devices=cuda_devices):
            cpu_generator = torch.Generator(device="cpu").manual_seed(context_seed)
            torch.set_rng_state(cpu_generator.get_state())
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(context_seed)
            return processor.forward([pred], [raw_targets])[0]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


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
    from framework.metrics import compute_FRVar, compute_s_mse
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
        source_length = int(sample["source_lengths"])
        inputs = (sample["speaker_audio"][None, :source_length].to(device),
                  sample["speaker_emotion"][None, :source_length].to(device),
                  sample["speaker_3dmm"][None, :source_length].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred = model(*inputs)[0].detach().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        predictions.append(official_prediction(pred))
        session = sample["session_id"]
        paired = facial / sample["clip_id"].replace("speaker/", "listener/", 1)
        paired = paired.with_suffix(".npy")
        alternatives = [p for p in pool[session] if p != paired]
        rng = _stable_rng(seed, sample["clip_id"])
        selected_gt = [paired] + (rng.sample(alternatives, 9) if len(alternatives) >= 9
                                  else rng.choices(alternatives or [paired], k=9))
        raw_targets = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                       for path in selected_gt]
        aligned = _deterministic_postprocess(
            processor, pred, raw_targets, device, seed, sample["clip_id"])
        if aligned.shape != pred.shape:
            raise ValueError(f"official processor alignment failed: {aligned.shape} vs {pred.shape}")
        targets.append(aligned)
        if session not in full_gt_cache:
            raw_descriptors = []
            for path in pool[session]:
                raw = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                raw_descriptors.append(full_reaction_descriptor(raw.to(device)))
            full_gt_cache[session] = scaler(torch.stack(raw_descriptors)).cpu()
        pred_desc = scaler(reaction_descriptor(pred.to(device))).cpu()
        gt_desc = full_gt_cache[session]
        distance = descriptor_distance(pred_desc[None], gt_desc[None])[0]
        context_metrics.append({"clip_id": sample["clip_id"],
                                "source_length": source_length,
                                "same_session_gt_count": len(pool[session]),
                                "gt_descriptor_coverage": float(distance.amin(0).mean()),
                                "prediction_validity": float(distance.amin(1).mean()),
                                "candidate_utilization": float(distance.argmin(0).unique().numel()
                                                               / model.num_predictions),
                                "cluster_coverage": _cluster_coverage(pred_desc, gt_desc)})
    frc, frd = exact_quality_metrics(predictions, targets, metric_workers)
    metrics = {"split": "val", "scope": "deterministic VAL pilot; AU-rounded predictions, official metric formulas and target processor, not full TEST",
               "contexts": len(predictions), "gt_policy": "paired + 9 deterministic same-session GTs",
               "target_alignment_rng": "sha256(target-postprocessor|eval_seed|clip_id)",
               "eval_seed": seed,
               "metric_aggregation": ("serial official per-context formulas" if metric_workers == 1
                                      else "IPC-safe multiprocessing of official per-context "
                                           f"formulas, workers={metric_workers}"),
               "FRC": frc,
               "exact_FRD": frd,
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
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "coreset_reactor" / "cache" / "train_descriptors_v2.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--metric-workers", type=int, default=1)
    args = parser.parse_args()
    from coreset_reactor.dataset import TrainSetData
    cache = TrainSetData(args.data_root, args.cache)
    state = torch.load(args.checkpoint, map_location="cpu")
    model = ParallelTCN(**state["model_config"])
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    model.to(device)
    metrics, speed = evaluate_model(model, args.data_root, cache.scaler,
                                    device, args.max_examples, args.eval_seed,
                                    args.metric_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "speed": speed}, indent=2))


if __name__ == "__main__":
    main()
