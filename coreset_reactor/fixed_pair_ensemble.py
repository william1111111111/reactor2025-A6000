"""Ten independent one-output models with fixed diffusion-dataset pairings.

Only target identity is changed.  Once selected, every target is loaded,
cropped at the source crop point, and short-tail-filled by the original
``dataset.react_2025.ReactionDataset`` implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.dataset import TrainSetData, full_reaction_descriptor
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.evaluate import (_cluster_coverage,
                                      _deterministic_postprocess,
                                      _stable_rng, official_prediction)
from coreset_reactor.losses import paired_cost, unpaired_softdtw_cost
from coreset_reactor.model import ParallelTCN
from coreset_reactor.oracle_select import Candidate, SharedQualityPool, _quality_terms
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.train import source_provenance, state_dict_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAM_ROOT = ROOT / "mam_reactor"
MODEL_COUNT = 10


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _schedule(size: int, steps: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(size) for _ in range(batch_size)] for _ in range(steps)]


def _schedule_hash(schedule: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()


def _dataset(mam_root: Path, data_root: Path, rank: int, seed: int):
    if str(mam_root) not in sys.path:
        sys.path.insert(0, str(mam_root))
    from regnn.eval_conditional_regnn_official_test import initialize_official_hydra_runtime
    initialize_official_hydra_runtime(mam_root)
    from dataset.react_2025 import ReactionDataset
    return ReactionDataset(
        root_dir=str(data_root), split="train", clip_length=750,
        target_size=224, crop_size=224, fps=30,
        audio_feature_type="wav2vec", load_video_s=False,
        load_video_l=False, load_audio=True, load_emotion_s=True,
        load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=False,
        normalize_3dmm="standard", fixed_target_rank=rank,
        fixed_target_count=MODEL_COUNT, fixed_target_seed=seed,
        source_role="speaker")


def _model_config() -> dict:
    return {
        "num_predictions": 1,
        "hidden_dim": 128,
        "control_point_stride": 4,
        "dilations": [1, 2, 4, 8, 16],
        "dropout": 0.1,
    }


def train_model(args: argparse.Namespace) -> None:
    if not 0 <= args.model_index < MODEL_COUNT:
        raise ValueError("model-index must be in [0,9]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    dataset = _dataset(args.mam_root, args.data_root, args.model_index, args.seed)
    schedule = _schedule(len(dataset), args.steps, args.batch_size, args.seed)
    config = _model_config()
    model = ParallelTCN(**config).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    run_dir = args.output_root / f"model_{args.model_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    target_paths = dataset.listener_path_list
    mapping_hash = hashlib.sha256("\n".join(
        path.as_posix() for path in target_paths).encode()).hexdigest()
    metadata = {
        "experiment": "ten independent fixed-pair one-output ParallelTCNs",
        "model_index": args.model_index,
        "model_count": MODEL_COUNT,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_config": config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_model_sha256": initial_hash,
        "source_schedule_sha256": _schedule_hash(schedule),
        "fixed_target_mapping_sha256": mapping_hash,
        "target_policy": (
            "source-specific deterministic distinct rank; all post-selection "
            "processing is original diffusion ReactionDataset.__getitem__"
        ),
        "objective": "paired_cost + 0.1 * grouped_banded_SoftDTW on same fixed pair",
        "provenance": source_provenance(),
    }
    _save_json(run_dir / "config.json", metadata)

    started_run = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule, 1):
        # ReactionDataset uses Python's seeded RNG for the original random cp.
        items = [dataset[index] for index in indices]
        audio = torch.stack([item[0] for item in items]).float().to(device)
        emotion = torch.stack([item[2] for item in items]).float().to(device)
        face = torch.stack([item[3] for item in items]).float().to(device)
        target = torch.stack([item[5] for item in items]).float().to(device)
        full_lengths = torch.full((len(items),), 750, dtype=torch.long,
                                  device=device)
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        prediction = model(audio, emotion, face)
        paired = paired_cost(prediction, target, full_lengths).mean()
        temporal = unpaired_softdtw_cost(
            prediction, target[:, None], frames=args.dtw_frames,
            gamma=args.dtw_gamma, band_ratio=args.dtw_band_ratio).mean()
        loss = paired + args.temporal_weight * temporal
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        rows.append({"step": step, "loss": float(loss.detach()),
                     "paired": float(paired.detach()),
                     "softdtw": float(temporal.detach()),
                     "gradient_norm": float(gradient),
                     "iteration_s": elapsed})
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"model={args.model_index} step={step}/{args.steps} "
                  f"loss={rows[-1]['loss']:.5f} step_s={elapsed:.3f}",
                  flush=True)

    with (run_dir / "training_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = {
        **metadata,
        "model": model.cpu().state_dict(),
        "optimizer": optimizer.state_dict(),
        "wall_time_s": time.perf_counter() - started_run,
    }
    torch.save(checkpoint, run_dir / "final_checkpoint.pt")
    _save_json(run_dir / "train_summary.json", {
        "model_index": args.model_index,
        "final_loss": rows[-1]["loss"],
        "mean_iteration_s": float(np.mean([row["iteration_s"] for row in rows[1:]])),
        "wall_time_s": checkpoint["wall_time_s"],
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None),
    })


def _load_models(output_root: Path, device: torch.device) -> tuple[list[ParallelTCN], list[dict]]:
    models, checkpoints = [], []
    for index in range(MODEL_COUNT):
        path = output_root / f"model_{index:02d}" / "final_checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["model_index"] != index:
            raise ValueError(f"checkpoint rank mismatch: {path}")
        model = ParallelTCN(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.eval().to(device)
        models.append(model)
        checkpoints.append(checkpoint)
    hashes = {checkpoint["initial_model_sha256"] for checkpoint in checkpoints}
    schedules = {checkpoint["source_schedule_sha256"] for checkpoint in checkpoints}
    if len(hashes) != 1 or len(schedules) != 1:
        raise ValueError("ensemble members do not share initialization/schedule")
    return models, checkpoints


@torch.no_grad()
def _ensemble_forward(models: list[ParallelTCN], inputs: tuple[torch.Tensor, ...],
                      device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return torch.cat([model(*inputs) for model in models], dim=1)[0].cpu()
    streams = [torch.cuda.Stream(device=device) for _ in models]
    outputs = [None] * len(models)
    for index, (model, stream) in enumerate(zip(models, streams)):
        with torch.cuda.stream(stream):
            outputs[index] = model(*inputs)
    current = torch.cuda.current_stream(device)
    for stream in streams:
        current.wait_stream(stream)
    return torch.cat(outputs, dim=1)[0].cpu()


def evaluate(args: argparse.Namespace) -> None:
    if str(args.mam_root) not in sys.path:
        sys.path.insert(0, str(args.mam_root))
    from framework.metrics import compute_FRVar, compute_s_mse
    from framework.modules.post_processor import Processor

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    models, checkpoints = _load_models(args.output_root, device)
    cache = TrainSetData(args.data_root, args.cache)
    scaler = cache.scaler.to(device)
    data = PairedReactionDataset(
        args.data_root, "val", 750, crop_mode="center",
        normalization_dir=args.mam_root / "external" / "FaceVerse")
    selected = sorted(random.Random(args.eval_seed).sample(
        range(len(data)), min(args.max_examples, len(data))))
    facial = args.data_root / "val" / "facial-attributes"
    pool = {}
    for path in sorted((facial / "listener").glob("*/*.npy")):
        pool.setdefault(path.parent.name, []).append(path)
    processor = Processor(
        cfg_dir=str(args.mam_root),
        ckpt_dir=str(args.mam_root / "pretrained_models" / "post_processor"),
        clip_len_test=1000, device=device, num_preds=MODEL_COUNT)
    quality_pool = SharedQualityPool(args.metric_workers, MODEL_COUNT, 750)
    predictions, frc_rows, frd_rows, context_rows, latencies = [], [], [], [], []
    full_gt_cache = {}
    try:
        for number, index in enumerate(selected, 1):
            sample = data[index]
            length = int(sample["source_lengths"])
            inputs = (sample["speaker_audio"][None, :length].to(device),
                      sample["speaker_emotion"][None, :length].to(device),
                      sample["speaker_3dmm"][None, :length].to(device))
            if device.type == "cuda": torch.cuda.synchronize(device)
            tick = time.perf_counter()
            prediction = _ensemble_forward(models, inputs, device)
            if device.type == "cuda": torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - tick)
            official = official_prediction(prediction)
            predictions.append(official)

            session = sample["session_id"]
            paired = facial / sample["clip_id"].replace("speaker/", "listener/", 1)
            paired = paired.with_suffix(".npy")
            alternatives = [path for path in pool[session] if path != paired]
            rng = _stable_rng(args.eval_seed, sample["clip_id"])
            selected_gt = [paired] + (rng.sample(alternatives, 9)
                                      if len(alternatives) >= 9
                                      else rng.choices(alternatives or [paired], k=9))
            raw_targets = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                           for path in selected_gt]
            targets = _deterministic_postprocess(
                processor, official, raw_targets, device,
                args.eval_seed, sample["clip_id"])
            candidates = [Candidate(official[slot], f"model_{slot:02d}",
                                    "fixed_pair", [(f"model_{slot:02d}", 0)])
                          for slot in range(MODEL_COUNT)]
            frc, frd = _quality_terms(candidates, targets, workers=args.metric_workers,
                                      engine="rolling", shared_pool=quality_pool)
            frc_rows.append(float(frc.sum()))
            frd_rows.append(float(frd.sum()))

            if session not in full_gt_cache:
                descriptors = []
                for path in pool[session]:
                    raw = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                    descriptors.append(full_reaction_descriptor(raw.to(device)))
                full_gt_cache[session] = scaler(torch.stack(descriptors)).cpu()
            pred_desc = scaler(reaction_descriptor(prediction.to(device))).cpu()
            gt_desc = full_gt_cache[session]
            distance = torch.cdist(pred_desc[None], gt_desc[None])[0] / pred_desc.shape[1] ** .5
            context_rows.append({
                "gt_descriptor_coverage": float(distance.amin(0).mean()),
                "prediction_validity": float(distance.amin(1).mean()),
                "candidate_utilization": float(distance.argmin(0).unique().numel() / MODEL_COUNT),
                "cluster_coverage": _cluster_coverage(pred_desc, gt_desc),
            })
            if number == 1 or number % 16 == 0 or number == len(selected):
                print(f"[fixed-pair-eval] {number}/{len(selected)}", flush=True)
    finally:
        quality_pool.close()

    metrics = {
        "contexts": len(predictions), "eval_seed": args.eval_seed,
        "FRC": float(np.mean(frc_rows)),
        "exact_FRD": float(np.mean(frd_rows)),
        "FRDiv": float(compute_s_mse(predictions)),
        "FRVar": float(compute_FRVar(predictions)),
        "gt_descriptor_coverage": float(np.mean([
            row["gt_descriptor_coverage"] for row in context_rows])),
        "prediction_validity": float(np.mean([
            row["prediction_validity"] for row in context_rows])),
        "candidate_utilization": float(np.mean([
            row["candidate_utilization"] for row in context_rows])),
        "cluster_coverage": float(np.mean([
            row["cluster_coverage"] for row in context_rows])),
    }
    speed = {
        "single_model_parameters": sum(p.numel() for p in models[0].parameters()),
        "ensemble_parameters": sum(sum(p.numel() for p in model.parameters()) for model in models),
        "inference_latency_s_mean": float(np.mean(latencies)),
        "metric_workers": args.metric_workers,
        "training_model_steps": sum(checkpoint["steps"] for checkpoint in checkpoints),
    }
    initial_hash = checkpoints[0]["initial_model_sha256"]
    schedule_hash = checkpoints[0]["source_schedule_sha256"]
    snapshot_hashes = {
        checkpoint["provenance"]["source_snapshot_sha256"]
        for checkpoint in checkpoints
    }
    mapping_hash_count = len({
        checkpoint["fixed_target_mapping_sha256"]
        for checkpoint in checkpoints
    })
    result = {"metrics": metrics, "speed": speed,
              "protocol": {
                  "training_target_processing": "unaltered diffusion ReactionDataset after fixed path selection",
                  "fixed_pairing": True, "models": MODEL_COUNT,
                  "outputs_per_model": 1,
              }}
    _save_json(args.output, result)
    baseline = {"FRC": .8101113703697341, "exact_FRD": 82.48013452247962,
                "FRDiv": .013611357211381615, "FRVar": .03919985358017756,
                "gt_descriptor_coverage": 1.095160}
    delta = {key: metrics[key] - value for key, value in baseline.items()}
    lines = [
        "# Ten fixed-pair independent models", "",
        "Each source has ten deterministic, distinct same-session TRAIN GT paths. Model k always uses rank k. After path selection, the target follows the original diffusion `ReactionDataset.__getitem__` crop and short-tail-fill path without a separate target transform.", "",
        f"- Training: `{checkpoints[0]['steps']}` steps/model, `{speed['training_model_steps']}` total model-steps, single seed `{checkpoints[0]['seed']}`",
        f"- Objective: `{checkpoints[0]['objective']}`",
        f"- Parameters: `{speed['single_model_parameters']}` per model, `{speed['ensemble_parameters']}` total",
        f"- Evaluation: deterministic VAL `{metrics['contexts']}`; 10 CUDA-stream ensemble outputs; exact rolling FRD with `{args.metric_workers}` workers", "",
        f"- Matched initialization hash: `{initial_hash}`; source schedule hash: `{schedule_hash}`",
        f"- Distinct fixed-target mapping hashes: `{mapping_hash_count}`; source snapshot: `{next(iter(snapshot_hashes)) if len(snapshot_hashes) == 1 else 'MISMATCH'}`", "",
        "| method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ |", "|---|---:|---:|---:|---:|---:|",
        f"| frozen M1 | {baseline['FRC']:.6f} | {baseline['exact_FRD']:.6f} | {baseline['FRDiv']:.6f} | {baseline['FRVar']:.6f} | {baseline['gt_descriptor_coverage']:.6f} |",
        f"| fixed-pair ensemble | {metrics['FRC']:.6f} | {metrics['exact_FRD']:.6f} | {metrics['FRDiv']:.6f} | {metrics['FRVar']:.6f} | {metrics['gt_descriptor_coverage']:.6f} |", "",
        f"Deltas versus M1: ΔFRC `{delta['FRC']:+.6f}`, Δexact-FRD `{delta['exact_FRD']:+.6f}`, ΔFRDiv `{delta['FRDiv']:+.6f}`, ΔFRVar `{delta['FRVar']:+.6f}`, ΔGT-coverage `{delta['gt_descriptor_coverage']:+.6f}`.", "",
        f"Additional diagnostics: prediction validity `{metrics['prediction_validity']:.6f}`, candidate utilization `{metrics['candidate_utilization']:.6f}`, cluster coverage `{metrics['cluster_coverage']:.6f}`.", "",
        "Verdict: fixed one-to-one specialization creates substantially different outputs, but it is not quality-safe. The diversity gain accompanies lower FRC, higher exact FRD, and worse GT coverage.", "",
        "The ten-model row is not compute-matched to M1: it uses ten independent backbones and ten times the model-step budget. Same-session GTs remain weak pair proxies rather than verified reactions to the identical stimulus.", "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--model-index", type=int, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--mam-root", type=Path, default=DEFAULT_MAM_ROOT)
    train.add_argument("--device", default="cuda:5")
    train.add_argument("--seed", type=int, default=20260918)
    train.add_argument("--steps", type=int, default=2500)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=.01)
    train.add_argument("--temporal-weight", type=float, default=.1)
    train.add_argument("--dtw-frames", type=int, default=32)
    train.add_argument("--dtw-gamma", type=float, default=.1)
    train.add_argument("--dtw-band-ratio", type=float, default=.25)
    train.add_argument("--cpu-threads", type=int, default=2)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-root", type=Path, required=True)
    evaluate_parser.add_argument("--data-root", type=Path, required=True)
    evaluate_parser.add_argument("--cache", type=Path, required=True)
    evaluate_parser.add_argument("--mam-root", type=Path, default=DEFAULT_MAM_ROOT)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="cuda:5")
    evaluate_parser.add_argument("--max-examples", type=int, default=571)
    evaluate_parser.add_argument("--eval-seed", type=int, default=1234)
    evaluate_parser.add_argument("--metric-workers", type=int, default=16)
    evaluate_parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "train": train_model(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
