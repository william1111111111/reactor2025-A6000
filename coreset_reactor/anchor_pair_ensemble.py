"""Evaluate ten Mam-Reactor ConditionalREGNN anchors as one reaction set."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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
from coreset_reactor.oracle_select import Candidate, SharedQualityPool, _quality_terms
from coreset_reactor.paired_data import PairedReactionDataset


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MAM_ROOT = ROOT / "mam_reactor"
if str(LOCAL_MAM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_MAM_ROOT))

from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig


MODEL_COUNT = 10


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _tlcc_numpy_worker(payload: tuple[np.ndarray, np.ndarray]) -> float:
    """Call the official TLCC context function without Torch tensor IPC."""
    from framework.metrics.TLCC import _func

    prediction, speaker = payload
    return float(_func(torch.from_numpy(prediction), torch.from_numpy(speaker)))


def ipc_safe_tlcc(predictions: list[torch.Tensor],
                  speakers: list[torch.Tensor], workers: int) -> float:
    """Preserve official TLCC while avoiding one FD per Torch tensor."""
    if len(predictions) != len(speakers) or not predictions:
        raise ValueError("TLCC inputs must be nonempty and aligned")
    payloads = ((prediction.numpy(), speaker.numpy())
                for prediction, speaker in zip(predictions, speakers))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        values = list(pool.imap(_tlcc_numpy_worker, payloads, chunksize=1))
    return float(np.mean(values))


def load_models(output_root: Path, epoch: int,
                device: torch.device) -> tuple[list[ConditionalREGNN], list[dict]]:
    models, checkpoints = [], []
    for rank in range(MODEL_COUNT):
        path = (output_root / f"model_{rank:02d}" /
                f"conditional-epoch{epoch:04d}-seed1.pth")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        training = checkpoint["training_config"]
        if training.get("fixed_pair_rank") != rank:
            raise ValueError(f"fixed-pair rank mismatch in {path}")
        model = ConditionalREGNN(
            ConditionalREGNNConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().to(device)
        models.append(model)
        checkpoints.append(checkpoint)
    matched_fields = (
        "epochs", "batch_size", "clip_length", "lr", "weight_decay",
        "ccc_weight", "velocity_weight", "seed", "precision",
        "fixed_pair_count", "fixed_pair_seed",
    )
    for field in matched_fields:
        if len({checkpoint["training_config"].get(field)
                for checkpoint in checkpoints}) != 1:
            raise ValueError(f"ensemble training mismatch: {field}")
    return models, checkpoints


@torch.no_grad()
def ensemble_forward(models: list[ConditionalREGNN],
                     inputs: tuple[torch.Tensor, ...],
                     lengths: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return torch.cat([
            model(*inputs, lengths)["prediction"] for model in models
        ], dim=0).cpu()
    streams = [torch.cuda.Stream(device=device) for _ in models]
    outputs = [None] * len(models)
    for index, (model, stream) in enumerate(zip(models, streams)):
        with torch.cuda.stream(stream):
            outputs[index] = model(*inputs, lengths)["prediction"]
    current = torch.cuda.current_stream(device)
    for stream in streams:
        current.wait_stream(stream)
    return torch.cat(outputs, dim=0).cpu()


def evaluate(args: argparse.Namespace) -> dict:
    if str(args.asset_mam_root) not in sys.path:
        sys.path.insert(0, str(args.asset_mam_root))
    from framework.metrics import compute_FRVar, compute_s_mse
    from framework.modules.post_processor import Processor

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    models, checkpoints = load_models(args.output_root, args.epoch, device)
    cache = TrainSetData(args.data_root, args.cache)
    scaler = cache.scaler.to(device)
    data = PairedReactionDataset(
        args.data_root, "val", 750, crop_mode="center",
        normalization_dir=args.asset_mam_root / "external" / "FaceVerse")
    selected = sorted(random.Random(args.eval_seed).sample(
        range(len(data)), min(args.max_examples, len(data))))
    facial = args.data_root / "val" / "facial-attributes"
    target_pool = {}
    for path in sorted((facial / "listener").glob("*/*.npy")):
        target_pool.setdefault(path.parent.name, []).append(path)
    processor = Processor(
        cfg_dir=str(args.asset_mam_root),
        ckpt_dir=str(args.asset_mam_root / "pretrained_models" / "post_processor"),
        clip_len_test=1000, device=device, num_preds=MODEL_COUNT)
    quality_pool = SharedQualityPool(args.metric_workers, MODEL_COUNT, 750)
    predictions, speakers = [], []
    frc_rows, frd_rows, diagnostics, latencies = [], [], [], []
    full_gt_cache = {}
    try:
        for number, data_index in enumerate(selected, 1):
            sample = data[data_index]
            length = int(sample["source_lengths"])
            lengths = torch.tensor([length], device=device)
            inputs = (sample["speaker_audio"][None, :length].to(device),
                      sample["speaker_emotion"][None, :length].to(device),
                      sample["speaker_3dmm"][None, :length].to(device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            prediction = ensemble_forward(models, inputs, lengths, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - tick)
            official = official_prediction(prediction)
            predictions.append(official)
            speakers.append(sample["speaker_emotion"][:length].cpu())

            session = sample["session_id"]
            paired = facial / sample["clip_id"].replace("speaker/", "listener/", 1)
            paired = paired.with_suffix(".npy")
            alternatives = [path for path in target_pool[session] if path != paired]
            rng = _stable_rng(args.eval_seed, sample["clip_id"])
            target_paths = [paired] + (rng.sample(alternatives, 9)
                                       if len(alternatives) >= 9
                                       else rng.choices(alternatives or [paired], k=9))
            raw_targets = [torch.from_numpy(
                np.load(path, allow_pickle=False).astype(np.float32))
                for path in target_paths]
            targets = _deterministic_postprocess(
                processor, official, raw_targets, device,
                args.eval_seed, sample["clip_id"])
            candidates = [Candidate(
                official[rank], f"anchor_{rank:02d}", "conditional_anchor",
                [(f"anchor_{rank:02d}", 0)]) for rank in range(MODEL_COUNT)]
            frc, frd = _quality_terms(
                candidates, targets, workers=args.metric_workers,
                engine="rolling", shared_pool=quality_pool)
            frc_rows.append(float(frc.sum()))
            frd_rows.append(float(frd.sum()))

            if session not in full_gt_cache:
                descriptors = []
                for path in target_pool[session]:
                    raw = torch.from_numpy(
                        np.load(path, allow_pickle=False).astype(np.float32))
                    descriptors.append(full_reaction_descriptor(raw.to(device)))
                full_gt_cache[session] = scaler(torch.stack(descriptors)).cpu()
            pred_desc = scaler(reaction_descriptor(prediction.to(device))).cpu()
            gt_desc = full_gt_cache[session]
            distance = torch.cdist(pred_desc[None], gt_desc[None])[0] / (
                pred_desc.shape[1] ** .5)
            diagnostics.append({
                "gt_descriptor_coverage": float(distance.amin(0).mean()),
                "prediction_validity": float(distance.amin(1).mean()),
                "candidate_utilization": float(
                    distance.argmin(0).unique().numel() / MODEL_COUNT),
                "cluster_coverage": _cluster_coverage(pred_desc, gt_desc),
            })
            if number == 1 or number % 16 == 0 or number == len(selected):
                print(f"[anchor-ensemble-eval] {number}/{len(selected)}", flush=True)
    finally:
        quality_pool.close()

    metrics = {
        "contexts": len(predictions), "eval_seed": args.eval_seed,
        "FRC": float(np.mean(frc_rows)),
        "exact_FRD": float(np.mean(frd_rows)),
        "FRDiv": float(compute_s_mse(predictions)),
        "FRVar": float(compute_FRVar(predictions)),
        "TLCC": ipc_safe_tlcc(predictions, speakers, args.metric_workers),
        "gt_descriptor_coverage": float(np.mean([
            row["gt_descriptor_coverage"] for row in diagnostics])),
        "prediction_validity": float(np.mean([
            row["prediction_validity"] for row in diagnostics])),
        "candidate_utilization": float(np.mean([
            row["candidate_utilization"] for row in diagnostics])),
        "cluster_coverage": float(np.mean([
            row["cluster_coverage"] for row in diagnostics])),
    }
    parameters = sum(parameter.numel() for parameter in models[0].parameters())
    batches = 3320 // int(checkpoints[0]["training_config"]["batch_size"])
    speed = {
        "single_model_parameters": parameters,
        "ensemble_parameters": parameters * MODEL_COUNT,
        "training_model_steps": batches * args.epoch * MODEL_COUNT,
        "inference_latency_s_mean": float(np.mean(latencies)),
        "metric_workers": args.metric_workers,
    }
    result = {
        "metrics": metrics, "speed": speed,
        "protocol": {
            "architecture": "Mam-Reactor ConditionalREGNN 25D anchor",
            "rank_0": "true same-basename paired listener",
            "ranks_1_to_9": "fixed distinct same-session listeners",
            "target_processing": (
                "ReactionDataset diffusion crop offset and speaker-emotion "
                "tail fill"
            ),
            "models": MODEL_COUNT, "outputs_per_model": 1,
            "evaluation_scope": (
                "deterministic VAL; paired plus nine seeded same-session "
                "targets; not official TEST"
            ),
        },
    }
    _save_json(args.output, result)
    baseline = {"FRC": .8101113703697341, "exact_FRD": 82.48013452247962,
                "FRDiv": .013611357211381615, "FRVar": .03919985358017756,
                "gt_descriptor_coverage": 1.095160}
    delta = {key: metrics[key] - value for key, value in baseline.items()}
    config = checkpoints[0]["training_config"]
    lines = [
        "# Mam anchor ensemble: true pair plus nine fixed targets", "",
        "Ten independent ConditionalREGNN anchors share architecture, initialization seed, source/crop schedule and optimizer settings. Rank 0 uses the true same-basename paired listener; ranks 1-9 use deterministic distinct same-session listeners. Once selected, every target follows the original diffusion `ReactionDataset` crop offset and short-target speaker-emotion tail-fill path.", "",
        f"- Training: `{args.epoch}` epochs/model, batch `{config['batch_size']}`, seed `{config['seed']}`, `{speed['training_model_steps']}` total model-steps",
        f"- Optimizer: AdamW lr `{config['lr']}`, weight decay `{config['weight_decay']}`, cosine decay; BF16 `{config['precision'] == 'bf16'}`",
        f"- Loss: MSE + `{config['ccc_weight']}` × (1-CCC) + `{config['velocity_weight']}` × velocity MSE",
        f"- Parameters: `{parameters}` per model, `{parameters * MODEL_COUNT}` total",
        f"- Evaluation: deterministic VAL `{metrics['contexts']}`, exact rolling FRD, `{args.metric_workers}` metric workers", "",
        "| method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC | GT coverage ↓ |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| frozen M1 | {baseline['FRC']:.6f} | {baseline['exact_FRD']:.6f} | {baseline['FRDiv']:.6f} | {baseline['FRVar']:.6f} | n/a | {baseline['gt_descriptor_coverage']:.6f} |",
        f"| paired + nine fixed anchors | {metrics['FRC']:.6f} | {metrics['exact_FRD']:.6f} | {metrics['FRDiv']:.6f} | {metrics['FRVar']:.6f} | {metrics['TLCC']:.6f} | {metrics['gt_descriptor_coverage']:.6f} |", "",
        f"Deltas versus M1: ΔFRC `{delta['FRC']:+.6f}`, Δexact-FRD `{delta['exact_FRD']:+.6f}`, ΔFRDiv `{delta['FRDiv']:+.6f}`, ΔFRVar `{delta['FRVar']:+.6f}`, ΔGT-coverage `{delta['gt_descriptor_coverage']:+.6f}`.", "",
        f"Additional diagnostics: prediction validity `{metrics['prediction_validity']:.6f}`, candidate utilization `{metrics['candidate_utilization']:.6f}`, cluster coverage `{metrics['cluster_coverage']:.6f}`.", "",
        "TLCC uses the repository's official implementation, which returns after its first prediction; it is reproduced for completeness but is not a ten-output aggregate.", "",
        "This is not compute-matched to M1: ten full ConditionalREGNN backbones are trained and retained.", "",
        "Verdict: fixed one-to-one target specialization produces substantially more diverse reactions, but it is not quality-safe. FRDiv rises strongly while exact FRD and descriptor coverage become materially worse.", "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--asset-mam-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--max-examples", type=int, default=571)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--metric-workers", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
