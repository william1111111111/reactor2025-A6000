"""Fast full-VAL all-session FRC/diversity evaluation for a shared student."""
from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.evaluate import _deterministic_postprocess, official_prediction
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.oracle_select import Candidate, SharedQualityPool, _quality_terms

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-mam-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--exact-frd", action="store_true")
    parser.add_argument("--metric-workers", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--target-cache", type=Path,
                        help="Persistent deterministic post-processed GT cache shared across students")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    from coreset_reactor.teacher_cache import sha256_file
    from framework.metrics import compute_FRVar, compute_s_mse
    from framework.metrics.FRC import _func
    from framework.modules.post_processor import Processor

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    config = state["student_config"]
    model = ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=ConditionalREGNNConfig(**config["backbone"]),
        num_experts=config["num_experts"],
        expert_embedding_dim=config["expert_embedding_dim"],
    )).to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    data = PairedReactionDataset(
        args.data_root, "val", 750, crop_mode="center",
        normalization_dir=args.asset_mam_root / "external/FaceVerse",
    )
    facial = args.data_root / "val/facial-attributes"
    pools = {}
    for path in sorted((facial / "listener").glob("*/*.npy")):
        pools.setdefault(path.parent.name, []).append(path)
    processor = None
    processor_path = args.asset_mam_root / 'pretrained_models/post_processor/checkpoint.pth'
    cache_protocol = {'postprocessor_sha256': sha256_file(processor_path),
                      'eval_seed': args.eval_seed, 'torch_version': torch.__version__,
                      'device_type': device.type, 'num_predictions': config['num_experts'],
                      'clip_len_test': 1000, 'version': 1,
                      'processor_source_sha256': sha256_file(MAM_ROOT/'framework/modules/post_processor.py'),
                      'deterministic_wrapper_sha256': sha256_file(ROOT/'coreset_reactor/evaluate.py')}
    if args.target_cache:
        args.target_cache.mkdir(parents=True, exist_ok=True)
    predictions, rows, latencies, frd_rows = [], [], [], []
    soft_predictions, cache_hits = [], 0
    quality_pool = (SharedQualityPool(
        args.metric_workers, config["num_experts"], 750,
        max_targets=max(map(len, pools.values())),
    ) if args.exact_frd else None)
    started = time.monotonic()
    with torch.inference_mode():
        for number in range(1, len(data) + 1):
            sample = data[number - 1]
            length = int(sample["source_lengths"])
            inputs = tuple(sample[key][None, :length].to(device) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm",
            ))
            torch.cuda.synchronize(device); tick = time.perf_counter()
            prediction = model.generate(
                *inputs, torch.tensor([length], device=device),
            )["prediction"][0].cpu()
            torch.cuda.synchronize(device); latencies.append(time.perf_counter() - tick)
            official = official_prediction(prediction)
            predictions.append(official)
            soft_predictions.append(prediction)
            paired = (facial / sample["clip_id"].replace(
                "speaker/", "listener/", 1)).with_suffix(".npy")
            paths = [paired] + [p for p in pools[sample["session_id"]] if p != paired]
            key = {**cache_protocol, 'clip_id': sample['clip_id'], 'length': length,
                   'gt_files': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]}
            fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            cache_file = args.target_cache / f'{fingerprint}.pt' if args.target_cache else None
            if cache_file and cache_file.exists():
                payload = torch.load(cache_file, map_location='cpu')
                if payload['fingerprint'] != fingerprint:
                    raise ValueError('GT cache fingerprint mismatch')
                target = payload['targets']
                cache_hits += 1
            else:
                if processor is None:
                    processor = Processor(
                        cfg_dir=str(args.asset_mam_root),
                        ckpt_dir=str(args.asset_mam_root / 'pretrained_models/post_processor'),
                        clip_len_test=1000, device=device, num_preds=config['num_experts'])
                raw = [torch.from_numpy(np.load(p, allow_pickle=False).astype(np.float32)) for p in paths]
                target = _deterministic_postprocess(processor, official, raw, device,
                                                   args.eval_seed, sample['clip_id']).cpu()
                if cache_file:
                    temporary = cache_file.with_suffix(f'.{os.getpid()}.tmp')
                    torch.save({'fingerprint': fingerprint, 'targets': target}, temporary)
                    os.replace(temporary, cache_file)
            if target.shape != (len(paths), length, 25) or not torch.isfinite(target).all():
                raise ValueError('Invalid post-processed GT tensor')
            values = candidate_frc(target.numpy(), official.numpy())
            if quality_pool is not None:
                candidates = [Candidate(
                    official[k], f"student_{k:02d}", "expert_student",
                    [("student", k)],
                ) for k in range(config["num_experts"])]
                _, frd = _quality_terms(
                    candidates, target, workers=args.metric_workers,
                    engine="rolling", shared_pool=quality_pool,
                )
                frd_rows.append(float(frd.sum()))
            if number == 1:
                reference = np.array([
                    float(_func(target, official[k:k + 1]))
                    for k in range(config["num_experts"])
                ])
                np.testing.assert_allclose(values, reference, rtol=1e-5, atol=1e-6)
            rows.append({"clip_id": sample["clip_id"], "gt_count": len(paths),
                         "frc": values.tolist(), 'target_cache_key': fingerprint,
                         'FRDiv': float(compute_s_mse([official])),
                         'soft_FRDiv': float(compute_s_mse([prediction])),
                         'exact_FRD': frd_rows[-1] if frd_rows else None})
            if number % 25 == 0 or number == len(data):
                args.output.with_suffix('.progress.json').write_text(json.dumps({
                    'completed': number, 'total': len(data), 'cache_hits': cache_hits,
                    'partial_FRC': float(np.mean([sum(r['frc']) for r in rows])),
                    'partial_FRDiv': float(np.mean([r['FRDiv'] for r in rows]))}))
                print(f"[student-fast-eval] {number}/{len(data)}", flush=True)
    means = np.mean([row["frc"] for row in rows], axis=0)
    result = {
        "metrics": {
            "contexts": len(data), "FRC": float(means.sum()),
            "FRC_per_expert": means.tolist(),
            "FRDiv": float(compute_s_mse(predictions)),
            "FRVar": float(compute_FRVar(predictions)),
            "soft_FRDiv": float(compute_s_mse(soft_predictions)),
            "exact_FRD": (float(np.mean(frd_rows))
                          if frd_rows else "not_evaluated"),
        },
        "speed": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "inference_latency_s_mean": float(np.mean(latencies)),
            "evaluation_seconds": time.monotonic() - started,
            "target_cache_hits": cache_hits,
            "inference_peak_allocated_bytes_including_postprocessor": torch.cuda.max_memory_allocated(device),
            "latency_caveat": "diagnostic FP32, no separate warmup; not a controlled speed benchmark",
        },
        "protocol": "full VAL571 speaker inputs; center750; all-session opposite-role GT; AU-rounded; vectorized scalar-parity FRC",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cache_protocol": cache_protocol,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "rows": rows,
    }
    torch.save({'clip_ids': [r['clip_id'] for r in rows],
                'soft_predictions': soft_predictions, 'official_predictions': predictions,
                'checkpoint_sha256': result['checkpoint_sha256']},
               args.output.with_suffix('.predictions.pt'))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if quality_pool is not None:
        quality_pool.close()
    print(json.dumps({"metrics": result["metrics"], "speed": result["speed"]}, indent=2))


if __name__ == "__main__":
    main()
