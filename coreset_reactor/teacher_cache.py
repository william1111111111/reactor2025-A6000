"""Cache deterministic fixed-view outputs from independent teachers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_data import ConditionalReactionDataset
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_teacher(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    config = ConditionalREGNNConfig(**checkpoint["model_config"])
    model = ConditionalREGNN(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--view-epoch", type=int, default=0)
    parser.add_argument("--crop-views", type=int, default=1)
    parser.add_argument(
        "--crop-view-mode", choices=("random", "stratified"), default="random"
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument('--session-allowlist', nargs='+',
                        help='Optional TRAIN eligibility filter, recorded for input reconstruction')
    args = parser.parse_args()
    if args.crop_views < 1:
        raise ValueError("crop-views must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    models, configs = [], []
    for path in args.teacher_checkpoints:
        model, config = load_teacher(path.resolve(), device)
        models.append(model)
        configs.append(config)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Teacher model configs differ")
    dataset = ConditionalReactionDataset(
        str(args.data_root.resolve()), "train", clip_length=750,
        training=True, target_mode="paired", seed=1,
        session_allowlist=args.session_allowlist,
        crop_view_mode=args.crop_view_mode,
        crop_view_count=args.crop_views,
    )
    dataset.set_epoch(args.view_epoch)
    source_count = (
        len(dataset)
        if args.max_examples <= 0
        else min(args.max_examples, len(dataset))
    )
    schedule = []
    source_lengths = []
    for sample_index, record in enumerate(dataset.records[:source_count]):
        relative = record.with_suffix("")
        paths = (
            dataset.audio_dir / relative.with_suffix(".npy"),
            dataset.emotion_dir / record,
            dataset.coefficient_dir / relative.with_suffix(".npy"),
        )
        source_length = min(len(np.load(path, mmap_mode="r")) for path in paths)
        source_lengths.append(source_length)
        views = args.crop_views if source_length > dataset.clip_length else 1
        schedule.extend((sample_index, view_id) for view_id in range(views))
    count = len(schedule)
    loader = DataLoader(
        Subset(dataset, schedule), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    shard, shard_names, completed = [], [], 0
    with torch.inference_mode():
        for batch in loader:
            conditions = [batch[key].to(device, non_blocking=True) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm", "length",
            )]
            predictions = torch.stack([
                model(*conditions)["prediction"].cpu().to(torch.float16)
                for model in models
            ], dim=1)
            if not torch.isfinite(predictions).all():
                raise ValueError("Teacher cache contains NaN or Inf")
            for row in range(predictions.shape[0]):
                sample_index, expected_view_id = schedule[completed]
                shard.append({
                    "sample_index": sample_index,
                    "sample_id": batch["clip_id"][row],
                    "view_id": int(batch["crop_view_id"][row]),
                    "view_epoch": args.view_epoch,
                    "start": int(batch["start"][row]),
                    "length": int(batch["length"][row]),
                    "generated_length": int(batch["length"][row]),
                    "teacher_seed": None,
                    "teacher_predictions": predictions[row],
                })
                if int(batch["crop_view_id"][row]) != expected_view_id:
                    raise ValueError("DataLoader crop view order changed")
                completed += 1
                if len(shard) == args.shard_size or completed == count:
                    name = f"shard_{len(shard_names):05d}.pt"
                    atomic_torch_save(shard, args.output_dir / name)
                    shard_names.append(name)
                    shard = []
            print(f"[teacher-cache] {completed}/{count}", flush=True)
    teacher_hashes = [sha256_file(path.resolve()) for path in args.teacher_checkpoints]
    manifest = {
        "schema_version": 1,
        "split": "train",
        "contexts": count,
        "source_contexts": source_count,
        "num_experts": len(models),
        "view_epoch": args.view_epoch,
        "source_seed": 1,
        "clip_length": 750,
        "teacher_checkpoint_hashes": teacher_hashes,
        "preprocessing_version": (
            "conditional-regnn-stratified-multiview-v2"
            if args.crop_view_mode == "stratified" and args.crop_views > 1
            else "conditional-regnn-fixed-view-v1"
        ),
        "teacher_generation": "deterministic eval/inference_mode",
        "crop_views": args.crop_views,
        "crop_view_mode": args.crop_view_mode,
        "crop_view_policy": (
            "one view for source_length<=750; evenly spaced head-to-tail views "
            "for longer sources"
            if args.crop_view_mode == "stratified"
            else "deterministic hashed random view IDs"
        ),
        "source_length_counts": {
            "short_or_equal": sum(length <= 750 for length in source_lengths),
            "long": sum(length > 750 for length in source_lengths),
        },
        "sampler_config_hash": hashlib.sha256(
            json.dumps({"view_epoch": args.view_epoch, "seed": 1,
                        "clip_length": 750, "crop_views": args.crop_views,
                        "crop_view_mode": args.crop_view_mode},
                       sort_keys=True).encode()
        ).hexdigest(),
        "model_config": configs[0].__dict__,
        "shards": shard_names,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if args.session_allowlist is not None:
        manifest['session_allowlist'] = sorted(set(args.session_allowlist))
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["cache_manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
