"""Cache fixed-view predictions from the trained 10-query REACT2024 teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_data import ConditionalReactionDataset  # noqa: E402
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig  # noqa: E402
from regnn.emotion_query_mamba import (  # noqa: E402
    EmotionQueryMambaConfig,
    EmotionQueryResidualMamba,
)


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


def load_teacher(checkpoint_path: Path, device: torch.device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    anchor_config = ConditionalREGNNConfig(**saved["anchor_model_config"])
    anchor = ConditionalREGNN(anchor_config)
    query_config = EmotionQueryMambaConfig.from_checkpoint_payload(
        dict(saved["emotion_query_model_config"])
    )
    model = EmotionQueryResidualMamba(anchor, query_config)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    if query_config.num_queries != 10:
        raise ValueError(f"Expected 10 teacher queries, got {query_config.num_queries}")
    return model, anchor_config, query_config, int(saved["epoch"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--view-epoch", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=3.0)
    parser.add_argument("--style-residual-scale", type=float, default=3.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {args.output_dir}")
    if args.batch_size < 1 or args.workers < 0 or args.shard_size < 1:
        raise ValueError("Invalid batch, worker, or shard size")
    if args.residual_scale < 0 or args.style_residual_scale < 0:
        raise ValueError("Residual scales must be nonnegative")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    teacher_path = args.teacher_checkpoint.resolve()
    teacher, backbone_config, query_config, teacher_epoch = load_teacher(
        teacher_path, device
    )
    dataset = ConditionalReactionDataset(
        str(args.data_root.resolve()), "train", clip_length=750,
        training=True, target_mode="paired", seed=1,
    )
    dataset.set_epoch(args.view_epoch)
    count = len(dataset) if args.max_examples <= 0 else min(args.max_examples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    shard: list[dict] = []
    shard_names: list[str] = []
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            conditions = [batch[key].to(device, non_blocking=True) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm", "length",
            )]
            output = teacher(
                *conditions, residual_scale=args.residual_scale,
                style_residual_scale=args.style_residual_scale,
            )["prediction"].float().cpu().to(torch.float16)
            if output.shape[1:] != (10, 750, 25) or not torch.isfinite(output).all():
                raise ValueError(f"Invalid teacher output: {tuple(output.shape)}")
            for row in range(output.shape[0]):
                shard.append({
                    "sample_index": completed,
                    "sample_id": batch["clip_id"][row],
                    "view_id": int(batch["crop_view_id"][row]),
                    "view_epoch": args.view_epoch,
                    "start": int(batch["start"][row]),
                    "length": int(batch["length"][row]),
                    "generated_length": int(batch["length"][row]),
                    "teacher_seed": None,
                    "teacher_predictions": output[row],
                })
                completed += 1
                if len(shard) == args.shard_size or completed == count:
                    name = f"shard_{len(shard_names):05d}.pt"
                    atomic_torch_save(shard, args.output_dir / name)
                    shard_names.append(name)
                    shard = []
            print(f"[react2024-teacher-cache] {completed}/{count}", flush=True)

    teacher_hash = sha256_file(teacher_path)
    manifest = {
        "schema_version": 1,
        "dataset": "REACT2024",
        "split": "train",
        "contexts": count,
        "num_experts": query_config.num_queries,
        "view_epoch": args.view_epoch,
        "source_seed": 1,
        "clip_length": 750,
        "teacher_checkpoint": str(teacher_path),
        "teacher_checkpoint_hashes": [teacher_hash],
        "teacher_epoch": teacher_epoch,
        "teacher_generation": (
            f"EmotionQueryResidualMamba forward; residual scale {args.residual_scale}; "
            f"style residual scale {args.style_residual_scale}; "
            "deterministic eval mode; raw float predictions (no AU rounding)"
        ),
        "preprocessing_version": "conditional-regnn-fixed-view-v1",
        "sampler_config_hash": hashlib.sha256(json.dumps({
            "view_epoch": args.view_epoch, "seed": 1, "clip_length": 750,
        }, sort_keys=True).encode()).hexdigest(),
        "model_config": asdict(backbone_config),
        "teacher_model_config": asdict(query_config),
        "shards": shard_names,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["cache_manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
