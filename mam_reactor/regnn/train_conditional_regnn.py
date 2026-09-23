from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler

from regnn.conditional_data import ConditionalReactionDataset
from regnn.conditional_model import (
    ConditionalREGNN,
    ConditionalREGNNConfig,
    ConditionalREGNNLoss,
)
from regnn.session_splits import load_session_allowlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train full-context Conditional-REGNN on diffusion conditions."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--session-split-manifest", type=Path)
    parser.add_argument(
        "--session-split-name",
        choices=("B_fit", "B_cal", "B_confirm"),
    )
    parser.add_argument("--clip-length", type=int, default=750)
    parser.add_argument(
        "--crop-stride",
        type=int,
        default=1,
        help="Restrict random training crop starts to this frame stride.",
    )
    parser.add_argument(
        "--crop-views-per-record",
        type=int,
        default=1,
        help="Draw this many deterministic crop views per record and epoch.",
    )
    parser.add_argument(
        "--model-max-seq-len",
        type=int,
        help="Model positional capacity; defaults to --clip-length.",
    )
    parser.add_argument(
        "--supervision-tail-frames",
        type=int,
        default=0,
        help=(
            "Restrict loss to the final N frames while retaining the full "
            "window as model context; zero supervises every valid frame."
        ),
    )
    parser.add_argument(
        "--online-rollout-blocks",
        type=int,
        default=1,
        help=(
            "Train on this many consecutive online output blocks per "
            "example. Each model call still sees only --clip-length frames; "
            "the retained tails are stitched before CCC/loss computation."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ccc-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help=(
            "Stop after this many optimizer steps across all epochs; zero "
            "runs the full epoch schedule."
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load model weights only and start a fresh optimizer schedule.",
    )
    parser.add_argument("--listener-3dmm-weight", type=float, default=0.0)
    parser.add_argument("--listener-3dmm-velocity-weight", type=float, default=0.1)
    parser.add_argument(
        "--fixed-pair-rank", type=int,
        help="Rank 0 uses the true pair; ranks 1-9 use fixed same-session alternatives.",
    )
    parser.add_argument("--fixed-pair-count", type=int, default=10)
    parser.add_argument("--fixed-pair-seed", type=int)
    parser.add_argument("--fixed-pair-manifest", type=Path)
    parser.add_argument(
        "--fixed-pair-alignment-policy",
        choices=("diffusion_legacy", "relative_time_masked"),
        default="diffusion_legacy",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ExpandedCropViewSampler(Sampler[tuple[int, int]]):
    """Shuffle repeated record indices while assigning distinct crop views."""

    def __init__(self, dataset_size: int, views_per_record: int, seed: int):
        if dataset_size <= 0 or views_per_record <= 0:
            raise ValueError("dataset_size and views_per_record must be positive")
        self.dataset_size = int(dataset_size)
        self.views_per_record = int(views_per_record)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.dataset_size * self.views_per_record

    def __iter__(self):
        values = [
            (record_index, view_index)
            for record_index in range(self.dataset_size)
            for view_index in range(self.views_per_record)
        ]
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(values), generator=generator).tolist()
        return iter(values[index] for index in order)


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_config(args: argparse.Namespace) -> ConditionalREGNNConfig:
    return ConditionalREGNNConfig(
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        graph_dim=args.graph_dim,
        graph_layers=args.graph_layers,
        graph_heads=args.graph_heads,
        dropout=args.dropout,
        max_seq_len=(args.model_max_seq_len or args.clip_length),
        listener_3dmm=args.listener_3dmm_weight > 0,
    )


def online_rollout_span(
    context_frames: int,
    output_frames: int,
    rollout_blocks: int,
) -> int:
    if context_frames <= 0 or rollout_blocks <= 0:
        raise ValueError("context_frames and rollout_blocks must be positive")
    if rollout_blocks == 1:
        return context_frames
    if output_frames <= 0 or output_frames > context_frames:
        raise ValueError(
            "multi-block rollout requires output_frames in [1,context_frames]"
        )
    return context_frames + (rollout_blocks - 1) * output_frames


def forward_online_rollout(
    model: ConditionalREGNN,
    speaker_audio: torch.Tensor,
    speaker_emotion: torch.Tensor,
    speaker_3dmm: torch.Tensor,
    lengths: torch.Tensor,
    *,
    context_frames: int,
    output_frames: int,
    rollout_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-size online calls and stitch only their retained tails."""
    expected_span = online_rollout_span(
        context_frames,
        output_frames,
        rollout_blocks,
    )
    streams = (speaker_audio, speaker_emotion, speaker_3dmm)
    if any(stream.ndim != 3 for stream in streams):
        raise ValueError("Every conditional stream must have shape [B,T,D]")
    if any(stream.shape[:2] != speaker_audio.shape[:2] for stream in streams):
        raise ValueError("Conditional streams must share batch/time dimensions")
    if speaker_audio.shape[1] != expected_span:
        raise ValueError(
            f"Rollout input has {speaker_audio.shape[1]} frames; "
            f"expected {expected_span}"
        )
    if lengths.shape != (speaker_audio.shape[0],):
        raise ValueError("lengths must have shape [B]")

    starts = [block * output_frames for block in range(rollout_blocks)]

    def windows(stream: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [stream[:, start : start + context_frames] for start in starts],
            dim=1,
        ).flatten(0, 1)

    window_lengths = torch.stack(
        [
            (lengths - start).clamp(min=0, max=context_frames)
            for start in starts
        ],
        dim=1,
    ).flatten()
    output = model(
        windows(speaker_audio),
        windows(speaker_emotion),
        windows(speaker_3dmm),
        window_lengths.clamp_min(1),
    )
    batch_size = speaker_audio.shape[0]
    prediction = (
        output["prediction"][:, -output_frames:]
        .reshape(batch_size, rollout_blocks, output_frames, -1)
        .flatten(1, 2)
    )
    valid_mask = (
        output["valid_mask"][:, -output_frames:]
        .reshape(batch_size, rollout_blocks, output_frames)
        .flatten(1, 2)
    )
    return prediction, valid_mask


def assert_initialization_config_compatible(
    source: Dict[str, object],
    destination: Dict[str, object],
) -> None:
    """Allow only sinusoidal positional-capacity changes on initialization."""
    source = dict(source)
    destination = dict(destination)
    source.setdefault("listener_3dmm", False)
    destination.setdefault("listener_3dmm", False)
    source.pop("max_seq_len", None)
    destination.pop("max_seq_len", None)
    if source != destination:
        raise ValueError(
            "Initialization checkpoint architecture does not match"
        )


def save_checkpoint(
    path: Path,
    model: ConditionalREGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        {
            "epoch": epoch,
            "seed": args.seed,
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_config": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.listener_3dmm_weight < 0 or args.listener_3dmm_velocity_weight < 0:
        raise ValueError("3DMM weights must be nonnegative")
    if args.listener_3dmm_weight > 0 and (args.online_rollout_blocks != 1 or args.supervision_tail_frames != 0):
        raise ValueError("Joint anchor currently supports full-context offline training only")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.clip_length <= 0:
        raise ValueError("clip-length must be positive")
    if args.crop_stride <= 0 or args.crop_views_per_record <= 0:
        raise ValueError("crop-stride and crop-views-per-record must be positive")
    if args.online_rollout_blocks <= 0:
        raise ValueError("online-rollout-blocks must be positive")
    if args.max_train_batches < 0 or args.max_train_steps < 0:
        raise ValueError("training step limits must be non-negative")
    if args.model_max_seq_len is not None and (
        args.model_max_seq_len < args.clip_length
    ):
        raise ValueError("model-max-seq-len must cover clip-length")
    if not 0 <= args.supervision_tail_frames <= args.clip_length:
        raise ValueError("supervision-tail-frames must be in [0,clip-length]")
    dataset_clip_length = online_rollout_span(
        args.clip_length,
        args.supervision_tail_frames,
        args.online_rollout_blocks,
    )
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("Choose only one of --resume and --initialize-from")
    if not torch.cuda.is_available():
        raise RuntimeError("Conditional-REGNN formal training requires CUDA")

    args.data_dir = args.data_dir.resolve()
    args.fixed_pair_manifest_sha256 = (
        hashlib.sha256(args.fixed_pair_manifest.read_bytes()).hexdigest()
        if args.fixed_pair_manifest is not None else None
    )
    args.run_dir = args.run_dir.resolve()
    session_allowlist, session_split_manifest_sha256 = (
        load_session_allowlist(
            args.session_split_manifest,
            args.session_split_name,
        )
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "train_metrics.jsonl"
    completion_path = args.run_dir / "TRAINING_COMPLETE"
    if completion_path.exists():
        raise RuntimeError(
            f"Run is already marked complete: {args.run_dir}"
        )
    if metrics_path.exists() and args.resume is None:
        raise RuntimeError(f"Refusing to overwrite existing metrics: {metrics_path}")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    dataset = ConditionalReactionDataset(
        root_dir=str(args.data_dir),
        split="train",
        clip_length=dataset_clip_length,
        training=True,
        target_mode="paired",
        session_allowlist=session_allowlist,
        seed=args.seed,
        crop_stride=args.crop_stride,
        load_listener_3dmm=args.listener_3dmm_weight > 0,
        fixed_pair_rank=args.fixed_pair_rank,
        fixed_pair_count=args.fixed_pair_count,
        fixed_pair_seed=args.fixed_pair_seed,
        fixed_pair_manifest=args.fixed_pair_manifest,
        fixed_pair_alignment_policy=args.fixed_pair_alignment_policy,
    )
    sampler = ExpandedCropViewSampler(
        len(dataset),
        views_per_record=args.crop_views_per_record,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        drop_last=True,
    )

    model = ConditionalREGNN(build_config(args)).to(device)
    criterion = ConditionalREGNNLoss(
        ccc_weight=args.ccc_weight,
        velocity_weight=args.velocity_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    start_epoch = 0
    initialization_checkpoint = None
    if args.initialize_from is not None:
        args.initialize_from = args.initialize_from.resolve()
        initialization_checkpoint = torch.load(
            args.initialize_from,
            map_location="cpu",
        )
        assert_initialization_config_compatible(
            initialization_checkpoint["model_config"],
            asdict(model.config),
        )
        model.load_state_dict(initialization_checkpoint["state_dict"])
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if asdict(ConditionalREGNNConfig(**checkpoint["model_config"])) != asdict(model.config):
            raise ValueError("Resume checkpoint model config does not match")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config_payload = {
        "method": "Conditional-REGNN-stage1-mean",
        "conditions": ["audio_768", "speaker_facial_25", "speaker_3dmm_58"],
        "output": "listener_facial_25",
        "dataset_size": len(dataset),
        "examples_per_epoch": len(sampler),
        "dataset_clip_length": dataset_clip_length,
        "online_rollout_output_frames": (
            args.online_rollout_blocks * args.supervision_tail_frames
            if args.online_rollout_blocks > 1
            else (args.supervision_tail_frames or args.clip_length)
        ),
        "parameter_count": parameter_count,
        "session_allowlist": session_allowlist,
        "session_split_manifest_sha256": session_split_manifest_sha256,
        "session_split_name": args.session_split_name,
        "model_config": asdict(model.config),
        "initialization_checkpoint": (
            str(args.initialize_from)
            if args.initialize_from is not None
            else None
        ),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (args.run_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config_payload, handle, indent=2, sort_keys=True)

    print(
        f"Conditional-REGNN parameters={parameter_count:,} "
        f"samples={len(dataset)} batches={len(dataloader)} "
        f"precision={args.precision}",
        flush=True,
    )
    use_bf16 = args.precision == "bf16"
    global_step = start_epoch * len(dataloader)

    for epoch_index in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch_index)
        sampler.set_epoch(epoch_index)
        model.train()
        epoch_start = time.perf_counter()
        sums = {
            "loss": 0.0,
            "mse": 0.0,
            "ccc": 0.0,
            "ccc_loss": 0.0,
            "velocity": 0.0,
        }
        if args.listener_3dmm_weight > 0:
            sums.update(geometry_mse=0.0, geometry_velocity=0.0)
        processed = 0
        for batch_index, batch in enumerate(dataloader):
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break
            if (
                args.max_train_batches > 0
                and batch_index >= args.max_train_batches
            ):
                break
            speaker_audio = batch["speaker_audio"].to(
                device, non_blocking=True
            )
            speaker_emotion = batch["speaker_emotion"].to(
                device, non_blocking=True
            )
            speaker_3dmm = batch["speaker_3dmm"].to(
                device, non_blocking=True
            )
            target = batch["target"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output_frames = (
                    args.supervision_tail_frames or args.clip_length
                )
                if args.listener_3dmm_weight > 0:
                    joint_output = model(speaker_audio, speaker_emotion, speaker_3dmm, lengths)
                    prediction = joint_output["prediction"]
                    valid_mask = joint_output["valid_mask"]
                else:
                    prediction, valid_mask = forward_online_rollout(
                        model,
                        speaker_audio,
                        speaker_emotion,
                        speaker_3dmm,
                        lengths,
                        context_frames=args.clip_length,
                        output_frames=output_frames,
                        rollout_blocks=args.online_rollout_blocks,
                    )
                target_start = args.clip_length - output_frames
                target = target[
                    :,
                    target_start : target_start + prediction.shape[1],
                ]
                losses = criterion(
                    prediction,
                    target,
                    valid_mask,
                    batch.get("velocity_supervision"),
                )
            if args.listener_3dmm_weight > 0:
                from regnn.anchor_3dmm_loss import geometry_loss
                geometry_metrics = geometry_loss(
                    joint_output["prediction_3dmm"],
                    batch["target_3dmm"].to(device, non_blocking=True), valid_mask,
                )
                losses.update(geometry_metrics)
                losses["loss"] = losses["loss"] + args.listener_3dmm_weight * (
                    losses["geometry_mse"] + args.listener_3dmm_velocity_weight * losses["geometry_velocity"]
                )
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch_index+1}, batch {batch_index+1}")
            losses["loss"].backward()
            gradient_norm = clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            processed += 1
            global_step += 1
            for key in sums:
                sums[key] += float(losses[key].detach())

            if (
                batch_index == 0
                or (batch_index + 1) % args.print_every == 0
            ):
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch [{epoch_index + 1}/{args.epochs}] "
                    f"step [{batch_index + 1}/{len(dataloader)}] "
                    f"loss={losses['loss'].item():.5f} "
                    f"mse={losses['mse'].item():.5f} "
                    f"ccc={losses['ccc'].item():.5f} "
                    f"velocity={losses['velocity'].item():.5f} "
                    f"grad={float(gradient_norm):.3f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if processed == 0:
            raise RuntimeError("No training batches were processed")
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_metrics = {
            "epoch": epoch_index + 1,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "gpu_peak_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            **{key: value / processed for key, value in sums.items()},
        }
        append_jsonl(metrics_path, epoch_metrics)
        print(
            "Epoch summary " + json.dumps(epoch_metrics, sort_keys=True),
            flush=True,
        )

        epoch = epoch_index + 1
        max_steps_reached = (
            args.max_train_steps > 0
            and global_step >= args.max_train_steps
        )
        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
            or max_steps_reached
        ):
            checkpoint_path = (
                args.run_dir
                / f"conditional-epoch{epoch:04d}-seed{args.seed}.pth"
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                args,
            )
            print(f"Saved {checkpoint_path}", flush=True)

        if args.max_train_batches > 0 or max_steps_reached:
            break

    if args.max_train_batches == 0:
        completion_path.write_text(
            json.dumps(
                {
                    "epochs": args.epochs,
                    "global_steps": global_step,
                    "max_train_steps": args.max_train_steps,
                    "seed": args.seed,
                    "completed_at_unix": time.time(),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
