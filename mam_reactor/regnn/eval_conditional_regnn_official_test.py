from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader

from dataset.react_2025 import ReactionDataset, custom_collate
from framework.metrics import compute_FRC
from framework.modules.post_processor import Processor
from regnn.conditional_model import (
    ConditionalREGNN,
    ConditionalREGNNConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Conditional-REGNN on the REACT2025 test split using "
            "the official offline diffusion test protocol and FRC metric."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=750)
    parser.add_argument("--num-preds", type=int, default=10)
    parser.add_argument("--loader-batch-size", type=int, default=4)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument(
        "--no-data-clamp",
        action="store_true",
        help=(
            "Disable the official diffusion test-time rounding of the first "
            "15 AU channels. The official-compatible default is to clamp."
        ),
    )
    return parser.parse_args()


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def initialize_official_hydra_runtime(project_dir: Path) -> None:
    """Provide the cwd context assumed by the official ReactionDataset."""
    if HydraConfig.initialized():
        return
    project_dir = project_dir.resolve()
    previous_cwd = Path.cwd()
    try:
        # ReactionDataset resolves FaceVerse statistics through Hydra's
        # original cwd. A lightweight Git worktree may intentionally omit
        # those large assets, so initialize Hydra from the explicit asset
        # project while restoring the caller's process cwd afterwards.
        os.chdir(project_dir)
        with initialize_config_dir(
            version_base=None,
            config_dir=str((project_dir / "configs").resolve()),
            job_name="conditional_regnn_official_test",
        ):
            config = compose(config_name=None, return_hydra_config=True)
            HydraConfig.instance().set_config(config)
    finally:
        os.chdir(previous_cwd)


def chunk_offline_stream(
    stream: torch.Tensor,
    length: int,
    clip_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the official offline diffusion test chunking convention."""
    if stream.ndim != 2:
        raise ValueError(f"Expected [T,D] stream, got {tuple(stream.shape)}")
    if length <= 0:
        raise ValueError("Sequence length must be positive")
    if stream.shape[0] < length:
        raise ValueError(
            f"Stream has {stream.shape[0]} frames but official length is "
            f"{length}"
        )

    stream = stream[:length]
    remain_length = length % clip_length
    chunk_count = math.ceil(
        (length + clip_length - remain_length) / clip_length
    )
    lengths = torch.tensor(
        [clip_length] * (chunk_count - 1) + [remain_length],
        dtype=torch.long,
    )
    padding = stream.new_zeros(
        clip_length - remain_length,
        stream.shape[-1],
    )
    chunks = torch.cat((stream, padding), dim=0).reshape(
        chunk_count,
        clip_length,
        stream.shape[-1],
    )
    return chunks, lengths


def build_official_test_loader(
    data_dir: Path,
    clip_length: int,
    batch_size: int,
    workers: int,
    selection_seed: int,
) -> Tuple[ReactionDataset, DataLoader]:
    # These flags preserve the official sample and 10-GT construction while
    # avoiding video-frame decoding and listener 3DMM loading, neither of
    # which is consumed by FRC.
    dataset = ReactionDataset(
        root_dir=str(data_dir),
        split="test",
        clip_length=clip_length,
        target_size=224,
        crop_size=224,
        fps=30,
        audio_feature_type="wav2vec",
        load_video_s=False,
        load_video_l=False,
        load_audio=True,
        load_emotion_s=True,
        load_emotion_l=True,
        load_3dmm_s=True,
        load_3dmm_l=False,
        normalize_3dmm="standard",
    )
    loader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=partial(
            seed_worker,
            base_seed=selection_seed,
        ),
    )
    return dataset, loader


def generate_official_offline_batch(
    model: ConditionalREGNN,
    speaker_audio_clips: Sequence[torch.Tensor],
    speaker_emotion_clips: Sequence[torch.Tensor],
    speaker_3dmm_clips: Sequence[torch.Tensor],
    speaker_seq_lengths: Sequence[int],
    device: torch.device,
    clip_length: int,
    clip_batch_size: int,
    num_preds: int,
    data_clamp: bool,
) -> List[torch.Tensor]:
    audio_chunks = []
    emotion_chunks = []
    face_chunks = []
    motion_lengths = []
    sample_chunk_counts = []

    for audio, emotion, face, raw_length in zip(
        speaker_audio_clips,
        speaker_emotion_clips,
        speaker_3dmm_clips,
        speaker_seq_lengths,
    ):
        length = int(raw_length)
        chunked_audio, lengths = chunk_offline_stream(
            audio,
            length,
            clip_length,
        )
        chunked_emotion, emotion_lengths = chunk_offline_stream(
            emotion,
            length,
            clip_length,
        )
        chunked_face, face_lengths = chunk_offline_stream(
            face,
            length,
            clip_length,
        )
        if not (
            torch.equal(lengths, emotion_lengths)
            and torch.equal(lengths, face_lengths)
        ):
            raise RuntimeError("Conditional stream chunk lengths disagree")
        audio_chunks.append(chunked_audio)
        emotion_chunks.append(chunked_emotion)
        face_chunks.append(chunked_face)
        motion_lengths.append(lengths)
        sample_chunk_counts.append(len(lengths))

    all_audio = torch.cat(audio_chunks, dim=0)
    all_emotion = torch.cat(emotion_chunks, dim=0)
    all_face = torch.cat(face_chunks, dim=0)
    all_lengths = torch.cat(motion_lengths, dim=0)

    predicted_chunks = []
    with torch.no_grad():
        for start in range(0, all_audio.shape[0], clip_batch_size):
            stop = start + clip_batch_size
            output = model(
                all_audio[start:stop].to(device, non_blocking=True),
                all_emotion[start:stop].to(device, non_blocking=True),
                all_face[start:stop].to(device, non_blocking=True),
                all_lengths[start:stop].to(device, non_blocking=True),
            )
            predicted_chunks.append(
                output["prediction"].detach().float().cpu()
            )
    predicted_chunks = torch.cat(predicted_chunks, dim=0)

    predictions = []
    offset = 0
    for chunk_count, lengths in zip(
        sample_chunk_counts,
        motion_lengths,
    ):
        sample_chunks = predicted_chunks[offset:offset + chunk_count]
        sequence_length = int(lengths.sum())
        mean_prediction = sample_chunks.reshape(-1, 25)[:sequence_length]
        if data_clamp:
            mean_prediction[:, :15] = torch.round(
                mean_prediction[:, :15]
            )
        predictions.append(
            mean_prediction.unsqueeze(0)
            .expand(num_preds, -1, -1)
            .contiguous()
        )
        offset += chunk_count
    return predictions


def main() -> None:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    if args.num_preds != 10:
        raise ValueError("Official REACT2025 test protocol requires 10 predictions")
    for name in (
        "loader_batch_size",
        "clip_batch_size",
        "metric_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.loader_workers < 0:
        raise ValueError("--loader-workers must be non-negative")
    if args.max_test_batches is not None and args.max_test_batches <= 0:
        raise ValueError("--max-test-batches must be positive")

    project_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    metrics_path = args.metrics_json.resolve()
    data_clamp = not args.no_data_clamp
    initialize_official_hydra_runtime(project_dir)

    random.seed(args.selection_seed)
    np.random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.selection_seed)

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ConditionalREGNNConfig(**checkpoint["model_config"])
    if config.max_seq_len != args.clip_length:
        raise ValueError(
            "Checkpoint and official chunk lengths must match: "
            f"{config.max_seq_len} vs {args.clip_length}"
        )
    model = ConditionalREGNN(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset, test_loader = build_official_test_loader(
        data_dir=data_dir,
        clip_length=args.clip_length,
        batch_size=args.loader_batch_size,
        workers=args.loader_workers,
        selection_seed=args.selection_seed,
    )
    processor = Processor(
        config_name="configs/model/emotion_autoencoder.yaml",
        ckpt_dir=str(project_dir / "pretrained_models" / "post_processor"),
        clip_len_test=1000,
        device=device,
        num_preds=args.num_preds,
    )

    predictions: List[torch.Tensor] = []
    raw_targets: List[Sequence[torch.Tensor]] = []
    generation_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, batch in enumerate(test_loader):
        if (
            args.max_test_batches is not None
            and batch_index >= args.max_test_batches
        ):
            break
        (
            speaker_audio_clips,
            _,
            speaker_emotion_clips,
            speaker_3dmm_clips,
            _,
            listener_emotion_clips,
            _,
            speaker_seq_lengths,
            _,
        ) = batch
        for target_group in listener_emotion_clips:
            if len(target_group) != args.num_preds:
                raise RuntimeError(
                    "Official test example must contain exactly "
                    f"{args.num_preds} real GT reactions, got "
                    f"{len(target_group)}"
                )
            for target in target_group:
                if target.ndim != 2 or target.shape[-1] != 25:
                    raise RuntimeError(
                        "Official FRC target must have shape [T,25], got "
                        f"{tuple(target.shape)}"
                    )
        raw_targets.extend(listener_emotion_clips)
        batch_predictions = generate_official_offline_batch(
            model=model,
            speaker_audio_clips=speaker_audio_clips,
            speaker_emotion_clips=speaker_emotion_clips,
            speaker_3dmm_clips=speaker_3dmm_clips,
            speaker_seq_lengths=speaker_seq_lengths,
            device=device,
            clip_length=args.clip_length,
            clip_batch_size=args.clip_batch_size,
            num_preds=args.num_preds,
            data_clamp=data_clamp,
        )
        for prediction in batch_predictions:
            if (
                prediction.ndim != 3
                or prediction.shape[0] != args.num_preds
                or prediction.shape[-1] != 25
            ):
                raise RuntimeError(
                    "Official FRC prediction must have shape [10,T,25], "
                    f"got {tuple(prediction.shape)}"
                )
        predictions.extend(batch_predictions)
        print(
            f"Generated official-test batch {batch_index + 1}/"
            f"{len(test_loader)}; examples={len(predictions)}",
            flush=True,
        )

    if not predictions:
        raise RuntimeError("No test examples were evaluated")
    if len(predictions) != len(raw_targets):
        raise RuntimeError(
            "Prediction/target count mismatch: "
            f"{len(predictions)} vs {len(raw_targets)}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_time = time.perf_counter() - generation_start
    peak_allocated_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )

    postprocess_start = time.perf_counter()
    targets = processor.forward(
        prediction_list=predictions,
        target_list=raw_targets,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    postprocess_time = time.perf_counter() - postprocess_start

    frc_start = time.perf_counter()
    frc = float(
        compute_FRC(
            predictions,
            targets,
            p=args.metric_workers,
        )
    )
    frc_time = time.perf_counter() - frc_start

    full_test = (
        args.max_test_batches is None
        and len(predictions) == len(dataset)
    )
    payload = {
        "FRC": frc,
        "FRC_time": frc_time,
        "checkpoint": str(checkpoint_path),
        "clip_batch_size": args.clip_batch_size,
        "clip_length": args.clip_length,
        "data_clamp": data_clamp,
        "epoch": int(checkpoint["epoch"]),
        "evaluation_type": "real_gt",
        "full_sequences": True,
        "full_test": full_test,
        "generation_time": generation_time,
        "gt_policy": "official_ReactionDataset_test_paired_plus_9_session",
        "loader_batch_size": args.loader_batch_size,
        "loader_workers": args.loader_workers,
        "max_test_batches": args.max_test_batches,
        "num_examples": len(predictions),
        "num_preds": args.num_preds,
        "peak_allocated_gib": peak_allocated_gib,
        "postprocess_time": postprocess_time,
        "prediction_policy": "conditional_mean_replicated_10",
        "protocol": "official_diffusion_offline_test_frc_only",
        "seed": int(checkpoint["seed"]),
        "selection_seed": args.selection_seed,
        "split": "test",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, metrics_path)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
