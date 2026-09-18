from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from framework.metrics import compute_FRC
from framework.modules.post_processor import Processor
from regnn.conditional_model import (
    ConditionalREGNN,
    ConditionalREGNNConfig,
)
from regnn.eval_conditional_regnn_official_test import (
    build_official_test_loader,
    chunk_offline_stream,
    initialize_official_hydra_runtime,
)
from regnn.query_mamba import (
    QueryMambaConfig,
    QueryResidualMamba,
    apply_residual_channel_multipliers as rebuild_channel_scaled_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official REACT2025 offline-test FRC for Query-Mamba."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=750)
    parser.add_argument("--loader-batch-size", type=int, default=4)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument(
        "--no-data-clamp",
        action="store_true",
        help="Disable official rounding of the first 15 AU channels.",
    )
    return parser.parse_args()


def generate_official_offline_query_batch(
    model: QueryResidualMamba,
    speaker_audio_clips: Sequence[torch.Tensor],
    speaker_emotion_clips: Sequence[torch.Tensor],
    speaker_3dmm_clips: Sequence[torch.Tensor],
    speaker_seq_lengths: Sequence[int],
    device: torch.device,
    clip_length: int,
    clip_batch_size: int,
    num_preds: int,
    residual_scale: float,
    data_clamp: bool,
    style_residual_scale: Optional[float] = None,
    residual_channel_multipliers: Optional[Sequence[float]] = None,
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
            model_kwargs = {"residual_scale": residual_scale}
            if style_residual_scale is not None:
                model_kwargs["style_residual_scale"] = style_residual_scale
            output = model(
                all_audio[start:stop].to(device, non_blocking=True),
                all_emotion[start:stop].to(device, non_blocking=True),
                all_face[start:stop].to(device, non_blocking=True),
                all_lengths[start:stop].to(device, non_blocking=True),
                **model_kwargs,
            )
            if residual_channel_multipliers is None:
                prediction = output["prediction"]
            else:
                prediction = apply_residual_channel_multipliers(
                    model,
                    output,
                    residual_channel_multipliers,
                )
            predicted_chunks.append(
                prediction.detach().float().cpu()
            )
    predicted_chunks = torch.cat(predicted_chunks, dim=0)
    if (
        predicted_chunks.ndim != 4
        or predicted_chunks.shape[1] != num_preds
        or predicted_chunks.shape[-1] != 25
    ):
        raise RuntimeError(
            "Query-Mamba chunks must have shape [C,10,T,25], got "
            f"{tuple(predicted_chunks.shape)}"
        )

    predictions = []
    offset = 0
    for chunk_count, lengths in zip(
        sample_chunk_counts,
        motion_lengths,
    ):
        sample_chunks = predicted_chunks[offset:offset + chunk_count]
        sequence_length = int(lengths.sum())
        prediction = (
            sample_chunks.permute(1, 0, 2, 3)
            .reshape(num_preds, -1, 25)[:, :sequence_length]
            .contiguous()
        )
        if data_clamp:
            prediction[:, :, :15] = torch.round(
                prediction[:, :, :15]
            )
        predictions.append(prediction)
        offset += chunk_count
    return predictions


def generate_official_online_query_batch(
    model: QueryResidualMamba,
    speaker_audio_clips: Sequence[torch.Tensor],
    speaker_emotion_clips: Sequence[torch.Tensor],
    speaker_3dmm_clips: Sequence[torch.Tensor],
    speaker_seq_lengths: Sequence[int],
    device: torch.device,
    window_size: int,
    speaker_ratio: int,
    clip_batch_size: int,
    num_preds: int,
    residual_scale: float,
    data_clamp: bool,
    style_residual_scale: Optional[float] = None,
    residual_channel_multipliers: Optional[Sequence[float]] = None,
    speaker_window_size: Optional[int] = None,
    visible_speaker_history_frames: Optional[int] = None,
    mask_final_padding: bool = False,
) -> List[torch.Tensor]:
    """Generate full sequences with a strictly causal online geometry.

    The REACT2025 online diffusion evaluator left-pads each speaker stream by
    ``speaker_window - listener_window`` frames, advances by one listener
    window, and keeps the final listener-window-sized part of every model
    output.  ``speaker_window_size`` can enlarge the *past* speaker history
    for a causal-memory diagnostic while keeping the listener output window
    fixed.  ``visible_speaker_history_frames`` optionally zeros older history
    while preserving the same input shape and positional coordinates, which
    provides a matched positional control.  In every case, the last speaker
    frame visible to a model call is the last frame of the output interval;
    no future speaker frame enters the call.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if speaker_ratio <= 0:
        raise ValueError("speaker_ratio must be positive")
    if clip_batch_size <= 0:
        raise ValueError("clip_batch_size must be positive")
    speaker_window = (
        int(speaker_window_size)
        if speaker_window_size is not None
        else speaker_ratio * window_size
    )
    if speaker_window < window_size:
        raise ValueError(
            "speaker_window_size must be at least the output window_size"
        )
    visible_history = (
        speaker_window
        if visible_speaker_history_frames is None
        else int(visible_speaker_history_frames)
    )
    if not window_size <= visible_history <= speaker_window:
        raise ValueError(
            "visible_speaker_history_frames must be between output "
            "window_size and speaker_window_size"
        )
    prefix_frames = speaker_window - window_size

    audio_windows = []
    emotion_windows = []
    face_windows = []
    sample_window_counts = []
    sample_lengths = []
    window_valid_lengths = []

    for audio, emotion, face, raw_length in zip(
        speaker_audio_clips,
        speaker_emotion_clips,
        speaker_3dmm_clips,
        speaker_seq_lengths,
    ):
        length = int(raw_length)
        if length <= 0:
            raise ValueError("Online sequences must contain at least one frame")
        if min(audio.shape[0], emotion.shape[0], face.shape[0]) < length:
            raise ValueError("A conditional stream is shorter than its length")
        window_count = math.ceil(length / window_size)
        suffix_frames = window_count * window_size - length

        def make_windows(stream: torch.Tensor) -> torch.Tensor:
            padded = torch.cat(
                (
                    stream.new_zeros(prefix_frames, stream.shape[-1]),
                    stream[:length],
                    stream.new_zeros(suffix_frames, stream.shape[-1]),
                ),
                dim=0,
            )
            return torch.stack(
                [
                    padded[
                        index * window_size:
                        index * window_size + speaker_window
                    ]
                    for index in range(window_count)
                ],
                dim=0,
            )

        audio_windows.append(make_windows(audio))
        emotion_windows.append(make_windows(emotion))
        face_windows.append(make_windows(face))
        sample_window_counts.append(window_count)
        sample_lengths.append(length)
        window_valid_lengths.extend(
            [speaker_window] * (window_count - 1)
            + [speaker_window - suffix_frames]
        )

    all_audio = torch.cat(audio_windows, dim=0)
    all_emotion = torch.cat(emotion_windows, dim=0)
    all_face = torch.cat(face_windows, dim=0)
    actual_window_lengths = torch.tensor(
        window_valid_lengths,
        dtype=torch.long,
    )
    if visible_history < speaker_window:
        frame_ids = torch.arange(speaker_window).unsqueeze(0)
        visible_starts = (
            actual_window_lengths - visible_history
        ).clamp_min(0).unsqueeze(1)
        visible_mask = frame_ids >= visible_starts
        all_audio = all_audio * visible_mask.unsqueeze(-1).to(all_audio)
        all_emotion = (
            all_emotion * visible_mask.unsqueeze(-1).to(all_emotion)
        )
        all_face = all_face * visible_mask.unsqueeze(-1).to(all_face)
    model_lengths = (
        actual_window_lengths
        if mask_final_padding
        else torch.full(
            (all_audio.shape[0],),
            speaker_window,
            dtype=torch.long,
        )
    )

    predicted_windows = []
    with torch.no_grad():
        for start in range(0, all_audio.shape[0], clip_batch_size):
            stop = start + clip_batch_size
            model_kwargs = {"residual_scale": residual_scale}
            if style_residual_scale is not None:
                model_kwargs["style_residual_scale"] = style_residual_scale
            output = model(
                all_audio[start:stop].to(device, non_blocking=True),
                all_emotion[start:stop].to(device, non_blocking=True),
                all_face[start:stop].to(device, non_blocking=True),
                model_lengths[start:stop].to(device, non_blocking=True),
                **model_kwargs,
            )
            if residual_channel_multipliers is None:
                prediction = output["prediction"]
            else:
                prediction = apply_residual_channel_multipliers(
                    model,
                    output,
                    residual_channel_multipliers,
                )
            predicted_windows.append(
                prediction[..., -window_size:, :].detach().float().cpu()
            )
    predicted_windows = torch.cat(predicted_windows, dim=0)
    if (
        predicted_windows.ndim != 4
        or predicted_windows.shape[1] != num_preds
        or predicted_windows.shape[2] != window_size
        or predicted_windows.shape[-1] != 25
    ):
        raise RuntimeError(
            "Online Mam-Reactor windows must have shape [W,10,w,25], got "
            f"{tuple(predicted_windows.shape)}"
        )

    predictions = []
    offset = 0
    for window_count, sequence_length in zip(
        sample_window_counts,
        sample_lengths,
    ):
        sample_windows = predicted_windows[offset:offset + window_count]
        prediction = (
            sample_windows.permute(1, 0, 2, 3)
            .reshape(num_preds, -1, 25)[:, :sequence_length]
            .contiguous()
        )
        if data_clamp:
            prediction[:, :, :15] = torch.round(
                prediction[:, :, :15]
            )
        predictions.append(prediction)
        offset += window_count
    return predictions


def apply_residual_channel_multipliers(
    model: QueryResidualMamba,
    output: Dict[str, torch.Tensor],
    multipliers: Sequence[float],
) -> torch.Tensor:
    """Compatibility wrapper for the shared train/eval implementation."""
    return rebuild_channel_scaled_predictions(
        model,
        output,
        multipliers,
    )


def main() -> None:
    args = parse_args()
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be positive")
    if args.residual_scale < 0.0:
        raise ValueError("--residual-scale must be non-negative")
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
    if not torch.cuda.is_available():
        raise RuntimeError("Official Query-Mamba evaluation requires CUDA")

    project_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    metrics_path = args.metrics_json.resolve()
    data_clamp = not args.no_data_clamp
    initialize_official_hydra_runtime(project_dir)

    random.seed(args.selection_seed)
    np.random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    torch.cuda.manual_seed_all(args.selection_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    anchor_config = ConditionalREGNNConfig(
        **checkpoint["anchor_model_config"]
    )
    query_config = QueryMambaConfig(
        **checkpoint["query_model_config"]
    )
    if (
        anchor_config.max_seq_len != args.clip_length
        or query_config.num_queries != 10
    ):
        raise ValueError("Checkpoint and official test contracts differ")
    anchor = ConditionalREGNN(anchor_config)
    model = QueryResidualMamba(anchor, query_config).to(device)
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
        cfg_dir=str(project_dir),
        clip_len_test=1000,
        device=device,
        num_preds=query_config.num_queries,
    )

    predictions: List[torch.Tensor] = []
    anchor_predictions: List[torch.Tensor] = []
    raw_targets: List[Sequence[torch.Tensor]] = []
    generation_start = time.perf_counter()
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
            if len(target_group) != query_config.num_queries:
                raise RuntimeError(
                    "Official test example must contain exactly 10 GTs"
                )
        raw_targets.extend(listener_emotion_clips)
        batch_predictions = generate_official_offline_query_batch(
            model=model,
            speaker_audio_clips=speaker_audio_clips,
            speaker_emotion_clips=speaker_emotion_clips,
            speaker_3dmm_clips=speaker_3dmm_clips,
            speaker_seq_lengths=speaker_seq_lengths,
            device=device,
            clip_length=args.clip_length,
            clip_batch_size=args.clip_batch_size,
            num_preds=query_config.num_queries,
            residual_scale=args.residual_scale,
            data_clamp=data_clamp,
        )
        predictions.extend(batch_predictions)
        anchor_predictions.extend(
            [
                prediction[:1]
                .expand(query_config.num_queries, -1, -1)
                .contiguous()
                for prediction in batch_predictions
            ]
        )
        print(
            f"Generated official Query-Mamba test batch "
            f"{batch_index + 1}/{len(test_loader)}; "
            f"examples={len(predictions)}",
            flush=True,
        )

    if not predictions:
        raise RuntimeError("No official test examples were evaluated")
    if not (
        len(predictions)
        == len(anchor_predictions)
        == len(raw_targets)
    ):
        raise RuntimeError("Prediction/target count mismatch")

    torch.cuda.synchronize(device)
    generation_time = time.perf_counter() - generation_start
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    postprocess_start = time.perf_counter()
    targets = processor.forward(
        prediction_list=predictions,
        target_list=raw_targets,
    )
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
    anchor_frc_start = time.perf_counter()
    anchor_frc = float(
        compute_FRC(
            anchor_predictions,
            targets,
            p=args.metric_workers,
        )
    )
    anchor_frc_time = time.perf_counter() - anchor_frc_start

    full_test = (
        args.max_test_batches is None
        and len(predictions) == len(dataset)
    )
    payload = {
        "FRC": frc,
        "FRC_delta_vs_anchor": frc - anchor_frc,
        "FRC_time": frc_time,
        "anchor_replicated_FRC": anchor_frc,
        "anchor_replicated_FRC_time": anchor_frc_time,
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
        "num_preds": query_config.num_queries,
        "peak_allocated_gib": peak_allocated_gib,
        "postprocess_time": postprocess_time,
        "prediction_policy": "anchor_plus_nine_query_mamba_residuals",
        "protocol": "official_diffusion_offline_test_frc_only",
        "residual_scale": args.residual_scale,
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
