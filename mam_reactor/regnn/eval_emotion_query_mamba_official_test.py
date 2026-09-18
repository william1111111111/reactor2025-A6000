from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from framework.metrics import (
    compute_FRC,
    compute_FRVar,
    compute_TLCC,
    compute_TLCC_robust,
    compute_s_mse,
)
from framework.modules.post_processor import Processor
from regnn.conditional_model import (
    ConditionalREGNN,
    ConditionalREGNNConfig,
)
from regnn.emotion_query_mamba import (
    EmotionQueryMambaConfig,
    EmotionQueryResidualMamba,
)
from regnn.eval_conditional_regnn_official_test import (
    build_official_test_loader,
    initialize_official_hydra_runtime,
)
from regnn.eval_query_mamba_official_test import (
    generate_official_offline_query_batch,
    generate_official_online_query_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official REACT2025 offline or online-window test FRC, FRDiv, "
            "and FRVar for Emotion Query-Mamba with an optional "
            "inference-time gate-floor override."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--asset-project-dir",
        type=Path,
        help=(
            "Project root containing configs/, external/, and "
            "pretrained_models/. Defaults to the evaluator Git worktree."
        ),
    )
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument(
        "--results-pt",
        type=Path,
        help=(
            "Optionally save PRED and GT tensors for exact FRD. The file can "
            "be deleted after the resumable FRD job completes."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("offline", "online"),
        default="offline",
    )
    parser.add_argument("--clip-length", type=int, default=750)
    parser.add_argument("--online-window-size", type=int, default=30)
    parser.add_argument("--online-speaker-ratio", type=int, default=2)
    parser.add_argument(
        "--online-speaker-window-size",
        type=int,
        help=(
            "Explicit causal speaker-history window. The output stride and "
            "retained listener window remain --online-window-size. This is "
            "intended for fixed-output memory diagnostics; every call ends "
            "at the current output boundary and never reads future frames."
        ),
    )
    parser.add_argument(
        "--online-visible-speaker-history-frames",
        type=int,
        help=(
            "Matched positional control: retain this many newest speaker "
            "frames and zero older frames inside the unchanged model input "
            "window. Defaults to the complete speaker window."
        ),
    )
    parser.add_argument(
        "--mask-final-online-padding",
        action="store_true",
        help=(
            "Mask right-padding in the final partial online window. This "
            "does not expose any additional input and is used consistently "
            "across the causal-history diagnostic."
        ),
    )
    parser.add_argument("--loader-batch-size", type=int, default=4)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument(
        "--compute-robust-tlcc",
        action="store_true",
        help=(
            "Also compute the non-official NaN-safe all-query FRSyn "
            "diagnostic. Disabled by default, so FRC-only/training "
            "evaluation speed is unchanged."
        ),
    )
    parser.add_argument(
        "--target-postprocess-batch-size",
        type=int,
        default=0,
        help=(
            "Process real GTs in bounded batches to reduce GPU memory; "
            "0 preserves the historical all-at-once call."
        ),
    )
    parser.add_argument(
        "--target-cache-results-pt",
        type=Path,
        help=(
            "Reuse the GT tensors from a validated official results.pt. "
            "Predictions in that cache are ignored."
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--style-residual-scale",
        type=float,
        help=(
            "Optional independent scale for the static style adapter. "
            "Defaults to --residual-scale for backward compatibility."
        ),
    )
    parser.add_argument("--au-residual-multiplier", type=float, default=1.0)
    parser.add_argument("--va-residual-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--expression-residual-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument("--gate-floor-override", type=float)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument("--no-data-clamp", action="store_true")
    parser.add_argument(
        "--skip-anchor-metrics",
        action="store_true",
        help=(
            "Skip the replicated-anchor comparison metrics; model FRC, "
            "FRDiv, and FRVar remain unchanged."
        ),
    )
    return parser.parse_args()


def postprocess_targets_in_batches(
    processor: Processor,
    predictions: Sequence[torch.Tensor],
    raw_targets: Sequence[Sequence[torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], int]:
    """Apply the official target processor with bounded GPU residency."""
    if len(predictions) != len(raw_targets):
        raise ValueError("Prediction/target count mismatch")
    if batch_size < 0:
        raise ValueError("target postprocess batch size must be non-negative")
    effective_batch_size = batch_size or len(predictions)
    targets: List[torch.Tensor] = []
    calls = 0
    for start in range(0, len(predictions), effective_batch_size):
        stop = min(start + effective_batch_size, len(predictions))
        targets.extend(
            processor.forward(
                predictions[start:stop],
                raw_targets[start:stop],
            )
        )
        calls += 1
        torch.cuda.empty_cache()
        print(
            f"Post-processed official targets {stop}/{len(predictions)}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    return targets, calls


def load_official_test_targets_from_results_cache(
    cache_path: Path,
    predictions: Sequence[torch.Tensor],
    data_dir: Path,
    selection_seed: int,
    clip_length: int,
) -> Tuple[List[torch.Tensor], dict]:
    """Load protocol-matched official GTs from a result or GT-only cache."""
    resolved_path = cache_path.resolve()
    cache = torch.load(resolved_path, map_location="cpu")
    metadata = dict(cache.get("metadata", {}))
    cache_format = metadata.get("cache_format")
    if cache_format not in {
        "react2025_official_frd_PRED_GT_v1",
        "react2025_official_full_GT_v1",
    }:
        raise ValueError(
            f"Unsupported official target cache format: {cache_format}"
        )
    expected = {
        "data_dir": str(data_dir.resolve()),
        "split": "test",
        "selection_seed": int(selection_seed),
        "num_examples": len(predictions),
        "num_preds": 10,
        "full_test": True,
    }
    # A full-GT-only cache stores complete target sequences and is independent
    # of the model chunk length. Legacy PRED+GT caches remain length-specific.
    if cache_format == "react2025_official_frd_PRED_GT_v1":
        expected["clip_length"] = int(clip_length)
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "Official target cache metadata mismatch: "
            f"expected={expected}, actual={actual}"
        )
    raw_cached_targets = cache.get("GT")
    if not isinstance(raw_cached_targets, list):
        raise ValueError("Official target cache has no GT list")
    targets = [target.detach().cpu().contiguous() for target in raw_cached_targets]
    del cache
    if len(targets) != len(predictions):
        raise ValueError("Official target cache has the wrong example count")
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        expected_shape = (10, prediction.shape[1], 25)
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"Official cached GT {index} has shape {tuple(target.shape)}, "
                f"expected {expected_shape}"
            )
    return targets, metadata


def save_official_test_targets_cache(
    cache_path: Path,
    targets: Sequence[torch.Tensor],
    predictions: Sequence[torch.Tensor],
    data_dir: Path,
    selection_seed: int,
    clip_length: int,
) -> dict:
    """Atomically save reusable full-length official GTs without predictions."""
    resolved_path = cache_path.resolve()
    if resolved_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite target cache: {resolved_path}"
        )
    metadata = {
        "cache_format": "react2025_official_full_GT_v1",
        "data_dir": str(data_dir.resolve()),
        "split": "test",
        "selection_seed": int(selection_seed),
        "clip_length": int(clip_length),
        "num_examples": len(predictions),
        "num_preds": 10,
        "full_test": True,
        "lengths": [int(prediction.shape[1]) for prediction in predictions],
    }
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    torch.save(
        {
            "metadata": metadata,
            "GT": [target.detach().cpu().contiguous() for target in targets],
        },
        temporary,
    )
    os.replace(temporary, resolved_path)
    return metadata


def main() -> None:
    args = parse_args()
    if (
        args.clip_length <= 0
        or args.online_window_size <= 0
        or args.online_speaker_ratio <= 0
        or args.residual_scale < 0.0
    ):
        raise ValueError("Invalid clip length or residual scale")
    if (
        args.style_residual_scale is not None
        and args.style_residual_scale < 0.0
    ):
        raise ValueError("style-residual-scale must be non-negative")
    if min(
        args.au_residual_multiplier,
        args.va_residual_multiplier,
        args.expression_residual_multiplier,
    ) < 0.0:
        raise ValueError("Residual channel multipliers must be non-negative")
    if args.gate_floor_override is not None and not (
        0.0 <= args.gate_floor_override <= 1.0
    ):
        raise ValueError("gate-floor-override must be in [0,1]")
    for name in (
        "loader_batch_size",
        "clip_batch_size",
        "metric_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.loader_workers < 0:
        raise ValueError("loader-workers must be non-negative")
    if args.target_postprocess_batch_size < 0:
        raise ValueError(
            "target-postprocess-batch-size must be non-negative"
        )
    if args.max_test_batches is not None and args.max_test_batches <= 0:
        raise ValueError("max-test-batches must be positive")
    online_speaker_window_size = (
        args.online_speaker_window_size
        if args.online_speaker_window_size is not None
        else args.online_speaker_ratio * args.online_window_size
    )
    if online_speaker_window_size < args.online_window_size:
        raise ValueError(
            "--online-speaker-window-size must be at least "
            "--online-window-size"
        )
    online_visible_speaker_history_frames = (
        args.online_visible_speaker_history_frames
        if args.online_visible_speaker_history_frames is not None
        else online_speaker_window_size
    )
    if not (
        args.online_window_size
        <= online_visible_speaker_history_frames
        <= online_speaker_window_size
    ):
        raise ValueError(
            "--online-visible-speaker-history-frames must be between "
            "--online-window-size and the effective speaker window"
        )
    online_speaker_ratio_effective = (
        online_speaker_window_size / args.online_window_size
    )
    official_online_geometry = (
        args.online_window_size == 30
        and online_speaker_window_size == 60
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Official Emotion Query-Mamba test requires CUDA")

    project_dir = Path(__file__).resolve().parents[1]
    asset_project_dir = (
        args.asset_project_dir.resolve()
        if args.asset_project_dir is not None
        else project_dir
    )
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    metrics_path = args.metrics_json.resolve()
    data_clamp = not args.no_data_clamp
    initialize_official_hydra_runtime(asset_project_dir)

    random.seed(args.selection_seed)
    np.random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    torch.cuda.manual_seed_all(args.selection_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("method") not in {
        "Emotion-Gated-Query-Mamba",
        "EQR-Mamba",
    }:
        raise ValueError("Checkpoint is not an Emotion-Query Mamba model")
    anchor_config = ConditionalREGNNConfig(
        **checkpoint["anchor_model_config"]
    )
    query_config_payload = dict(
        checkpoint["emotion_query_model_config"]
    )
    trained_gate_floor = float(query_config_payload["gate_floor"])
    if args.gate_floor_override is not None:
        query_config_payload["gate_floor"] = args.gate_floor_override
    query_config = EmotionQueryMambaConfig.from_checkpoint_payload(
        query_config_payload
    )
    if (
        anchor_config.max_seq_len != args.clip_length
        or query_config.num_queries != 10
        or query_config.num_emotions != 8
    ):
        raise ValueError("Checkpoint and official test contracts differ")
    if (
        args.task == "online"
        and online_speaker_window_size > anchor_config.max_seq_len
    ):
        raise ValueError(
            "Online speaker window exceeds checkpoint max_seq_len: "
            f"{online_speaker_window_size} > {anchor_config.max_seq_len}"
        )
    anchor = ConditionalREGNN(anchor_config)
    model = EmotionQueryResidualMamba(anchor, query_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset, test_loader = build_official_test_loader(
        data_dir=data_dir,
        clip_length=args.clip_length,
        batch_size=args.loader_batch_size,
        workers=args.loader_workers,
        selection_seed=args.selection_seed,
    )
    target_cache_available = (
        args.target_cache_results_pt is not None
        and args.target_cache_results_pt.is_file()
    )
    processor = None
    if not target_cache_available:
        processor = Processor(
            config_name="configs/model/emotion_autoencoder.yaml",
            ckpt_dir=str(
                asset_project_dir / "pretrained_models" / "post_processor"
            ),
            cfg_dir=str(asset_project_dir),
            clip_len_test=1000,
            device=device,
            num_preds=query_config.num_queries,
        )

    predictions: List[torch.Tensor] = []
    anchor_predictions: List[torch.Tensor] = []
    raw_targets = []
    speakers: List[torch.Tensor] = []
    semantic_profiles = []
    semantic_accuracy_values = []
    emotion_l1_values = []
    distinct_mode_values = []
    au_flip_values = []
    generation_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    channel_multipliers = (
        [args.au_residual_multiplier] * 15
        + [args.va_residual_multiplier] * 2
        + [args.expression_residual_multiplier] * 8
    )
    use_channel_multipliers = any(
        multiplier != 1.0 for multiplier in channel_multipliers
    )

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
        speakers.extend(speaker_emotion_clips)
        for target_group in listener_emotion_clips:
            if len(target_group) != query_config.num_queries:
                raise RuntimeError(
                    "Official test example must contain exactly 10 GTs"
                )
        if not target_cache_available:
            raw_targets.extend(listener_emotion_clips)
        generation_kwargs = {
            "model": model,
            "speaker_audio_clips": speaker_audio_clips,
            "speaker_emotion_clips": speaker_emotion_clips,
            "speaker_3dmm_clips": speaker_3dmm_clips,
            "speaker_seq_lengths": speaker_seq_lengths,
            "device": device,
            "clip_batch_size": args.clip_batch_size,
            "num_preds": query_config.num_queries,
            "residual_scale": args.residual_scale,
            "data_clamp": data_clamp,
            "style_residual_scale": args.style_residual_scale,
            "residual_channel_multipliers": (
                channel_multipliers if use_channel_multipliers else None
            ),
        }
        if args.task == "online":
            batch_predictions = generate_official_online_query_batch(
                **generation_kwargs,
                window_size=args.online_window_size,
                speaker_ratio=args.online_speaker_ratio,
                speaker_window_size=online_speaker_window_size,
                visible_speaker_history_frames=(
                    online_visible_speaker_history_frames
                ),
                mask_final_padding=args.mask_final_online_padding,
            )
        else:
            batch_predictions = generate_official_offline_query_batch(
                **generation_kwargs,
                clip_length=args.clip_length,
            )
        predictions.extend(batch_predictions)
        if not args.skip_anchor_metrics:
            anchor_predictions.extend(
                [
                    prediction[:1]
                    .expand(query_config.num_queries, -1, -1)
                    .contiguous()
                    for prediction in batch_predictions
                ]
            )
        for prediction in batch_predictions:
            profiles = prediction[1:9, :, 17:25].mean(dim=1)
            profiles = profiles / profiles.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            semantic_profiles.append(profiles)
            labels = profiles.argmax(dim=-1)
            semantic_accuracy_values.append(
                float((labels == torch.arange(8)).float().mean())
            )
            distinct_mode_values.append(float(labels.unique().numel()))
            pairwise_l1 = (
                profiles.unsqueeze(1) - profiles.unsqueeze(0)
            ).abs().sum(dim=-1)
            off_diagonal = ~torch.eye(8, dtype=torch.bool)
            emotion_l1_values.append(
                float(pairwise_l1[off_diagonal].mean())
            )
            au_flip_values.append(
                float(
                    (
                        prediction[1:, :, :15]
                        != prediction[:1, :, :15]
                    ).float().mean()
                )
            )
        print(
            f"Generated official Emotion Query-Mamba test batch "
            f"{batch_index + 1}/{len(test_loader)}; "
            f"examples={len(predictions)}",
            flush=True,
        )

    if not predictions:
        raise RuntimeError("No official test examples were evaluated")
    if (
        not target_cache_available
        and len(predictions) != len(raw_targets)
    ):
        raise RuntimeError("Prediction/target count mismatch")
    if len(predictions) != len(speakers):
        raise RuntimeError("Prediction/speaker count mismatch")
    if (
        not args.skip_anchor_metrics
        and len(anchor_predictions) != len(predictions)
    ):
        raise RuntimeError("Anchor prediction count mismatch")

    torch.cuda.synchronize(device)
    generation_time = time.perf_counter() - generation_start
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    model.to("cpu")
    torch.cuda.empty_cache()
    postprocess_start = time.perf_counter()
    target_cache_metadata = None
    if target_cache_available:
        targets, target_cache_metadata = (
            load_official_test_targets_from_results_cache(
                cache_path=args.target_cache_results_pt,
                predictions=predictions,
                data_dir=data_dir,
                selection_seed=args.selection_seed,
                clip_length=args.clip_length,
            )
        )
        target_postprocessing_calls = 0
        target_cache_status = "loaded_validated_official_GT"
    else:
        assert processor is not None
        targets, target_postprocessing_calls = (
            postprocess_targets_in_batches(
                processor=processor,
                predictions=predictions,
                raw_targets=raw_targets,
                batch_size=args.target_postprocess_batch_size,
                device=device,
            )
        )
        if args.target_cache_results_pt is None:
            target_cache_status = "disabled"
        else:
            target_cache_metadata = save_official_test_targets_cache(
                cache_path=args.target_cache_results_pt,
                targets=targets,
                predictions=predictions,
                data_dir=data_dir,
                selection_seed=args.selection_seed,
                clip_length=args.clip_length,
            )
            target_cache_status = "created_validated_official_GT"
    postprocess_time = time.perf_counter() - postprocess_start
    peak_total_allocated_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )

    results_pt_status = "disabled"
    if args.results_pt is not None:
        results_path = args.results_pt.resolve()
        if results_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite result cache: {results_path}"
            )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_metadata = {
            "cache_format": (
                (
                    "react2025_online_window_PRED_GT_v1"
                    if official_online_geometry
                    and online_visible_speaker_history_frames
                    == online_speaker_window_size
                    else "react2025_causal_history_PRED_GT_v1"
                )
                if args.task == "online"
                else "react2025_official_frd_PRED_GT_v1"
            ),
            "checkpoint": str(checkpoint_path),
            "data_dir": str(data_dir),
            "split": "test",
            "selection_seed": int(args.selection_seed),
            "clip_length": int(args.clip_length),
            "num_examples": len(predictions),
            "num_preds": query_config.num_queries,
            "full_test": (
                args.max_test_batches is None
                and len(predictions) == len(dataset)
            ),
            "task": args.task,
            "online_window_size": (
                args.online_window_size if args.task == "online" else None
            ),
            "online_speaker_ratio": (
                online_speaker_ratio_effective
                if args.task == "online"
                else None
            ),
            "online_speaker_ratio_argument": (
                args.online_speaker_ratio if args.task == "online" else None
            ),
            "online_speaker_window_size": (
                online_speaker_window_size
                if args.task == "online"
                else None
            ),
            "online_visible_speaker_history_frames": (
                online_visible_speaker_history_frames
                if args.task == "online"
                else None
            ),
            "online_causal_no_future_input": (
                True if args.task == "online" else None
            ),
            "online_final_padding_masked": (
                args.mask_final_online_padding
                if args.task == "online"
                else None
            ),
        }
        temporary_results_path = results_path.with_suffix(
            results_path.suffix + ".tmp"
        )
        torch.save(
            {
                "metadata": results_metadata,
                "PRED": predictions,
                "GT": targets,
            },
            temporary_results_path,
        )
        os.replace(temporary_results_path, results_path)
        results_pt_status = "saved_for_exact_frd"

    frc_start = time.perf_counter()
    frc = float(compute_FRC(predictions, targets, p=args.metric_workers))
    frc_time = time.perf_counter() - frc_start
    if args.skip_anchor_metrics:
        anchor_frc = None
        anchor_frc_time = 0.0
        anchor_frdiv = None
        anchor_frdiv_time = 0.0
        anchor_frvar = None
        anchor_frvar_time = 0.0
    else:
        anchor_frc_start = time.perf_counter()
        anchor_frc = float(
            compute_FRC(
                anchor_predictions,
                targets,
                p=args.metric_workers,
            )
        )
        anchor_frc_time = time.perf_counter() - anchor_frc_start

    frdiv_start = time.perf_counter()
    frdiv = float(compute_s_mse(predictions))
    frdiv_time = time.perf_counter() - frdiv_start
    if not args.skip_anchor_metrics:
        anchor_frdiv_start = time.perf_counter()
        anchor_frdiv = float(compute_s_mse(anchor_predictions))
        anchor_frdiv_time = time.perf_counter() - anchor_frdiv_start

    frvar_start = time.perf_counter()
    frvar = float(compute_FRVar(predictions))
    frvar_time = time.perf_counter() - frvar_start
    if not args.skip_anchor_metrics:
        anchor_frvar_start = time.perf_counter()
        anchor_frvar = float(compute_FRVar(anchor_predictions))
        anchor_frvar_time = time.perf_counter() - anchor_frvar_start
    tlcc_start = time.perf_counter()
    tlcc = float(
        compute_TLCC(
            predictions,
            speakers,
            p=args.metric_workers,
        )
    )
    tlcc_time = time.perf_counter() - tlcc_start
    robust_tlcc = None
    robust_tlcc_time = 0.0
    if args.compute_robust_tlcc:
        robust_tlcc_start = time.perf_counter()
        robust_tlcc = compute_TLCC_robust(
            predictions,
            speakers,
            p=args.metric_workers,
        )
        robust_tlcc_time = time.perf_counter() - robust_tlcc_start
    full_test = (
        args.max_test_batches is None
        and len(predictions) == len(dataset)
    )
    sample_paths = getattr(dataset, "speaker_path_list", None)
    if sample_paths is None:
        evaluated_sample_hash = None
        evaluated_sample_id_policy = "dataset_does_not_expose_sample_paths"
    else:
        evaluated_ids = [
            str(path) for path in sample_paths[:len(predictions)]
        ]
        evaluated_sample_hash = hashlib.sha256(
            "\n".join(evaluated_ids).encode("utf-8")
        ).hexdigest()
        evaluated_sample_id_policy = (
            "sha256_of_newline_joined_dataset_speaker_path_list_prefix"
        )
    checkpoint_training_config = dict(
        checkpoint.get("training_config", {})
    )
    supervision_tail_frames = int(
        checkpoint_training_config.get("supervision_tail_frames", 0)
    )
    training_task = (
        "online_window"
        if supervision_tail_frames > 0
        else "offline"
    )
    mean_semantic_profile = torch.stack(semantic_profiles).mean(dim=0)
    payload = {
        "FRC": frc,
        "FRC_delta_vs_anchor": (
            frc - anchor_frc if anchor_frc is not None else None
        ),
        "FRC_time": frc_time,
        "FRDiv": frdiv,
        "FRDiv_time": frdiv_time,
        "FRVar": frvar,
        "FRVar_time": frvar_time,
        "TLCC": tlcc,
        "TLCC_time": tlcc_time,
        "TLCC_qualification": (
            "official implementation evaluates prediction index 0 only; "
            "Emotion Query-Mamba prediction 0 is the frozen anchor"
        ),
        "FRSyn_robust": (
            robust_tlcc["FRSyn_robust"]
            if robust_tlcc is not None
            else None
        ),
        "FRSyn_robust_details": robust_tlcc,
        "FRSyn_robust_time": robust_tlcc_time,
        "FRSyn_robust_qualification": (
            "non-official diagnostic: undefined constant/non-finite channels "
            "are excluded and every generated query is evaluated"
            if robust_tlcc is not None
            else "not requested; pass --compute-robust-tlcc to enable"
        ),
        "anchor_replicated_FRC": anchor_frc,
        "anchor_replicated_FRC_time": anchor_frc_time,
        "anchor_replicated_FRDiv": anchor_frdiv,
        "anchor_replicated_FRDiv_time": anchor_frdiv_time,
        "anchor_replicated_FRVar": anchor_frvar,
        "anchor_replicated_FRVar_time": anchor_frvar_time,
        "anchor_metrics_skipped": args.skip_anchor_metrics,
        "asset_project_dir": str(asset_project_dir),
        "checkpoint": str(checkpoint_path),
        "clip_batch_size": args.clip_batch_size,
        "clip_length": args.clip_length,
        "data_clamp": data_clamp,
        "effective_gate_floor": query_config.gate_floor,
        "trained_gate_floor": trained_gate_floor,
        "epoch": int(checkpoint["epoch"]),
        "evaluation_type": "real_gt",
        "full_sequences": True,
        "full_test": full_test,
        "evaluated_sample_id_policy": evaluated_sample_id_policy,
        "evaluated_sample_sha256": evaluated_sample_hash,
        "generation_time": generation_time,
        "gt_policy": "official_ReactionDataset_test_paired_plus_9_session",
        "loader_batch_size": args.loader_batch_size,
        "loader_workers": args.loader_workers,
        "max_test_batches": args.max_test_batches,
        "num_examples": len(predictions),
        "num_preds": query_config.num_queries,
        "online_speaker_ratio": (
            online_speaker_ratio_effective
            if args.task == "online"
            else None
        ),
        "online_speaker_ratio_argument": (
            args.online_speaker_ratio if args.task == "online" else None
        ),
        "online_speaker_window_size": (
            online_speaker_window_size if args.task == "online" else None
        ),
        "online_visible_speaker_history_frames": (
            online_visible_speaker_history_frames
            if args.task == "online"
            else None
        ),
        "online_past_context_frames": (
            online_speaker_window_size - args.online_window_size
            if args.task == "online"
            else None
        ),
        "online_causal_no_future_input": (
            True if args.task == "online" else None
        ),
        "online_final_padding_masked": (
            args.mask_final_online_padding
            if args.task == "online"
            else None
        ),
        "official_60_to_30_geometry": (
            official_online_geometry if args.task == "online" else None
        ),
        "online_window_size": (
            args.online_window_size if args.task == "online" else None
        ),
        "peak_allocated_gib": peak_allocated_gib,
        "peak_total_allocated_gib": peak_total_allocated_gib,
        "postprocess_time": postprocess_time,
        "target_postprocess_batch_size": (
            args.target_postprocess_batch_size
        ),
        "target_postprocessing_calls": target_postprocessing_calls,
        "target_cache_results_pt": (
            str(args.target_cache_results_pt.resolve())
            if args.target_cache_results_pt is not None
            else None
        ),
        "target_cache_status": target_cache_status,
        "target_cache_source_checkpoint": (
            target_cache_metadata.get("checkpoint")
            if target_cache_metadata is not None
            else None
        ),
        "prediction_policy": (
            "anchor_plus_8_emotion_queries_plus_context_mixture"
        ),
        "metric_aliases": {
            "FRC": "FRCorr",
            "FRDiv": "official_smse",
            "TLCC": "FRSyn",
            "FRSyn_robust": "non_official_nan_safe_all_query_TLCC",
        },
        "protocol": (
            (
                "official_diffusion_online_window_test_frc_frdiv_frvar"
                if official_online_geometry
                and online_visible_speaker_history_frames
                == online_speaker_window_size
                else "causal_history_diagnostic_fixed_output_frc_frdiv_frvar"
            )
            if args.task == "online"
            else "official_diffusion_offline_test_frc_frdiv_frvar"
        ),
        "results_pt": (
            str(args.results_pt.resolve())
            if args.results_pt is not None
            else None
        ),
        "results_pt_status": results_pt_status,
        "task": args.task,
        "training_task": training_task,
        "training_supervision_tail_frames": supervision_tail_frames,
        "transfer_qualification": (
            (
                (
                    "official-window evaluation of a window-matched checkpoint"
                    if training_task == "online_window"
                    else "zero-shot official-window transfer of an "
                    "offline-trained checkpoint; no online-task fine-tuning"
                )
                if official_online_geometry
                and online_visible_speaker_history_frames
                == online_speaker_window_size
                else "strictly causal fixed-30-output history diagnostic; "
                "only speaker frames at or before the current output "
                "boundary are visible; no listener GT is a model input"
            )
            if args.task == "online"
            else None
        ),
        "residual_scale": args.residual_scale,
        "au_residual_multiplier": args.au_residual_multiplier,
        "va_residual_multiplier": args.va_residual_multiplier,
        "expression_residual_multiplier": (
            args.expression_residual_multiplier
        ),
        "style_residual_scale": (
            args.style_residual_scale
            if args.style_residual_scale is not None
            else args.residual_scale
        ),
        "seed": int(checkpoint["seed"]),
        "selection_seed": args.selection_seed,
        "split": "test",
        "semantic_query_top1_accuracy": sum(
            semantic_accuracy_values
        ) / len(semantic_accuracy_values),
        "semantic_query_profiles_E0_E7": mean_semantic_profile.tolist(),
        "mean_distinct_emotion_modes": sum(
            distinct_mode_values
        ) / len(distinct_mode_values),
        "mean_emotion_pairwise_l1": sum(
            emotion_l1_values
        ) / len(emotion_l1_values),
        "rounded_au_flip_rate": sum(au_flip_values) / len(au_flip_values),
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
