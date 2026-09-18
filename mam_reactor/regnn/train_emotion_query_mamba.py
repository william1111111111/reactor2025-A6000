from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from regnn.conditional_data import ConditionalReactionDataset
from regnn.conditional_model import (
    ConditionalREGNN,
    ConditionalREGNNConfig,
    restrict_valid_mask_to_tail,
)
from regnn.emotion_query_mamba import (
    EmotionQueryMambaConfig,
    EmotionQueryMambaLoss,
    EmotionQuerySequenceSetLoss,
    EmotionQuerySessionSetLoss,
    EmotionQueryResidualMamba,
    build_emotion_prototypes,
    masked_emotion_profile,
    target_set_emotion_support,
)
from regnn.query_mamba import (
    STYLE_DESCRIPTOR_DIM,
    STYLE_DESCRIPTOR_VERSION,
    apply_residual_channel_multipliers,
)
from regnn.train_conditional_regnn import online_rollout_span
from regnn.frd_surrogate import BandedSoftDTWFRDSurrogate
from regnn.va_calibration import (
    IsolatedVACalibrationConfig,
    isolated_va_objective,
    mask_output_head_gradients_to_va,
    restore_requires_grad,
    set_only_output_head_trainable,
)


class ViewAwareReplacementSampler:
    """Tag repeated weighted draws with deterministic per-epoch view ids."""

    def __init__(self, sampler: Iterable[int]):
        self.sampler = sampler
        self.last_draw_count = 0
        self.last_unique_index_count = 0
        self.last_repeated_draw_count = 0
        self.last_max_view_id = 0

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        view_counts: Dict[int, int] = {}
        tagged_indices = []
        for raw_index in self.sampler:
            index = int(raw_index)
            view_id = view_counts.get(index, 0)
            view_counts[index] = view_id + 1
            tagged_indices.append((index, view_id))
        self.last_draw_count = len(tagged_indices)
        self.last_unique_index_count = len(view_counts)
        self.last_repeated_draw_count = (
            self.last_draw_count - self.last_unique_index_count
        )
        self.last_max_view_id = max(
            (view_id for _, view_id in tagged_indices),
            default=0,
        )
        return iter(tagged_indices)

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train eight identifiable emotion queries and one context-mixture "
            "query over a frozen Conditional-REGNN."
        )
    )
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--style-cache", type=Path, required=True)
    parser.add_argument(
        "--require-style-cache-session-match",
        action="store_true",
        help="Require style-cache sessions and manifest hash to match training.",
    )
    parser.add_argument("--warmstart-query-checkpoint", type=Path)
    parser.add_argument("--warmstart-emotion-checkpoint", type=Path)
    parser.add_argument(
        "--warmstart-random-joint-checkpoint",
        type=Path,
        help=(
            "Continue a random_joint anchor+EQR path from a compatible "
            "checkpoint while rebuilding the optimizer."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--session-split-manifest", type=Path)
    parser.add_argument(
        "--session-split-name",
        choices=("B_fit", "B_cal", "B_confirm"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-tag", default="eqr_target_support")
    parser.add_argument(
        "--training-stage",
        choices=("full", "compatibility_head_only"),
        default="full",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=750)
    parser.add_argument(
        "--crop-stride",
        type=int,
        default=1,
        help="Restrict random source and target crop starts to this stride.",
    )
    parser.add_argument(
        "--supervision-tail-frames",
        type=int,
        default=0,
        help=(
            "Restrict every training objective to the final N frames while "
            "retaining the complete source window as context; zero keeps "
            "the offline full-window behavior."
        ),
    )
    parser.add_argument(
        "--online-rollout-blocks",
        type=int,
        default=1,
        help=(
            "Number of consecutive fixed-context online calls to stitch "
            "before sequence-set losses are computed. Each model call still "
            "receives exactly --clip-length frames."
        ),
    )
    parser.add_argument(
        "--allow-anchor-length-extension",
        action="store_true",
        help=(
            "Extend only the frozen anchor's non-persistent sinusoidal "
            "position encoding to --clip-length. Learned anchor weights "
            "still come from the shorter checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-anchor-length-shrink",
        action="store_true",
        help=(
            "Shrink only the frozen anchor's non-persistent sinusoidal "
            "position encoding to --clip-length. Learned anchor weights "
            "remain exactly those of the longer checkpoint."
        ),
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=10,
        help=(
            "Number of training target slots used by set-dependent losses. "
            "This does not change the fixed ten-query model output or the "
            "official evaluation target count."
        ),
    )
    parser.add_argument(
        "--target-selection-mode",
        choices=("deterministic_epoch_shuffle", "fixed_first"),
        default="deterministic_epoch_shuffle",
    )
    parser.add_argument(
        "--view-aware-replacement-crops",
        action="store_true",
        help=(
            "Give repeated WeightedRandomSampler draws distinct deterministic "
            "crop views while preserving sampled indices and target sets."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--anchor-lr",
        type=float,
        default=1e-4,
        help="Learning rate for a random jointly trained anchor.",
    )
    parser.add_argument(
        "--body-lr",
        type=float,
        help="Residual-body learning rate; defaults to --lr.",
    )
    parser.add_argument(
        "--compatibility-lr",
        type=float,
        help="Compatibility-head learning rate; defaults to --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--query-dim", type=int, default=64)
    parser.add_argument("--mamba-d-model", type=int, default=128)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--mamba-dropout", type=float, default=0.0)
    parser.add_argument(
        "--temporal-backbone",
        choices=("mamba", "mlp"),
        default="mamba",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=1034)
    parser.add_argument(
        "--anchor-initialization",
        choices=("pretrained_frozen", "random_joint"),
        default="pretrained_frozen",
        help=(
            "Use the pretrained frozen anchor, or use its architecture only "
            "and jointly train a random anchor end to end."
        ),
    )
    parser.add_argument("--max-residual-logit", type=float, default=1.5)
    parser.add_argument("--static-style-adapter", action="store_true")
    parser.add_argument("--style-adapter-hidden", type=int, default=128)
    parser.add_argument("--max-style-logit", type=float, default=0.75)
    parser.add_argument("--style-adapter-only", action="store_true")
    parser.add_argument(
        "--compatibility-mode",
        choices=("target_support", "legacy_relative_softmax"),
        default="target_support",
    )
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--compatibility-temperature",
        type=float,
        help="Deprecated alias for --gate-temperature.",
    )
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--gate-warmup-steps", type=int, default=500)
    parser.add_argument("--gate-top-m", type=int, default=0)
    parser.add_argument(
        "--mixture-mode",
        choices=("predicted_support", "uniform"),
        default="predicted_support",
    )
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--ccc-weight", type=float, default=0.5)
    parser.add_argument("--anchor-ccc-weight", type=float, default=10.0)
    parser.add_argument("--learned-set-ccc-weight", type=float, default=0.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--velocity-weight", type=float, default=0.05)
    parser.add_argument("--emotion-weight", type=float, default=0.25)
    parser.add_argument("--query-identity-weight", type=float, default=0.10)
    parser.add_argument(
        "--semantic-identity-mode",
        choices=("support_weighted", "uniform"),
        default="support_weighted",
    )
    parser.add_argument(
        "--semantic-identity-support-power",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--semantic-identity-min-support",
        type=float,
        default=0.0,
    )
    parser.add_argument("--compatibility-weight", type=float, default=0.10)
    parser.add_argument("--prototype-weight", type=float, default=0.05)
    parser.add_argument("--contrast-identity-weight", type=float, default=0.10)
    parser.add_argument("--contrast-prototype-weight", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=1.0)
    parser.add_argument("--diversity-margin", type=float, default=0.05)
    parser.add_argument("--style-diversity-weight", type=float, default=0.0)
    parser.add_argument("--style-diversity-margin", type=float, default=0.0)
    parser.add_argument("--centered-diversity-weight", type=float, default=0.0)
    parser.add_argument("--mixture-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--prototype-shrinkage", type=float, default=20.0)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--max-sample-weight", type=float, default=16.0)
    parser.add_argument("--train-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--train-style-residual-scale",
        type=float,
        help=(
            "Independent static-adapter scale used during training; "
            "defaults to --train-residual-scale."
        ),
    )
    parser.add_argument(
        "--session-set-training",
        action="store_true",
        help=(
            "Train all nine learned queries against nine real same-session "
            "reaction styles with a permutation-invariant set loss."
        ),
    )
    parser.add_argument(
        "--sequence-set-training",
        action="store_true",
        help=(
            "Train each candidate with official-aligned max CCC against ten "
            "cropped same-session reaction sequences."
        ),
    )
    parser.add_argument("--session-set-weight", type=float, default=1.0)
    parser.add_argument("--sinkhorn-temperature", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument("--sequence-transport-weight", type=float, default=0.0)
    parser.add_argument(
        "--sequence-transport-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--sequence-distance-mode",
        choices=("ccc", "banded_softdtw"),
        default="banded_softdtw",
    )
    parser.add_argument(
        "--set-matching-strategy",
        choices=(
            "balanced", "nearest", "paired", "row_softmax", "hard_hungarian"
        ),
        default="balanced",
        help=(
            "Balanced Sinkhorn, independent nearest-GT, or paired-target-only "
            "matching for every target-set-dependent objective."
        ),
    )
    parser.add_argument("--official-diversity-weight", type=float, default=0.0)
    parser.add_argument("--official-diversity-margin", type=float, default=0.0)
    parser.add_argument(
        "--official-diversity-batch-weight",
        type=float,
        default=0.0,
        help=(
            "Batch-mean FRDiv hinge weight. Unlike the legacy per-example "
            "hinge, this permits genuinely low-diversity contexts."
        ),
    )
    parser.add_argument("--channel-diversity-weight", type=float, default=0.0)
    parser.add_argument("--channel-diversity-budget", type=float, default=0.15)
    parser.add_argument(
        "--channel-diversity-no-cap",
        action="store_true",
        help=(
            "Use the full target-set FRDiv as the channel-diversity target; "
            "retain --channel-diversity-budget only as the normalization scale."
        ),
    )
    parser.add_argument("--train-au-residual-multiplier", type=float, default=1.0)
    parser.add_argument("--train-va-residual-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--train-expression-residual-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--isolated-va-calibration",
        action="store_true",
        help=(
            "After each main update, run a VA-only micro-update whose gradient "
            "is hard-masked to to_residual rows/biases 15:17."
        ),
    )
    parser.add_argument(
        "--isolated-va-start-epoch",
        type=int,
        default=3,
        help="One-indexed epoch at which isolated VA micro-updates begin.",
    )
    parser.add_argument("--isolated-va-lr", type=float, default=1e-5)
    parser.add_argument(
        "--isolated-va-tail-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--isolated-va-mean-risk-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--isolated-va-proximal-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--isolated-va-diversity-preservation-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--isolated-va-variance-preservation-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--isolated-va-preservation-ratio",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--isolated-va-gradient-clip",
        type=float,
        default=0.1,
    )
    parser.add_argument("--frd-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--frd-surrogate-target", type=float, default=160.0)
    parser.add_argument(
        "--frd-surrogate-mode",
        choices=("diagonal", "banded_softdtw"),
        default="banded_softdtw",
    )
    parser.add_argument("--frd-surrogate-stride", type=int, default=12)
    parser.add_argument(
        "--frd-surrogate-calibration",
        type=float,
        default=1.0,
        help=(
            "Validation-fitted scale from diagonal grouped distance to "
            "official FRD units. Keep 1.0 for a conservative upper bound."
        ),
    )
    parser.add_argument("--frd-softdtw-stride", type=int, default=6)
    parser.add_argument(
        "--frd-softdtw-band-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument("--frd-softdtw-gamma", type=float, default=0.1)
    parser.add_argument(
        "--frd-softdtw-pair-chunk-size",
        type=int,
        default=256,
    )
    parser.add_argument("--frd-tail-weight", type=float, default=0.0)
    parser.add_argument(
        "--frd-tail-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of the highest-FRD examples in each batch used by "
            "the optional CVaR-style hinge."
        ),
    )
    parser.add_argument("--frd-query-excess-weight", type=float, default=0.0)
    parser.add_argument(
        "--frd-query-excess-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of the nine learned queries with the largest FRD "
            "excess over the frozen anchor to penalize per example."
        ),
    )
    parser.add_argument("--frd-transport-weight", type=float, default=0.0)
    parser.add_argument(
        "--frd-transport-temperature",
        type=float,
        default=0.1,
        help=(
            "Sinkhorn temperature after normalizing each FRD pair cost by "
            "the per-query target."
        ),
    )
    parser.add_argument("--frd-transport-iterations", type=int, default=32)
    parser.add_argument("--sync-weight", type=float, default=0.0)
    parser.add_argument("--sync-max-lag", type=int, default=48)
    parser.add_argument("--sync-lag-step", type=int, default=12)
    parser.add_argument("--sync-temporal-stride", type=int, default=4)
    parser.add_argument("--sync-every-n-batches", type=int, default=1)
    parser.add_argument("--sync-temperature", type=float, default=0.05)
    parser.add_argument(
        "--sync-variance-epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=25)
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
        help="Stop after this many optimizer steps across epochs; zero disables.",
    )
    parser.add_argument("--listener-3dmm-weight", type=float, default=0.0)
    parser.add_argument("--listener-3dmm-velocity-weight", type=float, default=0.1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_session_allowlist(
    manifest_path: Path | None,
    split_name: str | None,
) -> tuple[list[str] | None, str | None]:
    if (manifest_path is None) != (split_name is None):
        raise ValueError(
            "session-split-manifest and session-split-name must be set together"
        )
    if manifest_path is None:
        return None, None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = payload.get("splits")
    if not isinstance(split_payload, dict):
        raise ValueError("Session manifest must contain a splits object")
    required = ("B_fit", "B_cal", "B_confirm")
    if any(name not in split_payload for name in required):
        raise ValueError("Session manifest is missing a required split")
    normalized = {
        name: [str(session) for session in split_payload[name]]
        for name in required
    }
    flattened = [session for name in required for session in normalized[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Session manifest contains cross-split leakage")
    selected = normalized[str(split_name)]
    if not selected:
        raise ValueError(f"Session split {split_name} is empty")
    return sorted(selected), sha256_file(manifest_path)


def forward_online_sequence_set_rollout(
    model: EmotionQueryResidualMamba,
    speaker_audio: torch.Tensor,
    speaker_emotion: torch.Tensor,
    speaker_3dmm: torch.Tensor,
    lengths: torch.Tensor,
    *,
    context_frames: int,
    output_frames: int,
    rollout_blocks: int,
    residual_scale: float,
    style_residual_scale: float,
) -> Dict[str, torch.Tensor]:
    """Run fixed online windows and stitch their retained output tails.

    Per-window condition-level quantities are averaged across the rollout;
    temporal tensors keep block order and become one contiguous output stream.
    No call receives more than ``context_frames`` speaker frames.
    """
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
    batch_size = speaker_audio.shape[0]
    if lengths.shape != (batch_size,):
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
    window_output = model(
        windows(speaker_audio),
        windows(speaker_emotion),
        windows(speaker_3dmm),
        window_lengths.clamp_min(1),
        residual_scale=residual_scale,
        style_residual_scale=style_residual_scale,
    )

    query_temporal_keys = {
        "prediction",
        "raw_prediction",
        "residual_raw",
        "dynamic_residual_raw",
        "style_residual_raw",
    }
    base_temporal_keys = {
        "base_prediction",
        "base_raw_prediction",
        "context",
    }
    stitched: Dict[str, torch.Tensor] = {}
    for key, value in window_output.items():
        if not isinstance(value, torch.Tensor):
            stitched[key] = value
        elif key in query_temporal_keys:
            reshaped = value.reshape(
                batch_size,
                rollout_blocks,
                *value.shape[1:],
            )
            stitched[key] = (
                reshaped[:, :, :, -output_frames:]
                .permute(0, 2, 1, 3, 4)
                .flatten(2, 3)
            )
        elif key in base_temporal_keys:
            reshaped = value.reshape(
                batch_size,
                rollout_blocks,
                *value.shape[1:],
            )
            stitched[key] = reshaped[:, :, -output_frames:].flatten(1, 2)
        elif key == "valid_mask":
            stitched[key] = (
                value.reshape(batch_size, rollout_blocks, context_frames)
                [:, :, -output_frames:]
                .flatten(1, 2)
            )
        elif value.ndim > 0 and value.shape[0] == batch_size * rollout_blocks:
            stitched[key] = value.reshape(
                batch_size,
                rollout_blocks,
                *value.shape[1:],
            ).mean(dim=1)
        else:
            stitched[key] = value
    return stitched


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_anchor_config_for_clip_length(
    checkpoint_payload: Dict[str, object],
    clip_length: int,
    allow_length_extension: bool,
    allow_length_shrink: bool = False,
) -> tuple[ConditionalREGNNConfig, int, bool]:
    """Resolve an optional sinusoidal-only frozen-anchor length resize."""
    checkpoint_config = ConditionalREGNNConfig(**checkpoint_payload)
    checkpoint_max_seq_len = checkpoint_config.max_seq_len
    if checkpoint_max_seq_len == clip_length:
        return checkpoint_config, checkpoint_max_seq_len, False
    if (
        clip_length > checkpoint_max_seq_len
        and not allow_length_extension
    ):
        raise ValueError(
            "Anchor is shorter than the training clip; enable length "
            "extension"
        )
    if (
        clip_length < checkpoint_max_seq_len
        and not allow_length_shrink
    ):
        raise ValueError(
            "Anchor is longer than the training clip; enable length shrink"
        )
    return (
        replace(checkpoint_config, max_seq_len=clip_length),
        checkpoint_max_seq_len,
        True,
    )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_model_config(
    args: argparse.Namespace,
) -> EmotionQueryMambaConfig:
    gate_temperature = (
        args.compatibility_temperature
        if args.compatibility_temperature is not None
        else args.gate_temperature
    )
    return EmotionQueryMambaConfig(
        query_dim=args.query_dim,
        d_model=args.mamba_d_model,
        d_state=args.mamba_d_state,
        d_conv=args.mamba_d_conv,
        expand=args.mamba_expand,
        layers=args.mamba_layers,
        dropout=args.mamba_dropout,
        max_residual_logit=args.max_residual_logit,
        bidirectional=not args.unidirectional,
        temporal_backbone=args.temporal_backbone,
        mlp_hidden_dim=args.mlp_hidden_dim,
        compatibility_mode=args.compatibility_mode,
        compatibility_temperature=gate_temperature,
        gate_temperature=gate_temperature,
        gate_floor=args.gate_floor,
        gate_warmup_steps=args.gate_warmup_steps,
        gate_top_m=args.gate_top_m,
        mixture_mode=args.mixture_mode,
        static_style_adapter=args.static_style_adapter,
        style_adapter_hidden=args.style_adapter_hidden,
        max_style_logit=args.max_style_logit,
        listener_3dmm=args.listener_3dmm_weight > 0,
    )


def load_shared_query_warmstart(
    model: EmotionQueryResidualMamba,
    checkpoint_path: Path,
) -> list[str]:
    """Load only the already-trained shared residual generator."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["state_dict"]
    destination_state = model.state_dict()
    excluded_prefixes = (
        "anchor.",
        "query_embeddings",
        "emotion_embeddings",
        "compatibility_head.",
    )
    shared_state = {
        name: value
        for name, value in source_state.items()
        if not name.startswith(excluded_prefixes)
        and name in destination_state
        and destination_state[name].shape == value.shape
    }
    if not shared_state:
        raise RuntimeError("No compatible shared Query-Mamba weights found")
    model.load_state_dict(shared_state, strict=False)
    return sorted(shared_state)


def load_emotion_warmstart(
    model: EmotionQueryResidualMamba,
    checkpoint_path: Path,
) -> tuple[list[str], str]:
    """Load a warm-start without importing incompatible gate semantics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_method = checkpoint.get("method")
    if checkpoint_method not in {
        "Emotion-Gated-Query-Mamba",
        "EQR-Mamba",
    }:
        raise ValueError("Emotion warmstart checkpoint has wrong method")
    imports_legacy_compatibility = (
        checkpoint_method == "Emotion-Gated-Query-Mamba"
        and model.config.compatibility_mode == "legacy_relative_softmax"
    )
    excluded_prefixes = (
        ("anchor.",)
        if imports_legacy_compatibility
        else (
            ("anchor.", "compatibility_head.")
            if checkpoint_method == "Emotion-Gated-Query-Mamba"
            else ("anchor.",)
        )
    )
    source_state = checkpoint["state_dict"]
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(excluded_prefixes)
    }
    adapter_names = {
        name
        for name in trainable_names
        if name.startswith(("static_style_adapter.", "to_3dmm."))
    }
    missing = sorted(trainable_names - source_state.keys() - adapter_names)
    mismatched = sorted(
        name
        for name in trainable_names & source_state.keys()
        if model.state_dict()[name].shape != source_state[name].shape
    )
    if missing or mismatched:
        raise RuntimeError(
            "Emotion warmstart is incompatible: "
            f"missing={missing}, mismatched={mismatched}"
        )
    trainable_state = {
        name: source_state[name]
        for name in trainable_names & source_state.keys()
    }
    model.load_state_dict(trainable_state, strict=False)
    return sorted(trainable_state), str(checkpoint_method)


def load_random_joint_warmstart(
    model: EmotionQueryResidualMamba,
    checkpoint_path: Path,
) -> tuple[list[str], str]:
    """Load every parameter from a compatible random-joint EQR checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_method = checkpoint.get("method")
    if checkpoint_method != "EQR-Mamba":
        raise ValueError("Random-joint warmstart checkpoint has wrong method")
    if checkpoint.get("anchor_initialization") != "random_joint":
        raise ValueError(
            "Random-joint warmstart requires a random_joint source checkpoint"
        )
    checkpoint_backbone = checkpoint.get("temporal_backbone", "mamba")
    if checkpoint_backbone != model.config.temporal_backbone:
        raise ValueError(
            "Random-joint warmstart temporal backbone mismatch: "
            f"source={checkpoint_backbone}, "
            f"destination={model.config.temporal_backbone}"
        )

    source_state = checkpoint["state_dict"]
    destination_state = model.state_dict()
    parameter_names = set(dict(model.named_parameters()))
    missing = sorted(parameter_names - source_state.keys())
    mismatched = sorted(
        name
        for name in parameter_names & source_state.keys()
        if destination_state[name].shape != source_state[name].shape
    )
    if missing or mismatched:
        raise RuntimeError(
            "Random-joint warmstart is incompatible: "
            f"missing={missing}, mismatched={mismatched}"
        )
    parameter_state = {
        name: source_state[name] for name in parameter_names
    }
    model.load_state_dict(parameter_state, strict=False)
    return sorted(parameter_state), str(checkpoint_method)


def initialize_compatibility_head_from_prior(
    model: EmotionQueryResidualMamba,
    support_prior: torch.Tensor,
) -> None:
    """Initialize independent Bernoulli logits from target-set prevalence."""
    support_prior = torch.as_tensor(
        support_prior,
        dtype=model.compatibility_head[-1].bias.dtype,
        device=model.compatibility_head[-1].bias.device,
    )
    if support_prior.shape != (8,):
        raise ValueError("support_prior must have shape [8]")
    support_prior = support_prior.clamp(1e-4, 1.0 - 1e-4)
    with torch.no_grad():
        torch.nn.init.zeros_(model.compatibility_head[-1].weight)
        model.compatibility_head[-1].bias.copy_(torch.logit(support_prior))


def freeze_except_compatibility_head(
    model: EmotionQueryResidualMamba,
) -> list[str]:
    """Expose only the context-to-support predictor for Stage A."""
    trainable_names = []
    for name, parameter in model.named_parameters():
        is_compatibility = name.startswith("compatibility_head.")
        parameter.requires_grad_(is_compatibility)
        if is_compatibility:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("Compatibility head has no trainable parameters")
    return sorted(trainable_names)


def freeze_except_static_style_adapter(
    model: EmotionQueryResidualMamba,
) -> list[str]:
    """Freeze the temporal generator and expose only the static adapter."""
    trainable_names = []
    for name, parameter in model.named_parameters():
        is_adapter = name.startswith("static_style_adapter.")
        parameter.requires_grad_(is_adapter)
        if is_adapter:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("Static style adapter has no trainable parameters")
    return sorted(trainable_names)


def validate_anchor_gradient_contract(
    model: EmotionQueryResidualMamba,
) -> None:
    anchor_gradients = [
        parameter.grad
        for parameter in model.anchor.parameters()
        if parameter.grad is not None
    ]
    if model.freeze_anchor and anchor_gradients:
        raise RuntimeError("Frozen anchor received a gradient")
    if not model.freeze_anchor and not anchor_gradients:
        raise RuntimeError("Jointly trained anchor received no gradient")


def build_sampling_weights(
    dataset: ConditionalReactionDataset,
    cache_paths: list[str],
    descriptors: torch.Tensor,
    balance_power: float,
    max_sample_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if balance_power < 0.0:
        raise ValueError("class-balance-power must be non-negative")
    if max_sample_weight < 1.0:
        raise ValueError("max-sample-weight must be at least one")
    descriptor_lookup = {
        path: descriptor
        for path, descriptor in zip(cache_paths, descriptors)
    }
    labels = []
    for record in dataset.records:
        target_path = dataset._paired_target(record).as_posix()
        if target_path not in descriptor_lookup:
            raise KeyError(f"Missing style descriptor for {target_path}")
        labels.append(
            int(descriptor_lookup[target_path][17:25].argmax())
        )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=8)
    safe_counts = counts.clamp_min(1).float()
    class_weights = (
        safe_counts.max() / safe_counts
    ).pow(balance_power).clamp(max=max_sample_weight)
    return class_weights[labels_tensor], labels_tensor, counts


def stage_a_sampling_protocol(
    *,
    dataset_size: int,
    class_balance_power: float,
    max_sample_weight: float,
    sample_weights: torch.Tensor,
) -> Dict[str, object]:
    """Validate and describe probability-calibrated Stage A sampling."""
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if class_balance_power != 0.0 or max_sample_weight != 1.0:
        raise ValueError(
            "Stage A requires class_balance_power=0 and "
            "max_sample_weight=1"
        )
    if sample_weights.shape != (dataset_size,):
        raise ValueError("sample_weights must have shape [dataset_size]")
    if not torch.equal(sample_weights, torch.ones_like(sample_weights)):
        raise ValueError("Stage A sample weights must all equal one")
    return {
        "sampler_type": "RandomSampler",
        "stage_a_sampling_distribution": "uniform_over_examples",
        "class_balance_power": float(class_balance_power),
        "max_sample_weight": float(max_sample_weight),
        "replacement_sampling": False,
        "shuffle": True,
        "drop_last": False,
        "num_samples_per_epoch": int(dataset_size),
        "per_example_draw_probability": 1.0 / float(dataset_size),
        "per_example_epoch_inclusion_probability": 1.0,
        "sample_weight_min": 1.0,
        "sample_weight_max": 1.0,
        "sample_weight_mean": 1.0,
        "importance_correction": False,
        "positive_class_weighting": False,
        "compatibility_bce_weighting": "unweighted",
    }


def validate_stage_a_calibration_protocol(args: argparse.Namespace) -> None:
    """Enforce B_fit-only, unweighted Stage A probability calibration."""
    if args.training_stage != "compatibility_head_only":
        return
    if args.session_split_manifest is None or args.session_split_name != "B_fit":
        raise ValueError(
            "Stage A support calibration and prior initialization are "
            "restricted to an explicit session-disjoint B_fit split"
        )
    if not args.sequence_set_training:
        raise ValueError(
            "Compatibility-head-only calibration requires "
            "--sequence-set-training"
        )
    if args.compatibility_mode != "target_support":
        raise ValueError(
            "Compatibility-head-only calibration requires target_support"
        )
    if args.style_adapter_only:
        raise ValueError(
            "Compatibility-head-only and style-adapter-only are exclusive"
        )
    if args.class_balance_power != 0.0 or args.max_sample_weight != 1.0:
        raise ValueError(
            "Stage A probability calibration requires "
            "--class-balance-power 0 and --max-sample-weight 1"
        )


def estimate_target_support_prior(
    dataset: ConditionalReactionDataset,
    cache_paths: list[str],
    descriptors: torch.Tensor,
) -> torch.Tensor:
    """Estimate epoch-zero target-set support prevalence without loading clips."""
    descriptor_lookup = {
        path: descriptor.float()
        for path, descriptor in zip(cache_paths, descriptors)
    }
    supports = []
    for record in dataset.records:
        target_paths, _, available = dataset._session_target_selection(record)
        profiles = []
        for target_path, is_available in zip(target_paths, available):
            if not is_available:
                continue
            key = target_path.as_posix()
            if key not in descriptor_lookup:
                raise KeyError(f"Missing style descriptor for {key}")
            profile = descriptor_lookup[key][17:25].clamp_min(0.0)
            profiles.append(profile / profile.sum().clamp_min(1e-8))
        if profiles:
            supports.append(torch.stack(profiles).amax(dim=0))
    if not supports:
        raise RuntimeError("Cannot estimate target support prior from empty sets")
    return torch.stack(supports).mean(dim=0).clamp(1e-4, 1.0 - 1e-4)


def compatibility_calibration_loss(
    binary_logits: torch.Tensor,
    support_probs: torch.Tensor,
    target_support: torch.Tensor,
    valid_examples: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """BCE objective and calibration diagnostics for Stage A."""
    if binary_logits.shape != target_support.shape or support_probs.shape != (
        target_support.shape
    ):
        raise ValueError("Compatibility tensors must share shape [B,8]")
    if valid_examples.shape != (target_support.shape[0],):
        raise ValueError("valid_examples must have shape [B]")
    weights = valid_examples.to(binary_logits)[:, None]
    denominator = weights.sum().clamp_min(1.0)
    per_entry_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        binary_logits.float(),
        target_support.float(),
        reduction="none",
    )
    per_entry_brier = (
        support_probs.float() - target_support.float()
    ).square()
    per_emotion_bce = (per_entry_bce * weights).sum(dim=0) / denominator
    per_emotion_brier = (per_entry_brier * weights).sum(dim=0) / denominator
    target_mean = (target_support * weights).sum(dim=0) / denominator
    prediction_mean = (support_probs * weights).sum(dim=0) / denominator
    active_025 = (
        (support_probs >= 0.25).to(support_probs) * weights
    ).sum(dim=-1).sum() / denominator
    active_050 = (
        (support_probs >= 0.50).to(support_probs) * weights
    ).sum(dim=-1).sum() / denominator
    result = {
        "loss": per_emotion_bce.mean(),
        "compatibility_bce": per_emotion_bce.mean(),
        "compatibility_brier": per_emotion_brier.mean(),
        "compatibility_mean": prediction_mean.mean(),
        "compatibility_min": support_probs.amin(),
        "compatibility_max": support_probs.amax(),
        "active_query_count_025": active_025,
        "active_query_count_050": active_050,
    }
    for emotion_index in range(8):
        result[f"compatibility_bce_e{emotion_index}"] = per_emotion_bce[
            emotion_index
        ]
        result[f"compatibility_brier_e{emotion_index}"] = per_emotion_brier[
            emotion_index
        ]
        result[f"target_support_e{emotion_index}"] = target_mean[emotion_index]
        result[f"predicted_support_e{emotion_index}"] = prediction_mean[
            emotion_index
        ]
    return result


def compatibility_epoch_calibration_metrics(
    support_probs: torch.Tensor,
    target_support: torch.Tensor,
) -> Dict[str, object]:
    """Compute sample-weighted Stage A metrics over a complete epoch."""
    if (
        support_probs.ndim != 2
        or support_probs.shape[1] != 8
        or support_probs.shape != target_support.shape
        or support_probs.shape[0] == 0
    ):
        raise ValueError(
            "epoch support probabilities and targets must share non-empty [N,8]"
        )
    probabilities = support_probs.float()
    targets = target_support.float()
    exact_bce = torch.nn.functional.binary_cross_entropy(
        probabilities.clamp(1e-7, 1.0 - 1e-7),
        targets,
        reduction="none",
    )
    exact_brier = (probabilities - targets).square()
    metrics: Dict[str, object] = {
        "compatibility_calibration_num_examples": int(probabilities.shape[0]),
        "compatibility_bce": float(exact_bce.mean()),
        "compatibility_brier": float(exact_brier.mean()),
        "compatibility_mean": float(probabilities.mean()),
        "compatibility_min": float(probabilities.min()),
        "compatibility_max": float(probabilities.max()),
        "active_query_count_025": float(
            (probabilities >= 0.25).float().sum(dim=-1).mean()
        ),
        "active_query_count_050": float(
            (probabilities >= 0.50).float().sum(dim=-1).mean()
        ),
    }
    for emotion_index in range(8):
        metrics[f"compatibility_bce_e{emotion_index}"] = float(
            exact_bce[:, emotion_index].mean()
        )
        metrics[f"compatibility_brier_e{emotion_index}"] = float(
            exact_brier[:, emotion_index].mean()
        )
        metrics[f"target_support_e{emotion_index}"] = float(
            targets[:, emotion_index].mean()
        )
        metrics[f"predicted_support_e{emotion_index}"] = float(
            probabilities[:, emotion_index].mean()
        )
    return metrics


def _average_ranks(values: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    del unique
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    average = 0.5 * (starts + cumulative - 1)
    return average[inverse].astype(np.float64)


def spearman_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Tie-aware Spearman correlation with a finite constant-vector fallback."""
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("Spearman inputs must share one-dimensional shape")
    rank_a = _average_ranks(values_a)
    rank_b = _average_ranks(values_b)
    centered_a = rank_a - rank_a.mean()
    centered_b = rank_b - rank_b.mean()
    denominator = np.sqrt(
        np.square(centered_a).sum() * np.square(centered_b).sum()
    )
    if denominator <= 1e-12:
        return 0.0
    return float((centered_a * centered_b).sum() / denominator)


def save_checkpoint(
    path: Path,
    model: EmotionQueryResidualMamba,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    anchor_sha256: str,
    style_cache_sha256: str,
    warmstart_sha256: str | None,
    emotion_warmstart_sha256: str | None,
    prototype_counts: torch.Tensor,
    emotion_priors: torch.Tensor,
    target_support_prior: torch.Tensor,
    emotion_warmstart_method: str | None,
    session_allowlist: list[str] | None,
    session_split_manifest_sha256: str | None,
    sampling_protocol: Dict[str, object],
    epoch_metrics: Dict[str, object],
    isolated_va_optimizer: torch.optim.Optimizer | None = None,
    random_joint_warmstart_sha256: str | None = None,
    random_joint_warmstart_method: str | None = None,
) -> None:
    temporal_backbone = getattr(
        args,
        "temporal_backbone",
        getattr(model.config, "temporal_backbone", "mamba"),
    )
    anchor_initialization = getattr(
        args,
        "anchor_initialization",
        "pretrained_frozen",
    )
    predicted_support_keys = [
        f"predicted_support_e{emotion_index}" for emotion_index in range(8)
    ]
    target_support_keys = [
        f"target_support_e{emotion_index}" for emotion_index in range(8)
    ]
    compatibility_statistics = None
    if all(key in epoch_metrics for key in predicted_support_keys):
        predicted_support_mean = [
            float(epoch_metrics[key]) for key in predicted_support_keys
        ]
        target_support_mean = (
            [float(epoch_metrics[key]) for key in target_support_keys]
            if all(key in epoch_metrics for key in target_support_keys)
            else None
        )
        compatibility_statistics = {
            "split_name": args.session_split_name,
            "sampling_distribution": sampling_protocol.get(
                "stage_a_sampling_distribution"
            ),
            "sampler_type": sampling_protocol.get("sampler_type"),
            "predicted_support_mean_E0_E7": predicted_support_mean,
            "predicted_support_global_mean": float(
                np.mean(predicted_support_mean)
            ),
            "target_support_mean_E0_E7": target_support_mean,
            "num_examples": epoch_metrics.get(
                "compatibility_calibration_num_examples"
            ),
            "compatibility_bce": epoch_metrics.get("compatibility_bce"),
            "compatibility_brier": epoch_metrics.get("compatibility_brier"),
        }
    atomic_torch_save(
        {
            "method": "EQR-Mamba",
            "method_full_name": (
                "Emotion-Query Residual Mamba"
                if temporal_backbone == "mamba"
                else "Emotion-Query Residual Frame-MLP"
            ),
            "temporal_backbone": temporal_backbone,
            "epoch": epoch,
            "seed": args.seed,
            "emotion_channels": [f"E{index}" for index in range(8)],
            "query_policy": {
                "q0": (
                    "frozen_pretrained_conditional_anchor"
                    if anchor_initialization == "pretrained_frozen"
                    else "random_jointly_trained_conditional_anchor"
                ),
                "q1_q8": "target_supported_emotion_channels_E0_E7",
                "q9": (
                    "normalized_predicted_support_mixture"
                    if args.mixture_mode == "predicted_support"
                    else "uniform_emotion_mixture"
                ),
            },
            "anchor_checkpoint": str(args.anchor_checkpoint),
            "anchor_template_checkpoint": str(args.anchor_checkpoint),
            "anchor_sha256": anchor_sha256,
            "anchor_initialization": anchor_initialization,
            "anchor_weights_loaded": (
                anchor_initialization == "pretrained_frozen"
            ),
            "style_cache": str(args.style_cache),
            "style_cache_sha256": style_cache_sha256,
            "warmstart_query_checkpoint": (
                str(args.warmstart_query_checkpoint)
                if args.warmstart_query_checkpoint is not None
                else None
            ),
            "warmstart_query_sha256": warmstart_sha256,
            "warmstart_emotion_checkpoint": (
                str(args.warmstart_emotion_checkpoint)
                if args.warmstart_emotion_checkpoint is not None
                else None
            ),
            "warmstart_emotion_sha256": emotion_warmstart_sha256,
            "warmstart_emotion_method": emotion_warmstart_method,
            "warmstart_random_joint_checkpoint": (
                str(args.warmstart_random_joint_checkpoint)
                if getattr(
                    args,
                    "warmstart_random_joint_checkpoint",
                    None,
                ) is not None
                else None
            ),
            "warmstart_random_joint_sha256": (
                random_joint_warmstart_sha256
            ),
            "warmstart_random_joint_method": (
                random_joint_warmstart_method
            ),
            "prototype_counts": prototype_counts.tolist(),
            "emotion_priors": emotion_priors.tolist(),
            "target_support_prior": target_support_prior.tolist(),
            "session_split_name": args.session_split_name,
            "session_allowlist": session_allowlist,
            "session_split_manifest_sha256": session_split_manifest_sha256,
            "sampling_protocol": sampling_protocol,
            "compatibility_statistics": compatibility_statistics,
            "anchor_model_config": asdict(model.anchor.config),
            "emotion_query_model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "isolated_va_optimizer": (
                isolated_va_optimizer.state_dict()
                if isolated_va_optimizer is not None
                else None
            ),
            "scheduler": scheduler.state_dict(),
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.listener_3dmm_weight < 0 or args.listener_3dmm_velocity_weight < 0:
        raise ValueError("3DMM loss weights must be nonnegative")
    if args.listener_3dmm_weight > 0:
        if not args.sequence_set_training or args.online_rollout_blocks != 1 or args.training_stage != "full" or args.style_adapter_only:
            raise ValueError("Joint 3DMM currently requires offline sequence-set full training")
        from regnn.joint_3dmm import listener_3dmm_loss
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.crop_stride <= 0:
        raise ValueError("crop-stride must be positive")
    if not 0 <= args.supervision_tail_frames <= args.clip_length:
        raise ValueError("supervision-tail-frames must be in [0,clip-length]")
    if args.online_rollout_blocks <= 0:
        raise ValueError("online-rollout-blocks must be positive")
    if args.online_rollout_blocks > 1 and (
        not args.sequence_set_training or args.supervision_tail_frames <= 0
    ):
        raise ValueError(
            "Multi-block online rollout requires sequence-set training and "
            "a positive supervision tail"
        )
    dataset_clip_length = online_rollout_span(
        args.clip_length,
        args.supervision_tail_frames,
        args.online_rollout_blocks,
    )
    if args.num_targets <= 0:
        raise ValueError("num-targets must be positive")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.anchor_lr <= 0.0:
        raise ValueError("anchor-lr must be positive")
    if args.body_lr is not None and args.body_lr <= 0.0:
        raise ValueError("body-lr must be positive")
    if args.compatibility_lr is not None and args.compatibility_lr <= 0.0:
        raise ValueError("compatibility-lr must be positive")
    if args.max_train_steps < 0:
        raise ValueError("max-train-steps must be non-negative")
    if args.mlp_hidden_dim <= 0:
        raise ValueError("mlp-hidden-dim must be positive")
    if args.save_every <= 0 or args.print_every <= 0:
        raise ValueError("save-every and print-every must be positive")
    if args.train_residual_scale < 0.0:
        raise ValueError("train-residual-scale must be non-negative")
    effective_gate_temperature = (
        args.compatibility_temperature
        if args.compatibility_temperature is not None
        else args.gate_temperature
    )
    if effective_gate_temperature <= 0.0:
        raise ValueError("gate-temperature must be positive")
    if not 0.0 <= args.gate_floor <= 1.0:
        raise ValueError("gate-floor must be in [0,1]")
    if args.gate_warmup_steps < 0:
        raise ValueError("gate-warmup-steps must be non-negative")
    if not 0 <= args.gate_top_m <= 8:
        raise ValueError("gate-top-m must be in [0,8]")
    if args.anchor_ccc_weight < 0.0 or args.learned_set_ccc_weight < 0.0:
        raise ValueError("CCC weights must be non-negative")
    if args.semantic_identity_support_power <= 0.0:
        raise ValueError("semantic-identity-support-power must be positive")
    if not 0.0 <= args.semantic_identity_min_support <= 1.0:
        raise ValueError("semantic-identity-min-support must be in [0,1]")
    if (
        args.train_style_residual_scale is not None
        and args.train_style_residual_scale < 0.0
    ):
        raise ValueError("train-style-residual-scale must be non-negative")
    if args.session_set_weight < 0.0:
        raise ValueError("session-set-weight must be non-negative")
    if args.sinkhorn_temperature <= 0.0 or args.sinkhorn_iterations <= 0:
        raise ValueError("Invalid Sinkhorn configuration")
    if args.sequence_transport_temperature <= 0.0:
        raise ValueError("Invalid sequence Sinkhorn temperature")
    if args.session_set_training and args.sequence_set_training:
        raise ValueError(
            "Choose only one of session-set or sequence-set training"
        )
    if (
        args.set_matching_strategy != "balanced"
        and not args.sequence_set_training
    ):
        raise ValueError(
            "Non-default set matching requires --sequence-set-training"
        )
    validate_stage_a_calibration_protocol(args)
    if args.diversity_margin < 0.0:
        raise ValueError("diversity-margin must be non-negative")
    if args.style_diversity_margin < 0.0:
        raise ValueError("style-diversity-margin must be non-negative")
    if args.official_diversity_margin < 0.0:
        raise ValueError("official-diversity-margin must be non-negative")
    if args.official_diversity_batch_weight < 0.0:
        raise ValueError(
            "official-diversity-batch-weight must be non-negative"
        )
    if args.channel_diversity_weight < 0.0:
        raise ValueError("channel-diversity-weight must be non-negative")
    if args.channel_diversity_budget <= 0.0:
        raise ValueError("channel-diversity-budget must be positive")
    if min(
        args.train_au_residual_multiplier,
        args.train_va_residual_multiplier,
        args.train_expression_residual_multiplier,
    ) < 0.0:
        raise ValueError("Train residual channel multipliers must be non-negative")
    va_calibration_config = IsolatedVACalibrationConfig(
        tail_fraction=args.isolated_va_tail_fraction,
        mean_risk_weight=args.isolated_va_mean_risk_weight,
        proximal_weight=args.isolated_va_proximal_weight,
        diversity_preservation_weight=(
            args.isolated_va_diversity_preservation_weight
        ),
        variance_preservation_weight=(
            args.isolated_va_variance_preservation_weight
        ),
        preservation_ratio=args.isolated_va_preservation_ratio,
    )
    va_calibration_config.validate()
    if args.isolated_va_calibration:
        if args.training_stage != "full" or not args.sequence_set_training:
            raise ValueError(
                "Isolated VA calibration requires full sequence-set training"
            )
        if args.session_split_name != "B_fit":
            raise ValueError(
                "Isolated VA calibration is restricted to session split B_fit"
            )
        if not 1 <= args.isolated_va_start_epoch <= args.epochs:
            raise ValueError(
                "isolated-va-start-epoch must be within the requested epochs"
            )
        if args.isolated_va_lr <= 0.0:
            raise ValueError("isolated-va-lr must be positive")
        if args.isolated_va_gradient_clip <= 0.0:
            raise ValueError("isolated-va-gradient-clip must be positive")
    if args.frd_surrogate_weight < 0.0:
        raise ValueError("frd-surrogate-weight must be non-negative")
    if args.frd_surrogate_target <= 0.0:
        raise ValueError("frd-surrogate-target must be positive")
    if args.frd_surrogate_stride <= 0:
        raise ValueError("frd-surrogate-stride must be positive")
    if args.frd_surrogate_calibration <= 0.0:
        raise ValueError("frd-surrogate-calibration must be positive")
    if args.frd_softdtw_stride <= 0:
        raise ValueError("frd-softdtw-stride must be positive")
    if not 0.0 <= args.frd_softdtw_band_ratio <= 1.0:
        raise ValueError("frd-softdtw-band-ratio must be in [0,1]")
    if args.frd_softdtw_gamma <= 0.0:
        raise ValueError("frd-softdtw-gamma must be positive")
    if args.frd_softdtw_pair_chunk_size <= 0:
        raise ValueError("frd-softdtw-pair-chunk-size must be positive")
    if args.frd_tail_weight < 0.0:
        raise ValueError("frd-tail-weight must be non-negative")
    if not 0.0 < args.frd_tail_fraction <= 1.0:
        raise ValueError("frd-tail-fraction must be in (0, 1]")
    if args.frd_query_excess_weight < 0.0:
        raise ValueError("frd-query-excess-weight must be non-negative")
    if not 0.0 < args.frd_query_excess_fraction <= 1.0:
        raise ValueError("frd-query-excess-fraction must be in (0, 1]")
    if args.frd_transport_weight < 0.0:
        raise ValueError("frd-transport-weight must be non-negative")
    if args.frd_transport_temperature <= 0.0:
        raise ValueError("frd-transport-temperature must be positive")
    if args.frd_transport_iterations <= 0:
        raise ValueError("frd-transport-iterations must be positive")
    if args.sync_weight < 0.0:
        raise ValueError("sync-weight must be non-negative")
    if args.sync_max_lag <= 0:
        raise ValueError("sync-max-lag must be positive")
    if (
        args.sync_lag_step <= 0
        or args.sync_max_lag % args.sync_lag_step != 0
    ):
        raise ValueError(
            "sync-lag-step must be positive and divide sync-max-lag"
        )
    if args.sync_temperature <= 0.0:
        raise ValueError("sync-temperature must be positive")
    if args.sync_temporal_stride <= 0:
        raise ValueError("sync-temporal-stride must be positive")
    if args.sync_every_n_batches <= 0:
        raise ValueError("sync-every-n-batches must be positive")
    if args.sync_variance_epsilon < 0.0:
        raise ValueError("sync-variance-epsilon must be non-negative")
    if args.sync_weight > 0.0 and not args.sequence_set_training:
        raise ValueError("sync loss requires --sequence-set-training")
    if (
        args.frd_surrogate_weight > 0.0
        or args.frd_tail_weight > 0.0
        or args.frd_query_excess_weight > 0.0
        or args.frd_transport_weight > 0.0
    ) and not args.sequence_set_training:
        raise ValueError(
            "FRD surrogate requires --sequence-set-training with ten GTs"
        )
    if args.style_adapter_hidden <= 0 or args.max_style_logit < 0.0:
        raise ValueError("Invalid static style adapter configuration")
    if args.style_adapter_only and not args.static_style_adapter:
        raise ValueError(
            "--style-adapter-only requires --static-style-adapter"
        )
    if args.style_adapter_only and (
        args.warmstart_emotion_checkpoint is None
    ):
        raise ValueError(
            "Adapter-only training requires an emotion warmstart checkpoint"
        )
    warmstart_inputs = (
        args.warmstart_query_checkpoint,
        args.warmstart_emotion_checkpoint,
        args.warmstart_random_joint_checkpoint,
    )
    if sum(value is not None for value in warmstart_inputs) > 1:
        raise ValueError(
            "Choose only one query, emotion, or random-joint warmstart"
        )
    if args.anchor_initialization == "random_joint":
        if args.training_stage != "full":
            raise ValueError("random_joint anchor requires full training")
        if (
            args.warmstart_query_checkpoint is not None
            or args.warmstart_emotion_checkpoint is not None
        ):
            raise ValueError(
                "random_joint anchor requires its dedicated warm-start option"
            )
    elif args.warmstart_random_joint_checkpoint is not None:
        raise ValueError(
            "A random-joint warmstart requires --anchor-initialization "
            "random_joint"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Emotion Query-Mamba training requires CUDA")

    args.anchor_checkpoint = args.anchor_checkpoint.resolve()
    args.style_cache = args.style_cache.resolve()
    if args.warmstart_query_checkpoint is not None:
        args.warmstart_query_checkpoint = (
            args.warmstart_query_checkpoint.resolve()
        )
    if args.warmstart_emotion_checkpoint is not None:
        args.warmstart_emotion_checkpoint = (
            args.warmstart_emotion_checkpoint.resolve()
        )
    if args.warmstart_random_joint_checkpoint is not None:
        args.warmstart_random_joint_checkpoint = (
            args.warmstart_random_joint_checkpoint.resolve()
        )
    args.data_dir = args.data_dir.resolve()
    if args.session_split_manifest is not None:
        args.session_split_manifest = args.session_split_manifest.resolve()
    session_allowlist, session_split_manifest_sha256 = (
        load_session_allowlist(
            args.session_split_manifest,
            args.session_split_name,
        )
    )
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "train_metrics.jsonl"
    completion_path = args.run_dir / "TRAINING_COMPLETE"
    if metrics_path.exists() or completion_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite an existing run: {args.run_dir}"
        )

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    anchor_checkpoint = torch.load(
        args.anchor_checkpoint,
        map_location="cpu",
    )
    (
        anchor_config,
        anchor_checkpoint_max_seq_len,
        anchor_position_encoding_resized,
    ) = resolve_anchor_config_for_clip_length(
        checkpoint_payload=anchor_checkpoint["model_config"],
        clip_length=args.clip_length,
        allow_length_extension=args.allow_anchor_length_extension,
        allow_length_shrink=args.allow_anchor_length_shrink,
    )
    anchor = ConditionalREGNN(anchor_config)
    freeze_anchor = args.anchor_initialization == "pretrained_frozen"
    if freeze_anchor:
        anchor.load_state_dict(anchor_checkpoint["state_dict"])
    model = EmotionQueryResidualMamba(
        anchor,
        build_model_config(args),
        freeze_anchor=freeze_anchor,
    )

    warmstart_parameter_names: list[str] = []
    warmstart_sha256 = None
    emotion_warmstart_sha256 = None
    emotion_warmstart_method = None
    random_joint_warmstart_sha256 = None
    random_joint_warmstart_method = None
    if args.warmstart_query_checkpoint is not None:
        warmstart_parameter_names = load_shared_query_warmstart(
            model,
            args.warmstart_query_checkpoint,
        )
        warmstart_sha256 = sha256_file(
            args.warmstart_query_checkpoint
        )
    elif args.warmstart_emotion_checkpoint is not None:
        (
            warmstart_parameter_names,
            emotion_warmstart_method,
        ) = load_emotion_warmstart(
            model,
            args.warmstart_emotion_checkpoint,
        )
        emotion_warmstart_sha256 = sha256_file(
            args.warmstart_emotion_checkpoint
        )
    elif args.warmstart_random_joint_checkpoint is not None:
        (
            warmstart_parameter_names,
            random_joint_warmstart_method,
        ) = load_random_joint_warmstart(
            model,
            args.warmstart_random_joint_checkpoint,
        )
        random_joint_warmstart_sha256 = sha256_file(
            args.warmstart_random_joint_checkpoint
        )
    adapter_parameter_names: list[str] = []
    compatibility_only_parameter_names: list[str] = []
    if args.training_stage == "compatibility_head_only":
        compatibility_only_parameter_names = freeze_except_compatibility_head(
            model
        )
    elif args.style_adapter_only:
        adapter_parameter_names = freeze_except_static_style_adapter(model)
    model = model.to(device)

    style_cache = torch.load(args.style_cache, map_location="cpu")
    if (
        int(style_cache["version"]) != STYLE_DESCRIPTOR_VERSION
        or int(style_cache["descriptor_dim"]) != STYLE_DESCRIPTOR_DIM
        or style_cache["split"] != "train"
    ):
        raise ValueError("Invalid training style cache")
    if args.require_style_cache_session_match:
        if session_allowlist is None or session_split_manifest_sha256 is None:
            raise ValueError(
                "Style-cache session matching requires a named session split"
            )
        cache_sessions = style_cache.get("session_allowlist")
        if (
            cache_sessions is None
            or tuple(cache_sessions) != tuple(session_allowlist)
            or style_cache.get("session_split_name") != args.session_split_name
            or style_cache.get("session_split_manifest_sha256")
            != session_split_manifest_sha256
        ):
            raise ValueError("Style cache does not match the training session split")
    emotion_prototypes, prototype_counts, emotion_priors = (
        build_emotion_prototypes(
            style_cache["descriptors"].float(),
            shrinkage=args.prototype_shrinkage,
        )
    )
    if args.training_stage == "compatibility_head_only":
        criterion = None
    elif args.sequence_set_training:
        criterion = EmotionQuerySequenceSetLoss(
            style_mean=style_cache["mean"],
            style_std=style_cache["std"],
            anchor_ccc_weight=args.anchor_ccc_weight,
            learned_set_ccc_weight=args.learned_set_ccc_weight,
            mse_weight=args.mse_weight,
            velocity_weight=args.velocity_weight,
            style_weight=args.session_set_weight,
            semantic_identity_weight=args.query_identity_weight,
            semantic_identity_mode=args.semantic_identity_mode,
            semantic_identity_support_power=(
                args.semantic_identity_support_power
            ),
            semantic_identity_min_support=(
                args.semantic_identity_min_support
            ),
            compatibility_weight=args.compatibility_weight,
            mixture_weight=args.mixture_weight,
            sequence_transport_weight=args.sequence_transport_weight,
            sequence_transport_temperature=(
                args.sequence_transport_temperature
            ),
            sequence_distance_mode=args.sequence_distance_mode,
            set_matching_strategy=args.set_matching_strategy,
            style_diversity_weight=args.style_diversity_weight,
            style_diversity_margin=args.style_diversity_margin,
            official_diversity_weight=args.official_diversity_weight,
            official_diversity_margin=args.official_diversity_margin,
            official_diversity_batch_weight=(
                args.official_diversity_batch_weight
            ),
            channel_diversity_weight=args.channel_diversity_weight,
            channel_diversity_budget=args.channel_diversity_budget,
            channel_diversity_no_cap=args.channel_diversity_no_cap,
            frd_surrogate_weight=args.frd_surrogate_weight,
            frd_surrogate_target=args.frd_surrogate_target,
            frd_surrogate_stride=args.frd_surrogate_stride,
            frd_surrogate_calibration=args.frd_surrogate_calibration,
            frd_surrogate_mode=args.frd_surrogate_mode,
            frd_softdtw_stride=args.frd_softdtw_stride,
            frd_softdtw_band_ratio=args.frd_softdtw_band_ratio,
            frd_softdtw_gamma=args.frd_softdtw_gamma,
            frd_softdtw_pair_chunk_size=(
                args.frd_softdtw_pair_chunk_size
            ),
            frd_tail_weight=args.frd_tail_weight,
            frd_tail_fraction=args.frd_tail_fraction,
            frd_query_excess_weight=args.frd_query_excess_weight,
            frd_query_excess_fraction=args.frd_query_excess_fraction,
            frd_transport_weight=args.frd_transport_weight,
            frd_transport_temperature=args.frd_transport_temperature,
            frd_transport_iterations=args.frd_transport_iterations,
            sync_weight=args.sync_weight,
            sync_max_lag=args.sync_max_lag,
            sync_lag_step=args.sync_lag_step,
            sync_temporal_stride=args.sync_temporal_stride,
            sync_temperature=args.sync_temperature,
            sync_variance_epsilon=args.sync_variance_epsilon,
            residual_weight=args.residual_weight,
            sinkhorn_temperature=args.sinkhorn_temperature,
            sinkhorn_iterations=args.sinkhorn_iterations,
        ).to(device)
    elif args.session_set_training:
        criterion = EmotionQuerySessionSetLoss(
            style_mean=style_cache["mean"],
            style_std=style_cache["std"],
            style_weight=args.session_set_weight,
            paired_ccc_weight=args.ccc_weight,
            paired_mse_weight=args.mse_weight,
            velocity_weight=args.velocity_weight,
            diversity_weight=args.diversity_weight,
            diversity_margin=args.diversity_margin,
            residual_weight=args.residual_weight,
            semantic_identity_weight=args.query_identity_weight,
            sinkhorn_temperature=args.sinkhorn_temperature,
            sinkhorn_iterations=args.sinkhorn_iterations,
        ).to(device)
    else:
        criterion = EmotionQueryMambaLoss(
            style_mean=style_cache["mean"],
            style_std=style_cache["std"],
            emotion_prototypes=emotion_prototypes,
            ccc_weight=args.ccc_weight,
            mse_weight=args.mse_weight,
            velocity_weight=args.velocity_weight,
            emotion_weight=args.emotion_weight,
            query_identity_weight=args.query_identity_weight,
            compatibility_weight=args.compatibility_weight,
            prototype_weight=args.prototype_weight,
            contrast_identity_weight=args.contrast_identity_weight,
            contrast_prototype_weight=args.contrast_prototype_weight,
            diversity_weight=args.diversity_weight,
            diversity_margin=args.diversity_margin,
            style_diversity_weight=args.style_diversity_weight,
            style_diversity_margin=args.style_diversity_margin,
            centered_diversity_weight=args.centered_diversity_weight,
            mixture_weight=args.mixture_weight,
            residual_weight=args.residual_weight,
        ).to(device)

    dataset = ConditionalReactionDataset(
        root_dir=str(args.data_dir),
        split="train",
        clip_length=dataset_clip_length,
        training=True,
        target_mode=(
            "session_cropped"
            if args.sequence_set_training
            else (
                "session_style" if args.session_set_training else "paired"
            )
        ),
        num_targets=args.num_targets,
        style_cache_path=(
            str(args.style_cache) if args.session_set_training else None
        ),
        seed=args.seed,
        target_selection_mode=args.target_selection_mode,
        session_allowlist=session_allowlist,
        crop_stride=args.crop_stride,
        load_listener_3dmm=args.listener_3dmm_weight > 0,
    )
    target_support_prior = estimate_target_support_prior(
        dataset,
        list(style_cache["paths"]),
        style_cache["descriptors"],
    )
    compatibility_head_loaded = any(
        name.startswith("compatibility_head.")
        for name in warmstart_parameter_names
    )
    compatibility_head_reinitialized = (
        args.compatibility_mode == "target_support"
        and not compatibility_head_loaded
    )
    if compatibility_head_reinitialized:
        initialize_compatibility_head_from_prior(
            model,
            target_support_prior,
        )
    sample_weights, sampler_labels, sampler_counts = (
        build_sampling_weights(
            dataset,
            list(style_cache["paths"]),
            style_cache["descriptors"],
            balance_power=args.class_balance_power,
            max_sample_weight=args.max_sample_weight,
        )
    )
    worker_generator = torch.Generator().manual_seed(args.seed + 1_000_003)
    view_aware_sampler = None
    if args.training_stage == "compatibility_head_only":
        if args.view_aware_replacement_crops:
            raise ValueError(
                "View-aware replacement crops require replacement sampling"
            )
        sampling_protocol = stage_a_sampling_protocol(
            dataset_size=len(dataset),
            class_balance_power=args.class_balance_power,
            max_sample_weight=args.max_sample_weight,
            sample_weights=sample_weights,
        )
        sampling_protocol.update(
            {
                "target_support_prior_source": "B_fit",
                "target_support_prior_reads_B_cal": False,
                "target_support_prior_reads_B_confirm": False,
            }
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker,
            generator=worker_generator,
            drop_last=False,
        )
    else:
        sampler_generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=sampler_generator,
        )
        if args.view_aware_replacement_crops:
            view_aware_sampler = ViewAwareReplacementSampler(sampler)
            sampler = view_aware_sampler
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker,
            generator=worker_generator,
            drop_last=True,
        )
        sampling_protocol = {
            "sampler_type": (
                "ViewAwareReplacementSampler[WeightedRandomSampler]"
                if args.view_aware_replacement_crops
                else "WeightedRandomSampler"
            ),
            "view_aware_replacement_crops": bool(
                args.view_aware_replacement_crops
            ),
            "crop_view_policy": (
                "per_epoch_repeated_index_occurrence"
                if args.view_aware_replacement_crops
                else "single_epoch_path_view"
            ),
            "target_selection_view_aware": False,
            "stage_a_sampling_distribution": None,
            "class_balance_power": float(args.class_balance_power),
            "max_sample_weight": float(args.max_sample_weight),
            "replacement_sampling": True,
            "shuffle": False,
            "drop_last": True,
            "num_samples_per_epoch": len(dataset),
            "per_example_draw_probability_min": float(
                sample_weights.min() / sample_weights.sum()
            ),
            "per_example_draw_probability_max": float(
                sample_weights.max() / sample_weights.sum()
            ),
            "sample_weight_min": float(sample_weights.min()),
            "sample_weight_max": float(sample_weights.max()),
            "sample_weight_mean": float(sample_weights.mean()),
            "importance_correction": False,
            "positive_class_weighting": False,
            "compatibility_bce_weighting": "unweighted",
        }

    effective_body_lr = args.body_lr if args.body_lr is not None else args.lr
    effective_compatibility_lr = (
        args.compatibility_lr
        if args.compatibility_lr is not None
        else args.lr
    )
    compatibility_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("compatibility_head.")
    ]
    anchor_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("anchor.")
    ]
    body_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("compatibility_head.")
        and not name.startswith("anchor.")
    ]
    parameter_groups = []
    if compatibility_parameters:
        parameter_groups.append(
            {
                "params": compatibility_parameters,
                "lr": effective_compatibility_lr,
                "group_name": "compatibility_head",
            }
        )
    if body_parameters:
        parameter_groups.append(
            {
                "params": body_parameters,
                "lr": effective_body_lr,
                "group_name": "residual_body",
            }
        )
    if anchor_parameters:
        parameter_groups.append(
            {
                "params": anchor_parameters,
                "lr": args.anchor_lr,
                "group_name": "anchor",
            }
        )
    if not parameter_groups:
        raise RuntimeError("Model has no trainable parameters")
    trainable_parameters = (
        compatibility_parameters + body_parameters + anchor_parameters
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.05
        + 0.95
        * 0.5
        * (1.0 + math.cos(math.pi * min(epoch, args.epochs) / args.epochs)),
    )
    isolated_va_optimizer = None
    isolated_va_surrogate = None
    isolated_va_trainable_scalar_count = 0
    if args.isolated_va_calibration:
        isolated_va_optimizer = torch.optim.AdamW(
            [model.to_residual.weight, model.to_residual.bias],
            lr=args.isolated_va_lr,
            weight_decay=0.0,
            betas=(0.9, 0.99),
        )
        isolated_va_surrogate = BandedSoftDTWFRDSurrogate(
            temporal_stride=args.frd_softdtw_stride,
            band_ratio=args.frd_softdtw_band_ratio,
            gamma=args.frd_softdtw_gamma,
            pair_chunk_size=args.frd_softdtw_pair_chunk_size,
        ).to(device)
        isolated_va_trainable_scalar_count = int(
            2 * model.to_residual.weight.shape[1] + 2
        )
    anchor_sha256 = sha256_file(args.anchor_checkpoint)
    style_cache_sha256 = sha256_file(args.style_cache)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in model.anchor.parameters()
        if not parameter.requires_grad
    )
    trainable_anchor_parameter_count = sum(
        parameter.numel()
        for parameter in model.anchor.parameters()
        if parameter.requires_grad
    )
    config_payload = {
        "method": "EQR-Mamba",
        "method_full_name": (
            "Emotion-Query Residual Mamba"
            if args.temporal_backbone == "mamba"
            else "Emotion-Query Residual Frame-MLP"
        ),
        "temporal_backbone": args.temporal_backbone,
        "anchor": str(args.anchor_checkpoint),
        "anchor_epoch": (
            int(anchor_checkpoint["epoch"])
            if freeze_anchor
            else None
        ),
        "anchor_initialization": args.anchor_initialization,
        "anchor_template_checkpoint": str(args.anchor_checkpoint),
        "anchor_weights_loaded": freeze_anchor,
        "anchor_sha256": anchor_sha256,
        "style_cache": str(args.style_cache),
        "style_cache_sha256": style_cache_sha256,
        "warmstart_query_checkpoint": (
            str(args.warmstart_query_checkpoint)
            if args.warmstart_query_checkpoint is not None
            else None
        ),
        "warmstart_query_sha256": warmstart_sha256,
        "warmstart_emotion_checkpoint": (
            str(args.warmstart_emotion_checkpoint)
            if args.warmstart_emotion_checkpoint is not None
            else None
        ),
        "warmstart_emotion_sha256": emotion_warmstart_sha256,
        "warmstart_emotion_method": emotion_warmstart_method,
        "warmstart_random_joint_checkpoint": (
            str(args.warmstart_random_joint_checkpoint)
            if args.warmstart_random_joint_checkpoint is not None
            else None
        ),
        "warmstart_random_joint_sha256": (
            random_joint_warmstart_sha256
        ),
        "warmstart_random_joint_method": (
            random_joint_warmstart_method
        ),
        "warmstarted_parameter_count": len(
            warmstart_parameter_names
        ),
        "style_adapter_only_parameter_names": adapter_parameter_names,
        "compatibility_only_parameter_names": (
            compatibility_only_parameter_names
        ),
        "compatibility_head_loaded": compatibility_head_loaded,
        "compatibility_head_reinitialized": (
            compatibility_head_reinitialized
        ),
        "target_support_prior": target_support_prior.tolist(),
        "session_split_name": args.session_split_name,
        "session_allowlist": session_allowlist,
        "session_split_manifest_sha256": session_split_manifest_sha256,
        "sampling_protocol": sampling_protocol,
        "optimizer_group_lrs": {
            group["group_name"]: group["lr"] for group in parameter_groups
        },
        "dataset_size": len(dataset),
        "dataset_clip_length": dataset_clip_length,
        "online_rollout_output_frames": (
            args.online_rollout_blocks * args.supervision_tail_frames
            if args.online_rollout_blocks > 1
            else (args.supervision_tail_frames or args.clip_length)
        ),
        "sampler_label_counts": sampler_counts.tolist(),
        "sampler_label_checksum": int(sampler_labels.sum()),
        "prototype_counts": prototype_counts.tolist(),
        "emotion_priors": emotion_priors.tolist(),
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_anchor_parameter_count": (
            trainable_anchor_parameter_count
        ),
        "isolated_va_trainable_scalar_count": (
            isolated_va_trainable_scalar_count
        ),
        "isolated_va_gradient_contract": (
            "to_residual rows 15:17 and biases 15:17 only"
            if args.isolated_va_calibration
            else None
        ),
        "isolated_va_reads_B_cal": False,
        "isolated_va_reads_B_confirm": False,
        "frozen_anchor_parameter_count": frozen_parameter_count,
        "anchor_model_config": asdict(anchor_config),
        "anchor_checkpoint_max_seq_len": anchor_checkpoint_max_seq_len,
        "anchor_position_encoding_resized": anchor_position_encoding_resized,
        "anchor_position_encoding_extended": (
            anchor_position_encoding_resized
            and args.clip_length > anchor_checkpoint_max_seq_len
        ),
        "anchor_position_encoding_shrunk": (
            anchor_position_encoding_resized
            and args.clip_length < anchor_checkpoint_max_seq_len
        ),
        "emotion_query_model_config": asdict(model.config),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Emotion-Query Mamba trainable={trainable_parameter_count:,} "
        f"frozen_anchor={frozen_parameter_count:,} "
        f"trainable_anchor={trainable_anchor_parameter_count:,} "
        f"samples={len(dataset)} batches={len(dataloader)} "
        f"warmstarted={len(warmstart_parameter_names)} "
        f"stage={args.training_stage} "
        f"head_reset={compatibility_head_reinitialized} "
        f"sampler={sampling_protocol['sampler_type']} "
        f"sampler_counts={sampler_counts.tolist()} "
        f"precision={args.precision}",
        flush=True,
    )

    use_bf16 = args.precision == "bf16"
    global_step = 0
    if args.training_stage == "compatibility_head_only":
        metric_names = (
            "loss",
            "compatibility_bce",
            "compatibility_brier",
            "compatibility_mean",
            "compatibility_min",
            "compatibility_max",
            "active_query_count_025",
            "active_query_count_050",
            *tuple(
                f"compatibility_bce_e{emotion_index}"
                for emotion_index in range(8)
            ),
            *tuple(
                f"compatibility_brier_e{emotion_index}"
                for emotion_index in range(8)
            ),
            *tuple(
                f"target_support_e{emotion_index}"
                for emotion_index in range(8)
            ),
            *tuple(
                f"predicted_support_e{emotion_index}"
                for emotion_index in range(8)
            ),
        )
    elif args.sequence_set_training:
        metric_names = (
            "loss",
            "loss_appropriateness",
            "loss_semantic_allocation",
            "loss_distribution_risk",
            "set_max_ccc",
            "paired_anchor_ccc",
            "learned_set_max_ccc",
            "sequence_transport_ccc",
            "sequence_transport_alignment_cost",
            "matched_mse",
            "matched_velocity",
            "style_ot",
            "semantic_identity",
            "semantic_accuracy",
            "semantic_weighted_accuracy",
            "semantic_identity_mean_weight",
            "compatibility_bce",
            "compatibility_mean",
            "compatibility_min",
            "compatibility_max",
            "active_query_count_025",
            "active_query_count_050",
            "mixture_query_consistency",
            "gate_warmup_alpha",
            "target_support_e0",
            "target_support_e1",
            "target_support_e2",
            "target_support_e3",
            "target_support_e4",
            "target_support_e5",
            "target_support_e6",
            "target_support_e7",
            "predicted_support_e0",
            "predicted_support_e1",
            "predicted_support_e2",
            "predicted_support_e3",
            "predicted_support_e4",
            "predicted_support_e5",
            "predicted_support_e6",
            "predicted_support_e7",
            "style_pairwise_smse",
            "style_diversity_loss",
            "style_diversity_active_fraction",
            "surrogate_official_frdiv",
            "official_diversity_loss",
            "official_diversity_active_fraction",
            "official_diversity_batch_loss",
            "channel_diversity_loss",
            "channel_diversity_valid_fraction",
            "prediction_frdiv_au",
            "prediction_frdiv_va",
            "prediction_frdiv_expression",
            "target_frdiv_au",
            "target_frdiv_va",
            "target_frdiv_expression",
            "desired_frdiv_au",
            "desired_frdiv_va",
            "desired_frdiv_expression",
            "surrogate_official_frd",
            "frd_surrogate_loss",
            "frd_surrogate_active_fraction",
            "surrogate_official_frd_tail",
            "frd_tail_loss",
            "frd_tail_active_fraction",
            "surrogate_official_frd_query_excess",
            "surrogate_official_frd_query_tail_excess",
            "frd_query_excess_loss",
            "frd_query_excess_active_fraction",
            "frd_match_coverage",
            "target_match_coverage",
            "worst_query_excess",
            "learned_queries_better_than_anchor_fraction",
            "frd_query_0_mean_best_target",
            "frd_query_1_mean_best_target",
            "frd_query_2_mean_best_target",
            "frd_query_3_mean_best_target",
            "frd_query_4_mean_best_target",
            "frd_query_5_mean_best_target",
            "frd_query_6_mean_best_target",
            "frd_query_7_mean_best_target",
            "frd_query_8_mean_best_target",
            "frd_query_9_mean_best_target",
            "frd_query_0_mean_excess",
            "frd_query_1_mean_excess",
            "frd_query_2_mean_excess",
            "frd_query_3_mean_excess",
            "frd_query_4_mean_excess",
            "frd_query_5_mean_excess",
            "frd_query_6_mean_excess",
            "frd_query_7_mean_excess",
            "frd_query_8_mean_excess",
            "frd_query_9_mean_excess",
            "surrogate_official_frd_transport",
            "frd_transport_loss",
            "frd_transport_active_fraction",
            "frd_transport_coverage",
            "frd_transport_entropy",
            "sync_loss",
            "surrogate_sync_expected_abs_lag",
            "surrogate_sync_hard_abs_lag",
            "surrogate_sync_zero_lag_correlation",
            "surrogate_sync_peak_correlation",
            "sync_valid_pair_fraction",
            "sync_valid_channels_at_peak",
            "sync_active_fraction",
            "residual_l2",
            "match_coverage",
            "transport_entropy",
            "sequence_transport_entropy",
        )
    elif args.session_set_training:
        metric_names = (
            "loss",
            "style_ot",
            "paired_ccc",
            "paired_mse",
            "velocity",
            "centered_smse",
            "diversity_loss",
            "residual_l2",
            "au_flip_rate",
            "transport_entropy",
            "semantic_identity",
            "semantic_accuracy",
        )
    else:
        metric_names = (
            "loss",
            "selected_ccc",
            "selected_mse",
            "velocity",
            "emotion_ce",
            "query_identity",
            "query_accuracy",
            "compatibility_ce",
            "compatibility_bce",
            "compatibility_accuracy",
            "prototype_loss",
            "contrast_query_identity",
            "contrast_query_accuracy",
            "contrast_prototype_loss",
            "pairwise_smse",
            "style_pairwise_smse",
            "centered_pairwise_smse",
            "diversity_loss",
            "style_diversity_loss",
            "diversity_active_fraction",
            "style_diversity_active_fraction",
            "mixture_ccc",
            "mixture_mse",
            "residual_l2",
            "selected_gate",
        )
    torch.cuda.reset_peak_memory_stats(device)
    if args.listener_3dmm_weight > 0:
        metric_names += ("listener_3dmm_mse", "listener_3dmm_velocity")
    sync_conditional_metric_names = {
        "sync_loss",
        "surrogate_sync_expected_abs_lag",
        "surrogate_sync_hard_abs_lag",
        "surrogate_sync_zero_lag_correlation",
        "surrogate_sync_peak_correlation",
        "sync_valid_pair_fraction",
        "sync_valid_channels_at_peak",
    }
    isolated_va_metric_names = (
        "loss",
        "tail_risk_ratio",
        "mean_risk_ratio",
        "proximal",
        "FRDiv_ratio",
        "FRVar_ratio",
        "diversity_shortfall",
        "variance_shortfall",
        "gradient_norm",
    )

    for epoch_index in range(args.epochs):
        dataset.set_epoch(epoch_index)
        model.train()
        epoch_start = time.perf_counter()
        sums = {name: 0.0 for name in metric_names}
        metric_counts = {name: 0 for name in metric_names}
        isolated_va_sums = {
            name: 0.0 for name in isolated_va_metric_names
        }
        isolated_va_steps = 0
        isolated_va_epoch_active = (
            args.isolated_va_calibration
            and epoch_index + 1 >= args.isolated_va_start_epoch
        )
        selected_counts = torch.zeros(8, dtype=torch.long)
        target_pool_seen: set[tuple[str, str]] = set()
        target_available_slots = 0
        target_duplicate_slots = 0
        target_total_slots = 0
        calibration_prediction_batches: list[torch.Tensor] = []
        calibration_target_batches: list[torch.Tensor] = []
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
                device,
                non_blocking=True,
            )
            speaker_emotion = batch["speaker_emotion"].to(
                device,
                non_blocking=True,
            )
            speaker_3dmm = batch["speaker_3dmm"].to(
                device,
                non_blocking=True,
            )
            paired_target = batch["target"].to(
                device,
                non_blocking=True,
            )
            lengths = batch["length"].to(device, non_blocking=True)
            frame_ids = torch.arange(
                paired_target.shape[1],
                device=device,
            )
            valid_mask = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)
            supervision_mask = restrict_valid_mask_to_tail(
                valid_mask,
                args.supervision_tail_frames,
            )
            target_profile = masked_emotion_profile(
                paired_target,
                supervision_mask,
            )
            emotion_indices = target_profile.argmax(dim=-1)
            contrast_offsets = torch.randint(
                1,
                8,
                emotion_indices.shape,
                device=device,
            )
            contrast_emotion_indices = (
                emotion_indices + contrast_offsets
            ) % 8

            model.set_gate_step(global_step)
            optimizer.zero_grad(set_to_none=True)
            isolated_va_reference = None
            isolated_va_replay_cpu_rng = None
            isolated_va_replay_cuda_rng = None
            if isolated_va_epoch_active:
                isolated_va_replay_cpu_rng = torch.get_rng_state()
                isolated_va_replay_cuda_rng = torch.cuda.get_rng_state(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                if args.training_stage == "compatibility_head_only":
                    sequence_targets = batch["targets"].to(
                        device,
                        non_blocking=True,
                    )
                    target_lengths = batch["target_lengths"].to(
                        device,
                        non_blocking=True,
                    )
                    target_available_mask = batch[
                        "target_available_mask"
                    ].to(device, non_blocking=True)
                    target_frame_ids = torch.arange(
                        sequence_targets.shape[2],
                        device=device,
                    )
                    target_valid_mask = (
                        target_frame_ids[None, None]
                        < target_lengths[:, :, None]
                    ) & target_available_mask[:, :, None]
                    target_valid_mask = target_valid_mask & (
                        supervision_mask[:, None, :]
                    )
                    output = model.forward_compatibility(
                        speaker_audio,
                        speaker_emotion,
                        speaker_3dmm,
                        lengths,
                    )
                    target_support = target_set_emotion_support(
                        sequence_targets,
                        target_valid_mask,
                    )
                    support_example_mask = target_valid_mask.any(dim=-1).any(
                        dim=-1
                    )
                    losses = compatibility_calibration_loss(
                        output["compatibility_binary_logits"],
                        output["compatibility_probs"],
                        target_support,
                        support_example_mask,
                    )
                    calibration_prediction_batches.append(
                        output["compatibility_probs"][
                            support_example_mask
                        ].detach().float().cpu()
                    )
                    calibration_target_batches.append(
                        target_support[support_example_mask].detach().float().cpu()
                    )
                    target_available_slots += int(
                        batch["target_available_mask"].sum()
                    )
                    target_duplicate_slots += int(
                        batch["target_is_duplicate"].sum()
                    )
                    target_total_slots += int(
                        batch["target_available_mask"].numel()
                    )
                    collated_paths = batch["target_paths"]
                    for sample_index, clip_id in enumerate(batch["clip_id"]):
                        for target_index in range(len(collated_paths)):
                            if not bool(
                                batch["target_available_mask"][
                                    sample_index,
                                    target_index,
                                ]
                            ):
                                continue
                            target_path = collated_paths[target_index][
                                sample_index
                            ]
                            target_pool_seen.add((clip_id, target_path))
                elif args.sequence_set_training:
                    style_residual_scale = (
                        args.train_style_residual_scale
                        if args.train_style_residual_scale is not None
                        else args.train_residual_scale
                    )
                    if args.online_rollout_blocks > 1:
                        output = forward_online_sequence_set_rollout(
                            model,
                            speaker_audio,
                            speaker_emotion,
                            speaker_3dmm,
                            lengths,
                            context_frames=args.clip_length,
                            output_frames=args.supervision_tail_frames,
                            rollout_blocks=args.online_rollout_blocks,
                            residual_scale=args.train_residual_scale,
                            style_residual_scale=style_residual_scale,
                        )
                    else:
                        output = model(
                            speaker_audio,
                            speaker_emotion,
                            speaker_3dmm,
                            lengths,
                            residual_scale=args.train_residual_scale,
                            style_residual_scale=style_residual_scale,
                        )
                    channel_multipliers = (
                        [args.train_au_residual_multiplier] * 15
                        + [args.train_va_residual_multiplier] * 2
                        + [args.train_expression_residual_multiplier] * 8
                    )
                    if any(value != 1.0 for value in channel_multipliers):
                        output = dict(output)
                        output["prediction"] = (
                            apply_residual_channel_multipliers(
                                model,
                                output,
                                channel_multipliers,
                            )
                        )
                        residual_scale = torch.as_tensor(
                            channel_multipliers,
                            device=output["dynamic_residual_raw"].device,
                            dtype=output["dynamic_residual_raw"].dtype,
                        )
                        output["residual_raw"] = (
                            output["dynamic_residual_raw"]
                            + output["style_residual_raw"]
                        ) * residual_scale
                    sequence_targets = batch["targets"].to(
                        device,
                        non_blocking=True,
                    )
                    target_lengths = batch["target_lengths"].to(
                        device,
                        non_blocking=True,
                    )
                    target_available_mask = batch[
                        "target_available_mask"
                    ].to(device, non_blocking=True)
                    sync_speaker_emotion = speaker_emotion
                    if args.online_rollout_blocks > 1:
                        retained_frames = (
                            args.online_rollout_blocks
                            * args.supervision_tail_frames
                        )
                        target_start = (
                            args.clip_length - args.supervision_tail_frames
                        )
                        target_stop = target_start + retained_frames
                        sequence_targets = sequence_targets[
                            :, :, target_start:target_stop
                        ]
                        target_lengths = (target_lengths - target_start).clamp(
                            min=0,
                            max=retained_frames,
                        )
                        target_frame_ids = torch.arange(
                            retained_frames,
                            device=device,
                        )
                        supervision_mask = output["valid_mask"]
                        sync_speaker_emotion = speaker_emotion[
                            :, target_start:target_stop
                        ]
                    else:
                        target_frame_ids = frame_ids
                    target_valid_mask = (
                        target_frame_ids[None, None]
                        < target_lengths[:, :, None]
                    ) & target_available_mask[:, :, None]
                    target_valid_mask = target_valid_mask & (
                        supervision_mask[:, None, :]
                    )
                    loss_output = dict(output)
                    loss_output["valid_mask"] = supervision_mask
                    losses = criterion(
                        loss_output,
                        sequence_targets,
                        target_valid_mask,
                        speaker_emotion=(
                            sync_speaker_emotion
                            if batch_index % args.sync_every_n_batches == 0
                            else None
                        ),
                        compute_sync=(
                            batch_index % args.sync_every_n_batches == 0
                        ),
                        sync_loss_scale=args.sync_every_n_batches,
                    )
                    if isolated_va_epoch_active:
                        isolated_va_reference = output["prediction"].detach()
                    target_available_slots += int(
                        batch["target_available_mask"].sum()
                    )
                    target_duplicate_slots += int(
                        batch["target_is_duplicate"].sum()
                    )
                    target_total_slots += int(
                        batch["target_available_mask"].numel()
                    )
                    collated_paths = batch["target_paths"]
                    for sample_index, clip_id in enumerate(batch["clip_id"]):
                        for target_index in range(len(collated_paths)):
                            if not bool(
                                batch["target_available_mask"][
                                    sample_index,
                                    target_index,
                                ]
                            ):
                                continue
                            target_path = collated_paths[target_index][
                                sample_index
                            ]
                            target_pool_seen.add((clip_id, target_path))
                elif args.session_set_training:
                    output = model(
                        speaker_audio,
                        speaker_emotion,
                        speaker_3dmm,
                        lengths,
                        residual_scale=args.train_residual_scale,
                        style_residual_scale=(
                            args.train_style_residual_scale
                            if args.train_style_residual_scale is not None
                            else args.train_residual_scale
                        ),
                    )
                    losses = criterion(
                        output,
                        paired_target,
                        batch["target_styles"].to(
                            device,
                            non_blocking=True,
                        ),
                    )
                else:
                    output = model.forward_selected(
                        speaker_audio,
                        speaker_emotion,
                        speaker_3dmm,
                        lengths,
                        emotion_indices,
                        contrast_emotion_indices=contrast_emotion_indices,
                        teacher_emotion_profile=target_profile,
                        residual_scale=args.train_residual_scale,
                        style_residual_scale=args.train_style_residual_scale,
                    )
                    losses = criterion(
                        {
                            **output,
                            "valid_mask": supervision_mask,
                        },
                        paired_target,
                        emotion_indices,
                    )
            if args.listener_3dmm_weight > 0:
                geometry_losses = listener_3dmm_loss(
                    output["prediction_3dmm"], output["prediction"][:, 1:],
                    batch["targets_3dmm"].to(device, non_blocking=True),
                    sequence_targets, target_valid_mask,
                )
                losses.update(geometry_losses)
                losses["loss"] = losses["loss"] + args.listener_3dmm_weight * (
                    losses["listener_3dmm_mse"] + args.listener_3dmm_velocity_weight * losses["listener_3dmm_velocity"]
                )
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch_index + 1}, "
                    f"batch {batch_index + 1}"
                )
            losses["loss"].backward()
            if args.training_stage == "compatibility_head_only":
                unexpected_gradients = [
                    name
                    for name, parameter in model.named_parameters()
                    if not name.startswith("compatibility_head.")
                    and parameter.grad is not None
                ]
                if unexpected_gradients:
                    raise RuntimeError(
                        "Stage A produced non-head gradients: "
                        f"{unexpected_gradients}"
                    )
            validate_anchor_gradient_contract(model)
            gradient_norm = clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()

            isolated_va_losses = None
            if isolated_va_epoch_active:
                if (
                    isolated_va_optimizer is None
                    or isolated_va_surrogate is None
                    or isolated_va_reference is None
                    or isolated_va_replay_cpu_rng is None
                    or isolated_va_replay_cuda_rng is None
                ):
                    raise RuntimeError(
                        "Isolated VA calibration was not initialized correctly"
                    )
                # Remove main-update gradients before the isolated micro-step.
                # The second forward replays the main forward's dropout masks;
                # the caller-visible RNG stream is restored afterward.
                optimizer.zero_grad(set_to_none=True)
                isolated_va_optimizer.zero_grad(set_to_none=True)
                resume_cpu_rng = torch.get_rng_state()
                resume_cuda_rng = torch.cuda.get_rng_state(device)
                non_va_weight_before = torch.cat(
                    (
                        model.to_residual.weight[:15].detach().clone(),
                        model.to_residual.weight[17:].detach().clone(),
                    ),
                    dim=0,
                )
                non_va_bias_before = torch.cat(
                    (
                        model.to_residual.bias[:15].detach().clone(),
                        model.to_residual.bias[17:].detach().clone(),
                    ),
                    dim=0,
                )
                requires_grad_states = set_only_output_head_trainable(model)
                try:
                    torch.set_rng_state(isolated_va_replay_cpu_rng)
                    torch.cuda.set_rng_state(
                        isolated_va_replay_cuda_rng,
                        device,
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        if args.online_rollout_blocks > 1:
                            isolated_output = (
                                forward_online_sequence_set_rollout(
                                    model,
                                    speaker_audio,
                                    speaker_emotion,
                                    speaker_3dmm,
                                    lengths,
                                    context_frames=args.clip_length,
                                    output_frames=(
                                        args.supervision_tail_frames
                                    ),
                                    rollout_blocks=(
                                        args.online_rollout_blocks
                                    ),
                                    residual_scale=(
                                        args.train_residual_scale
                                    ),
                                    style_residual_scale=(
                                        style_residual_scale
                                    ),
                                )
                            )
                        else:
                            isolated_output = model(
                                speaker_audio,
                                speaker_emotion,
                                speaker_3dmm,
                                lengths,
                                residual_scale=args.train_residual_scale,
                                style_residual_scale=style_residual_scale,
                            )
                        isolated_predictions = (
                            apply_residual_channel_multipliers(
                                model,
                                isolated_output,
                                channel_multipliers,
                            )
                        )
                        isolated_va_losses = isolated_va_objective(
                            isolated_predictions,
                            isolated_va_reference,
                            sequence_targets,
                            supervision_mask,
                            target_valid_mask,
                            isolated_va_surrogate,
                            va_calibration_config,
                        )
                    q0_difference = float(
                        (
                            isolated_predictions[:, 0]
                            - isolated_va_reference[:, 0]
                        ).abs().max()
                    )
                    if (
                        args.anchor_initialization == "pretrained_frozen"
                        and q0_difference > 2e-6
                    ):
                        raise RuntimeError(
                            "q0 changed during isolated VA replay: "
                            f"max_abs={q0_difference}"
                        )
                    if not torch.isfinite(isolated_va_losses["loss"]):
                        raise RuntimeError("Non-finite isolated VA loss")
                    isolated_va_losses["loss"].backward()
                    mask_output_head_gradients_to_va(model)
                    isolated_va_gradient_norm = clip_grad_norm_(
                        [model.to_residual.weight, model.to_residual.bias],
                        args.isolated_va_gradient_clip,
                    )
                    isolated_va_optimizer.step()
                    non_va_weight_after = torch.cat(
                        (
                            model.to_residual.weight[:15].detach(),
                            model.to_residual.weight[17:].detach(),
                        ),
                        dim=0,
                    )
                    non_va_bias_after = torch.cat(
                        (
                            model.to_residual.bias[:15].detach(),
                            model.to_residual.bias[17:].detach(),
                        ),
                        dim=0,
                    )
                    if not torch.equal(
                        non_va_weight_before,
                        non_va_weight_after,
                    ) or not torch.equal(
                        non_va_bias_before,
                        non_va_bias_after,
                    ):
                        raise RuntimeError(
                            "Isolated VA optimizer changed a non-VA output row"
                        )
                finally:
                    restore_requires_grad(model, requires_grad_states)
                    torch.set_rng_state(resume_cpu_rng)
                    torch.cuda.set_rng_state(resume_cuda_rng, device)

                isolated_va_steps += 1
                for name in isolated_va_metric_names:
                    if name == "gradient_norm":
                        value = float(isolated_va_gradient_norm)
                    else:
                        value = float(isolated_va_losses[name].detach())
                    isolated_va_sums[name] += value

            processed += 1
            global_step += 1
            selected_counts += torch.bincount(
                emotion_indices.detach().cpu(),
                minlength=8,
            )
            for name in metric_names:
                if (
                    name in sync_conditional_metric_names
                    and float(losses["sync_active_fraction"].detach()) == 0.0
                ):
                    continue
                sums[name] += float(losses[name].detach())
                metric_counts[name] += 1
            if (
                batch_index == 0
                or (batch_index + 1) % args.print_every == 0
            ):
                elapsed = time.perf_counter() - epoch_start
                samples_per_second = (
                    processed * args.batch_size / max(elapsed, 1e-8)
                )
                if args.training_stage == "compatibility_head_only":
                    detail = (
                        f"bce={losses['compatibility_bce'].item():.5f} "
                        f"brier={losses['compatibility_brier'].item():.5f} "
                        f"gate={losses['compatibility_mean'].item():.3f} "
                        f"active={losses['active_query_count_050'].item():.2f} "
                    )
                elif args.sequence_set_training:
                    detail = (
                        f"setCCC={losses['set_max_ccc'].item():.5f} "
                        f"setOT={losses['sequence_transport_alignment_cost'].item():.5f} "
                        f"styleOT={losses['style_ot'].item():.5f} "
                        f"offDiv={losses['surrogate_official_frdiv'].item():.5f} "
                        f"groups={losses['prediction_frdiv_au'].item():.3f}/"
                        f"{losses['prediction_frdiv_va'].item():.3f}/"
                        f"{losses['prediction_frdiv_expression'].item():.3f} "
                        f"frd={losses['surrogate_official_frd'].item():.2f} "
                        f"frdL={losses['frd_surrogate_loss'].item():.4f} "
                        f"frdTail={losses['surrogate_official_frd_tail'].item():.2f} "
                        f"frdOT={losses['surrogate_official_frd_transport'].item():.2f} "
                        f"gate={losses['compatibility_mean'].item():.3f} "
                        f"active={losses['active_query_count_050'].item():.2f} "
                        f"sync={losses['surrogate_sync_hard_abs_lag'].item():.2f}/"
                        f"{losses['surrogate_sync_expected_abs_lag'].item():.2f} "
                        f"semacc={losses['semantic_accuracy'].item():.3f} "
                        f"coverage={losses['match_coverage'].item():.3f} "
                    )
                elif args.session_set_training:
                    detail = (
                        f"ccc={losses['paired_ccc'].item():.5f} "
                        f"styleOT={losses['style_ot'].item():.5f} "
                        f"semacc={losses['semantic_accuracy'].item():.3f} "
                        f"AUflip={losses['au_flip_rate'].item():.3f} "
                        f"center={losses['centered_smse'].item():.5f} "
                    )
                else:
                    detail = (
                        f"ccc={losses['selected_ccc'].item():.5f} "
                        f"qacc={losses['query_accuracy'].item():.3f} "
                        f"cacc={losses['compatibility_accuracy'].item():.3f} "
                        f"proto={losses['prototype_loss'].item():.5f} "
                        f"pair={losses['pairwise_smse'].item():.5f} "
                        f"style={losses['style_pairwise_smse'].item():.5f} "
                        f"center={losses['centered_pairwise_smse'].item():.5f} "
                        f"sdiv={losses['style_diversity_loss'].item():.5f} "
                        f"xacc={losses['contrast_query_accuracy'].item():.3f} "
                        f"gate={losses['selected_gate'].item():.3f} "
                    )
                if isolated_va_losses is not None:
                    detail += (
                        f"vaRisk={isolated_va_losses['mean_risk_ratio'].item():.4f} "
                        f"vaTail={isolated_va_losses['tail_risk_ratio'].item():.4f} "
                        f"vaDiv={isolated_va_losses['FRDiv_ratio'].item():.4f} "
                        f"vaVar={isolated_va_losses['FRVar_ratio'].item():.4f} "
                    )
                print(
                    f"Epoch [{epoch_index + 1}/{args.epochs}] "
                    f"step [{batch_index + 1}/{len(dataloader)}] "
                    f"loss={losses['loss'].item():.5f} "
                    f"{detail}"
                    f"grad={float(gradient_norm):.3f} "
                    f"speed={samples_per_second:.2f}/s "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if processed == 0:
            raise RuntimeError("No training batches were processed")
        optimizer_lrs = {
            str(group.get("group_name", f"group_{index}")): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }
        scheduler.step()
        next_optimizer_lrs = {
            str(group.get("group_name", f"group_{index}")): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_metrics: Dict[str, object] = {
            "epoch": epoch_index + 1,
            "global_step": global_step,
            "lr": next(iter(optimizer_lrs.values())),
            "optimizer_lrs": optimizer_lrs,
            "next_optimizer_lrs": next_optimizer_lrs,
            "processed_batches": processed,
            "selected_emotion_counts": selected_counts.tolist(),
            "epoch_seconds": epoch_seconds,
            "samples_per_second": (
                processed * args.batch_size / epoch_seconds
            ),
            "gpu_peak_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            **{
                name: value / processed
                if metric_counts[name] == processed
                else value / max(metric_counts[name], 1)
                for name, value in sums.items()
            },
        }
        epoch_metrics.update(
            {
                "isolated_va_calibration_enabled": (
                    args.isolated_va_calibration
                ),
                "isolated_va_epoch_active": isolated_va_epoch_active,
                "isolated_va_steps": isolated_va_steps,
                "isolated_va_trainable_scalar_count": (
                    isolated_va_trainable_scalar_count
                ),
                "isolated_va_reads_B_cal": False,
                "isolated_va_reads_B_confirm": False,
            }
        )
        if isolated_va_steps > 0:
            epoch_metrics.update(
                {
                    f"isolated_va_{name}": value / isolated_va_steps
                    for name, value in isolated_va_sums.items()
                }
            )
        if args.training_stage == "compatibility_head_only":
            if calibration_prediction_batches:
                calibration_prediction_tensor = torch.cat(
                    calibration_prediction_batches,
                    dim=0,
                )
                calibration_target_tensor = torch.cat(
                    calibration_target_batches,
                    dim=0,
                )
                calibration_predictions = calibration_prediction_tensor.numpy()
                calibration_targets = calibration_target_tensor.numpy()
                epoch_metrics.update(
                    compatibility_epoch_calibration_metrics(
                        calibration_prediction_tensor,
                        calibration_target_tensor,
                    )
                )
                prior_predictions = target_support_prior[None].expand_as(
                    calibration_target_tensor
                )
                prior_bce = torch.nn.functional.binary_cross_entropy(
                    prior_predictions,
                    calibration_target_tensor,
                )
                prior_brier = (
                    prior_predictions - calibration_target_tensor
                ).square().mean()
                epoch_metrics["prior_baseline_bce"] = float(prior_bce)
                epoch_metrics["prior_baseline_brier"] = float(prior_brier)
                epoch_metrics["bce_improvement_over_prior"] = float(
                    prior_bce - epoch_metrics["compatibility_bce"]
                )
                epoch_metrics["brier_improvement_over_prior"] = float(
                    prior_brier - epoch_metrics["compatibility_brier"]
                )
                correlations = [
                    spearman_correlation(
                        calibration_predictions[:, emotion_index],
                        calibration_targets[:, emotion_index],
                    )
                    for emotion_index in range(8)
                ]
                for emotion_index, correlation in enumerate(correlations):
                    epoch_metrics[
                        f"support_spearman_e{emotion_index}"
                    ] = correlation
                epoch_metrics["support_spearman_mean"] = float(
                    np.mean(correlations)
                )
                epoch_metrics["predicted_support_std_mean"] = float(
                    calibration_predictions.std(axis=0).mean()
                )
                epoch_metrics[
                    "predicted_prior_mean_absolute_error"
                ] = float(
                    np.abs(
                        calibration_predictions.mean(axis=0)
                        - target_support_prior.numpy()
                    ).mean()
                )
            else:
                epoch_metrics["support_spearman_mean"] = 0.0
        if args.sequence_set_training:
            epoch_metrics.update(
                {
                    "target_selection_epoch": dataset.epoch,
                    "target_pool_unique_pairs": len(target_pool_seen),
                    "target_pool_pair_count": (
                        dataset.target_pool_pair_count
                    ),
                    "target_pool_coverage": (
                        len(target_pool_seen)
                        / max(dataset.target_pool_pair_count, 1)
                    ),
                    "target_available_fraction": (
                        target_available_slots / max(target_total_slots, 1)
                    ),
                    "target_duplicate_fraction": (
                        target_duplicate_slots / max(target_total_slots, 1)
                    ),
                }
            )
        if view_aware_sampler is not None:
            epoch_metrics.update(
                {
                    "crop_view_draw_count": (
                        view_aware_sampler.last_draw_count
                    ),
                    "crop_view_unique_index_count": (
                        view_aware_sampler.last_unique_index_count
                    ),
                    "crop_view_repeated_draw_count": (
                        view_aware_sampler.last_repeated_draw_count
                    ),
                    "crop_view_repeated_draw_fraction": (
                        view_aware_sampler.last_repeated_draw_count
                        / max(view_aware_sampler.last_draw_count, 1)
                    ),
                    "crop_view_max_id": (
                        view_aware_sampler.last_max_view_id
                    ),
                }
            )
        append_jsonl(metrics_path, epoch_metrics)
        print(
            "Epoch summary "
            + json.dumps(epoch_metrics, sort_keys=True),
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
            or args.max_train_batches > 0
            or max_steps_reached
        ):
            checkpoint_path = (
                args.run_dir
                / (
                    f"emotion-query-mamba-epoch{epoch:04d}"
                    f"-seed{args.seed}.pth"
                )
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                anchor_sha256,
                style_cache_sha256,
                warmstart_sha256,
                emotion_warmstart_sha256,
                prototype_counts,
                emotion_priors,
                target_support_prior,
                emotion_warmstart_method,
                session_allowlist,
                session_split_manifest_sha256,
                sampling_protocol,
                epoch_metrics,
                isolated_va_optimizer,
                random_joint_warmstart_sha256=(
                    random_joint_warmstart_sha256
                ),
                random_joint_warmstart_method=(
                    random_joint_warmstart_method
                ),
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
                    "training_stage": args.training_stage,
                    "isolated_va_calibration": args.isolated_va_calibration,
                    "isolated_va_start_epoch": args.isolated_va_start_epoch,
                    "isolated_va_reads_B_cal": False,
                    "isolated_va_reads_B_confirm": False,
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
