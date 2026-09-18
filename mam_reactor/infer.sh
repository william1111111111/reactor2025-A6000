#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REACT_PYTHON:-$ROOT_DIR/../.venv/bin/python}"
DATA_DIR="${REACT_DATA_DIR:-}"
TASK="${MAM_REACTOR_TASK:-offline}"
VARIANT="${MAM_REACTOR_VARIANT:-offline_evidence}"
ASSET_DIR="${MAM_REACTOR_ASSET_PROJECT_DIR:-$ROOT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${MAM_REACTOR_OUTPUT_DIR:-$ROOT_DIR/runs/inference_${TASK}_${STAMP}}"

if [[ -z "$DATA_DIR" ]]; then
  echo "Set REACT_DATA_DIR to the external REACT2025 data directory." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "Data directory not found: $DATA_DIR" >&2; exit 2; }

case "$VARIANT" in
  offline|offline_evidence)
    TASK="offline"
    CHECKPOINT="${MAM_REACTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/mam_reactor_offline_evidence_epoch0003.pth}"
    CLIP_LENGTH="${MAM_REACTOR_CLIP_LENGTH:-750}"
    ;;
  online_frozen_offline_anchor|online_frozen)
    TASK="online"
    CHECKPOINT="${MAM_REACTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/mam_reactor_online_frozen_offline_anchor_epoch0010.pth}"
    CLIP_LENGTH="${MAM_REACTOR_CLIP_LENGTH:-60}"
    ;;
  online_adapted_anchor|online_adapted)
    TASK="online"
    CHECKPOINT="${MAM_REACTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/mam_reactor_online_adapted_anchor_epoch0010.pth}"
    CLIP_LENGTH="${MAM_REACTOR_CLIP_LENGTH:-60}"
    ;;
  *)
    echo "Unknown MAM_REACTOR_VARIANT: $VARIANT" >&2
    echo "Use offline_evidence, online_frozen_offline_anchor, or online_adapted_anchor." >&2
    exit 2
    ;;
esac

TARGET_CACHE="${REACT_TARGET_CACHE_RESULTS_PT:-${MAM_REACTOR_TARGET_CACHE_RESULTS_PT:-$ROOT_DIR/../regnn/cache/official_test1142_targets_full_seed1234.pt}}"
RESULTS_PT="${MAM_REACTOR_RESULTS_PT:-$OUT_DIR/results_frd.pt}"
METRICS_JSON="${MAM_REACTOR_METRICS_JSON:-$OUT_DIR/metrics.json}"

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
mkdir -p "$OUT_DIR"
export MAM_REACTOR_ROOT="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MAM_REACTOR_MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${MAM_REACTOR_GPU_ID:-6}}"

cmd=(
  "$PYTHON_BIN" -m regnn.eval_emotion_query_mamba_official_test
  --checkpoint "$CHECKPOINT"
  --data-dir "$DATA_DIR"
  --asset-project-dir "$ASSET_DIR"
  --metrics-json "$METRICS_JSON"
  --results-pt "$RESULTS_PT"
  --task "$TASK"
  --clip-length "$CLIP_LENGTH"
  --loader-batch-size "${MAM_REACTOR_LOADER_BATCH_SIZE:-4}"
  --loader-workers "${MAM_REACTOR_LOADER_WORKERS:-4}"
  --clip-batch-size "${MAM_REACTOR_CLIP_BATCH_SIZE:-8}"
  --metric-workers "${MAM_REACTOR_METRIC_WORKERS:-4}"
  --selection-seed "${MAM_REACTOR_SELECTION_SEED:-1234}"
  --residual-scale "${MAM_REACTOR_RESIDUAL_SCALE:-0.95}"
  --style-residual-scale "${MAM_REACTOR_STYLE_RESIDUAL_SCALE:-0.95}"
  --au-residual-multiplier "${MAM_REACTOR_AU_RESIDUAL_MULTIPLIER:-1.8}"
  --va-residual-multiplier "${MAM_REACTOR_VA_RESIDUAL_MULTIPLIER:-0.4}"
  --expression-residual-multiplier "${MAM_REACTOR_EXPRESSION_RESIDUAL_MULTIPLIER:-1.6}"
)

if [[ "$TASK" == online ]]; then
  cmd+=(
    --online-window-size "${MAM_REACTOR_ONLINE_WINDOW_SIZE:-30}"
    --online-speaker-ratio "${MAM_REACTOR_ONLINE_SPEAKER_RATIO:-2}"
    --skip-anchor-metrics
  )
fi

if [[ -f "$TARGET_CACHE" ]]; then
  cmd+=(--target-cache-results-pt "$TARGET_CACHE")
elif [[ ! -f "$ASSET_DIR/pretrained_models/post_processor/checkpoint.pth" ]]; then
  echo "No official target cache or post-processor checkpoint found." >&2
  echo "Set REACT_TARGET_CACHE_RESULTS_PT, or set MAM_REACTOR_ASSET_PROJECT_DIR to a complete evaluator project." >&2
  exit 2
fi

cd "$ROOT_DIR"
exec "${cmd[@]}"
