#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REACT_PYTHON:-$ROOT_DIR/../.venv/bin/python}"
DATA_DIR="${REACT_DATA_DIR:-}"
GPU_ID="${MAM_REACTOR_GPU_ID:-6}"
RUN_DIR="${MAM_REACTOR_RUN_DIR:-$ROOT_DIR/runs/offline_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$DATA_DIR" ]]; then
  echo "Set REACT_DATA_DIR to the external REACT2025 data directory." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "Data directory not found: $DATA_DIR" >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run directory: $RUN_DIR" >&2; exit 3; }

export MAM_REACTOR_ROOT="$ROOT_DIR"
export EQR_PROJECT_DIR="$ROOT_DIR"
export REACT_PYTHON="$PYTHON_BIN"
export REACT_DATA_DIR="$DATA_DIR"
export EQR_ANCHOR_CHECKPOINT="${EQR_ANCHOR_CHECKPOINT:-$ROOT_DIR/checkpoints/conditional_regnn_offline_anchor_epoch0050.pth}"
export EQR_STYLE_CACHE="${EQR_STYLE_CACHE:-$ROOT_DIR/assets/train_reaction_style_v1.pt}"
export EQR_WARMSTART_CHECKPOINT="${EQR_WARMSTART_CHECKPOINT:-$ROOT_DIR/checkpoints/eqr_legacy_warmstart_epoch0003.pth}"
export EQR_ANCHOR_INITIALIZATION="${EQR_ANCHOR_INITIALIZATION:-pretrained_frozen}"
export EQR_GPU_ID="$GPU_ID"
export EQR_RUN_DIR="$RUN_DIR"
export EQR_EXPERIMENT_TAG="${EQR_EXPERIMENT_TAG:-mam_reactor_offline_evidence_repro}"
export EQR_EPOCHS="${EQR_EPOCHS:-3}"
export EQR_BATCH_SIZE="${EQR_BATCH_SIZE:-16}"
export EQR_WORKERS="${EQR_WORKERS:-4}"
export EQR_CLIP_LENGTH="${EQR_CLIP_LENGTH:-750}"
export EQR_SUPERVISION_TAIL_FRAMES="${EQR_SUPERVISION_TAIL_FRAMES:-0}"
export EQR_MIN_FREE_MIB="${EQR_MIN_FREE_MIB:-8192}"
export EQR_MAX_GPU_UTIL="${EQR_MAX_GPU_UTIL:-10}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

for path in "$EQR_ANCHOR_CHECKPOINT" "$EQR_STYLE_CACHE" "$EQR_WARMSTART_CHECKPOINT" "$ROOT_DIR/regnn/configs/eqr_session_splits_seed1.json"; do
  [[ -f "$path" ]] || { echo "Missing Mam-Reactor input: $path" >&2; exit 2; }
done

cd "$ROOT_DIR"
exec bash "$ROOT_DIR/launch/train_eqr_joint_isolated_va_channelot_bfit.sh"
