#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REACT_DATA_DIR="${REACT_DATA_DIR:-$ROOT_DIR/../data}"
export MAM_REACTOR_GPU_ID="${MAM_REACTOR_GPU_ID:-5}"
export EQR_LISTENER_3DMM_WEIGHT="${EQR_LISTENER_3DMM_WEIGHT:-1.0}"
export EQR_LISTENER_3DMM_VELOCITY_WEIGHT="${EQR_LISTENER_3DMM_VELOCITY_WEIGHT:-0.1}"
export MAM_REACTOR_RUN_DIR="${MAM_REACTOR_RUN_DIR:-$ROOT_DIR/runs/joint_3dmm_$(date +%Y%m%d_%H%M%S)}"
export EQR_EXPERIMENT_TAG=mam_reactor_joint_listener_3dmm
exec bash "$ROOT_DIR/train.sh"
