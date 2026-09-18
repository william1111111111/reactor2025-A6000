#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MAM_REACTOR_ROOT="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="${MAM_REACTOR_GPU_ID:-5}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -n 65536
run_dir="${MAM_ANCHOR_RUN_DIR:-$ROOT_DIR/runs/joint_anchor_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$run_dir"
cd "$ROOT_DIR"
exec "$ROOT_DIR/../.venv/bin/python" -u -m regnn.train_conditional_regnn \
  --data-dir "$ROOT_DIR/../data" --run-dir "$run_dir" \
  --epochs "${MAM_ANCHOR_EPOCHS:-50}" --batch-size "${MAM_ANCHOR_BATCH_SIZE:-32}" --workers 4 \
  --clip-length 750 --lr 1e-4 --weight-decay 1e-4 --ccc-weight 1 --velocity-weight 0.05 \
  --listener-3dmm-weight 1.0 --listener-3dmm-velocity-weight 0.1 \
  --seed 1 --save-every 1 --print-every 20 --precision bf16
