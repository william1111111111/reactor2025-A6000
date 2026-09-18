#!/usr/bin/env bash
set -euo pipefail
cd /public/zhaowenjie/react2025_new/mam_reactor
ulimit -n 65536
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD" MAM_REACTOR_ROOT="$PWD" MAM_REACTOR_FRD_WORKERS=16
run="$PWD/runs/joint_anchor_scratch_20260917"
for attempt in $(seq 1 360); do
  if [[ -f "$run/TRAINING_COMPLETE" && -f "$run/conditional-epoch0050-seed1.pth" ]]; then break; fi
  sleep 10
done
test -f "$run/TRAINING_COMPLETE"
test -f "$run/conditional-epoch0050-seed1.pth"
../.venv/bin/python -u tools/evaluate_anchor_pair.py
base="$PWD/runs/anchor_comparison_20260917"
bash run_exact_frd.sh "$base/old/results_frd.pt" "$base/old/frd_exact" > "$base/old.frd.log" 2>&1 &
old_pid=$!
bash run_exact_frd.sh "$base/new/results_frd.pt" "$base/new/frd_exact" > "$base/new.frd.log" 2>&1 &
new_pid=$!
wait "$old_pid"
wait "$new_pid"
../.venv/bin/python tools/summarize_anchor_comparison.py
echo 'Anchor comparison complete'
