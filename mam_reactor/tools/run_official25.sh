#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65536
cd /public/zhaowenjie/react2025_new/mam_reactor
export CUDA_VISIBLE_DEVICES=5
export REACT_DATA_DIR=/public/zhaowenjie/react2025_new/data
export REACT_TARGET_CACHE_RESULTS_PT="$PWD/assets/evaluation/generated_local_test1142_seed1234.pt"
export MAM_REACTOR_ROOT="$PWD"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MAM_REACTOR_FRD_WORKERS=16
base="$PWD/runs/official25_comparison_20260917_v3"
mkdir "$base"
for variant in joint baseline; do
  if [[ "$variant" == joint ]]; then
    checkpoint="$PWD/runs/joint_3dmm_trial_20260917/emotion-query-mamba-epoch0003-seed1.pth"
  else
    checkpoint="$PWD/runs/offline_20260917_172106/emotion-query-mamba-epoch0003-seed1.pth"
  fi
  mkdir "$base/$variant"
  ../.venv/bin/python -u -m regnn.eval_emotion_query_mamba_official_test \
    --checkpoint "$checkpoint" --data-dir "$REACT_DATA_DIR" --asset-project-dir "$PWD" \
    --metrics-json "$base/$variant/metrics.json" --results-pt "$base/$variant/results_frd.pt" \
    --task offline --clip-length 750 --loader-batch-size 4 --loader-workers 4 \
    --clip-batch-size 8 --metric-workers 8 --selection-seed 1234 \
    --residual-scale 0.95 --style-residual-scale 0.95 \
    --au-residual-multiplier 1.8 --va-residual-multiplier 0.4 --expression-residual-multiplier 1.6 \
    --target-cache-results-pt "$REACT_TARGET_CACHE_RESULTS_PT" \
    --target-postprocess-batch-size 4 > "$base/$variant.log" 2>&1
  echo "$variant FRC/FRDiv/FRVar/TLCC complete"
done
for variant in joint baseline; do
  bash run_exact_frd.sh "$base/$variant/results_frd.pt" "$base/$variant/frd_exact" > "$base/$variant.frd.log" 2>&1 &
  if [[ "$variant" == joint ]]; then joint_pid=$!; else baseline_pid=$!; fi
done
wait "$joint_pid"
wait "$baseline_pid"
echo 'All official25 evaluations completed'
