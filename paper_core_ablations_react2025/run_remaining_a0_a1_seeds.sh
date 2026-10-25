#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source ./env.sh

EXP=paper_core_ablations_react2025
CACHE=coreset_reactor/reports/multiview_boundary_student_seed1_20261018/train_cache_head_center_tail/manifest.json
INIT=coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3/arms/T1/rank_00/conditional-epoch0006-seed1.pth
A1_SEED1=coreset_reactor/reports/multiview_boundary_student_seed1_20261018/student_compute_matched_4epochs/student.pth
A1_RAW=coreset_reactor/reports/react2025_multiview_ablation_bidir_official5_all_session_20261022_v2/comparison.json
A1_TRANSITION=coreset_reactor/reports/react2025_p4_sparse_transition_official5_all_session_20261022/results.json

"$REACT_PYTHON" "$EXP/assemble_seed1_controls.py" \
  --a0-eval "$EXP/runs/a0_full_seed1/eval_combined" \
  --a0-checkpoint "$EXP/runs/a0_full_seed1/student.pth" \
  --a1-raw "$A1_RAW" --a1-transition "$A1_TRANSITION" \
  --a1-checkpoint "$A1_SEED1" \
  --a3-eval "$EXP/runs/a3_group_weighted_seed1/eval_combined" \
  --a3-checkpoint "$EXP/runs/a3_group_weighted_seed1/student.pth" \
  --a2-velocity-eval "$EXP/runs/a2_direct_velocity_seed1/eval_combined" \
  --a2-velocity-checkpoint "$EXP/runs/a2_direct_velocity_seed1/student.pth" \
  --a1-no-velocity-eval "$EXP/runs/a1_no_velocity_seed1/eval_combined" \
  --a1-no-velocity-checkpoint "$EXP/runs/a1_no_velocity_seed1/student.pth" \
  --a2-no-velocity-eval "$EXP/runs/a2_direct_no_velocity_seed1/eval_combined" \
  --a2-no-velocity-checkpoint "$EXP/runs/a2_direct_no_velocity_seed1/student.pth" \
  --a3-diagnostic "$EXP/runs/seed1_a3_difference_diagnostic/summary.json" \
  --gradient-diagnostic "$EXP/runs/a0_a3_gradient_diagnostic.json" \
  --a2-cache-audit "$EXP/manifests/A2_cache_audit.json" \
  --output-dir "$EXP" | tee "$EXP/runs/assemble_seed1_controls.log"

for seed in 2 3; do
  CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
    --data-root data --cache-manifest "$CACHE" \
    --run-dir "$EXP/runs/a0_full_seed${seed}" --device cuda:0 \
    --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed "$seed" \
    --expert-embedding-dim 64 --initialize-backbone "$INIT" \
    --velocity-weight 0.05 --relation-grad-ratio 0.25 \
    --relation-objective centered_residual --print-every 100 --save-every 1 \
    > "$EXP/runs/a0_full_seed${seed}.train.log" 2>&1 &
  a0_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
    --data-root data --cache-manifest "$CACHE" \
    --run-dir "$EXP/runs/a1_trajectory_seed${seed}" --device cuda:0 \
    --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed "$seed" \
    --expert-embedding-dim 64 --initialize-backbone "$INIT" \
    --velocity-weight 0.05 --relation-grad-ratio 0 \
    --relation-objective distance --print-every 100 --save-every 1 \
    > "$EXP/runs/a1_trajectory_seed${seed}.train.log" 2>&1 &
  a1_pid=$!
  wait "$a0_pid"
  wait "$a1_pid"
done

"$REACT_PYTHON" "$EXP/audit_seed_repeats.py" \
  --experiment-dir "$EXP" --output "$EXP/manifests/A0_A1_seed_repeats_audit.json" \
  | tee "$EXP/runs/audit_seed_repeats.log"

for seed in 2 3; do
  for arm in a0_full a1_trajectory; do
    "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
      --output "$EXP/runs/${arm}_seed${seed}/eval_combined" \
      --checkpoint "$EXP/runs/${arm}_seed${seed}/student.pth" --prepare
  done
  CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/a0_full_seed${seed}/eval_combined" --shard 0 --shards 1 \
    | tee "$EXP/runs/a0_full_seed${seed}/eval_combined/shard0.log" &
  a0_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/a1_trajectory_seed${seed}/eval_combined" --shard 0 --shards 1 \
    | tee "$EXP/runs/a1_trajectory_seed${seed}/eval_combined/shard0.log" &
  a1_pid=$!
  wait "$a0_pid"
  wait "$a1_pid"
  for arm in a0_full a1_trajectory; do
    "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
      --output "$EXP/runs/${arm}_seed${seed}/eval_combined" --aggregate \
      | tee "$EXP/runs/${arm}_seed${seed}/eval_combined/aggregate.log"
  done
done

"$REACT_PYTHON" "$EXP/assemble_multiseed_a0_a1.py" \
  --experiment-dir "$EXP" --seed1-a1-raw "$A1_RAW" \
  --seed1-a1-transition "$A1_TRANSITION" \
  --seed1-a1-checkpoint "$A1_SEED1" \
  | tee "$EXP/runs/assemble_multiseed_a0_a1.log"
