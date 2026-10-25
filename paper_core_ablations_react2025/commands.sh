#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source ./env.sh

EXP=paper_core_ablations_react2025
CACHE=coreset_reactor/reports/multiview_boundary_student_seed1_20261018/train_cache_head_center_tail/manifest.json
INIT=coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3/arms/T1/rank_00/conditional-epoch0006-seed1.pth
A1=coreset_reactor/reports/multiview_boundary_student_seed1_20261018/student_compute_matched_4epochs/student.pth
A0="$EXP/runs/a0_full_seed1/student.pth"

# A0 seed 1. A1 is the provenance-matched existing run and is not retrained.
CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$CACHE" \
  --run-dir "$EXP/runs/a0_full_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0.05 --relation-grad-ratio 0.25 \
  --relation-objective centered_residual --print-every 100 --save-every 1

# Raw p=4 and the fixed query-0 postprocess in one inference pass.  Unchanged
# rows reuse the raw metrics, so exact DTW is only repeated for changed rows.
"$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/a0_full_seed1/eval_combined" --checkpoint "$A0" --prepare
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="$shard" "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/a0_full_seed1/eval_combined" --shard "$shard" --shards 2 &
done
wait
"$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/a0_full_seed1/eval_combined" --aggregate

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/diagnose_difference_transfer.py" \
  --cache-manifest "$CACHE" --data-root data --a0 "$A0" --a1 "$A1" \
  --output "$EXP/runs/seed1_difference_diagnostic" --device cuda:0

"$REACT_PYTHON" "$EXP/assemble_a0_a1_results.py" \
  --a0-raw "$EXP/runs/a0_full_seed1/eval_combined/comparison_raw.json" \
  --a0-post "$EXP/runs/a0_full_seed1/eval_combined/comparison_postprocessed.json" \
  --a1-raw coreset_reactor/reports/react2025_multiview_ablation_bidir_official5_all_session_20261022_v2/comparison.json \
  --a1-transition coreset_reactor/reports/react2025_p4_sparse_transition_official5_all_session_20261022/results.json \
  --a0-checkpoint "$A0" --a1-checkpoint "$A1" \
  --diagnostic "$EXP/runs/seed1_difference_diagnostic/summary.json" \
  --output-dir "$EXP"

# Second-priority controls.  The real-reference cache keeps the source-cache
# row metadata and the ten preassigned target identities fixed.
FIXED_MANIFESTS=()
for rank_number in $(seq 0 9); do
  rank=$(printf '%02d' "$rank_number")
  FIXED_MANIFESTS+=(
    "coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3/manifests/T1/rank_${rank}.json"
  )
done
CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" \
  "$EXP/build_multiview_real_gt_cache.py" \
  --data-root data --source-manifest "$CACHE" \
  --fixed-manifests "${FIXED_MANIFESTS[@]}" \
  --output "$EXP/runs/a2_multiview_real_gt_cache"

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$CACHE" \
  --run-dir "$EXP/runs/a3_group_weighted_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0.05 \
  --frozen-relation-provenance "$EXP/configs/A3_frozen_A0_scale_weight.json" \
  --relation-objective group_weighted_mse --print-every 100 --save-every 1

CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$CACHE" \
  --run-dir "$EXP/runs/a1_no_velocity_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0 --relation-grad-ratio 0 \
  --relation-objective distance --print-every 100 --save-every 1

A2_CACHE="$EXP/runs/a2_multiview_real_gt_cache/manifest.json"
CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$A2_CACHE" \
  --run-dir "$EXP/runs/a2_direct_velocity_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0.05 --relation-grad-ratio 0 \
  --relation-objective distance --print-every 100 --save-every 1

CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$A2_CACHE" \
  --run-dir "$EXP/runs/a2_direct_no_velocity_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0 --relation-grad-ratio 0 \
  --relation-objective distance --print-every 100 --save-every 1

"$REACT_PYTHON" "$EXP/audit_a2_cache.py" \
  --teacher-manifest "$CACHE" --direct-manifest "$A2_CACHE" \
  --output "$EXP/manifests/A2_cache_audit.json"

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/diagnose_a0_a3_gradients.py" \
  --data-root data --cache-manifest "$CACHE" --initializer "$INIT" \
  --a0-dir "$EXP/runs/a0_full_seed1" \
  --a3-dir "$EXP/runs/a3_group_weighted_seed1" \
  --output "$EXP/runs/a0_a3_gradient_diagnostic.json" \
  --device cuda:0 --batch-size 4

CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" "$EXP/diagnose_a3_difference_control.py" \
  --cache-manifest "$CACHE" --data-root data --a0 "$A0" --a1 "$A1" \
  --a3 "$EXP/runs/a3_group_weighted_seed1/student.pth" \
  --output "$EXP/runs/seed1_a3_difference_diagnostic" \
  --device cuda:0 --batch-size 4

# Each formal control evaluation writes raw and fixed-postprocess VAL1142 rows.
# The four runs were evaluated two at a time on the two authorized free GPUs.
for arm in a3_group_weighted_seed1 a2_direct_velocity_seed1 \
           a1_no_velocity_seed1 a2_direct_no_velocity_seed1; do
  "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/$arm/eval_combined" \
    --checkpoint "$EXP/runs/$arm/student.pth" --prepare
done

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/a2_direct_velocity_seed1/eval_combined" \
  --shard 0 --shards 1 &
CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/a3_group_weighted_seed1/eval_combined" \
  --shard 0 --shards 1 &
wait
for arm in a2_direct_velocity_seed1 a3_group_weighted_seed1; do
  "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/$arm/eval_combined" --aggregate
done

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/a1_no_velocity_seed1/eval_combined" \
  --shard 0 --shards 1 &
CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/a2_direct_no_velocity_seed1/eval_combined" \
  --shard 0 --shards 1 &
wait
for arm in a1_no_velocity_seed1 a2_direct_no_velocity_seed1; do
  "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/$arm/eval_combined" --aggregate
done

"$REACT_PYTHON" "$EXP/assemble_seed1_controls.py" \
  --a0-eval "$EXP/runs/a0_full_seed1/eval_combined" \
  --a0-checkpoint "$EXP/runs/a0_full_seed1/student.pth" \
  --a1-raw coreset_reactor/reports/react2025_multiview_ablation_bidir_official5_all_session_20261022_v2/comparison.json \
  --a1-transition coreset_reactor/reports/react2025_p4_sparse_transition_official5_all_session_20261022/results.json \
  --a1-checkpoint "$A1" \
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
  --output-dir "$EXP"

# Predeclared student-seed repeats for the core A0/A1 comparison.  This runner
# freezes the seed-specific complete initial state and observed source schedule,
# trains both arms, evaluates full bidirectional VAL1142, and assembles the
# per-seed session bootstrap plus across-seed mean/sample SD.
bash "$EXP/run_remaining_a0_a1_seeds.sh"

# Matched fixed/uniform/target-specific reference-prior controls.  The target
# arm reuses the audited seed-1 A0 assets; fixed and uniform caches/students are
# constructed with the same source views, initialization, budget, and frozen A0
# centered-residual scales/weight before all teacher and student arms receive
# the same formal VAL1142 evaluation.
bash "$EXP/run_reference_controls.sh"
