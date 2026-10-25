#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source ./env.sh

EXP=paper_core_ablations_react2025
AUDIT="$EXP/manifests/reference_teacher_banks_audit.json"
Q0=coreset_reactor/reports/teacher_convergence_2400_seed1/arms/T1/rank_00/conditional-step000600-seed1.pth
INIT=coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3/arms/T1/rank_00/conditional-epoch0006-seed1.pth
TARGET_CACHE=coreset_reactor/reports/multiview_boundary_student_seed1_20261018/train_cache_head_center_tail/manifest.json
TARGET_STUDENT="$EXP/runs/a0_full_seed1/student.pth"

FIXED=()
UNIFORM=("$Q0")
for rank_number in $(seq 0 9); do
  rank=$(printf '%02d' "$rank_number")
  FIXED+=("coreset_reactor/reports/teacher_convergence_2400_seed1/arms/T1/rank_${rank}/conditional-step000600-seed1.pth")
  if [[ "$rank_number" -gt 0 ]]; then
    UNIFORM+=("coreset_reactor/reports/react2025_key_component_ablations_seed1/uniform/rank_${rank}/conditional-step000600-seed1.pth")
  fi
done

"$REACT_PYTHON" "$EXP/audit_reference_teacher_banks.py" \
  --project-root . --output "$AUDIT" \
  | tee "$EXP/runs/audit_reference_teacher_banks.log"

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" -m coreset_reactor.teacher_cache \
  --data-root data --teacher-checkpoints "${FIXED[@]}" \
  --output-dir "$EXP/runs/reference_fixed_cache" --device cuda:0 \
  --batch-size 4 --workers 4 --view-epoch 0 --crop-views 3 \
  --crop-view-mode stratified --shard-size 64 \
  > "$EXP/runs/reference_fixed_cache.log" 2>&1 &
fixed_cache_pid=$!
CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" -m coreset_reactor.teacher_cache \
  --data-root data --teacher-checkpoints "${UNIFORM[@]}" \
  --output-dir "$EXP/runs/reference_uniform_cache" --device cuda:0 \
  --batch-size 4 --workers 4 --view-epoch 0 --crop-views 3 \
  --crop-view-mode stratified --shard-size 64 \
  > "$EXP/runs/reference_uniform_cache.log" 2>&1 &
uniform_cache_pid=$!
wait "$fixed_cache_pid"
wait "$uniform_cache_pid"

"$REACT_PYTHON" "$EXP/audit_reference_student_caches.py" \
  --target-manifest "$TARGET_CACHE" \
  --fixed-manifest "$EXP/runs/reference_fixed_cache/manifest.json" \
  --uniform-manifest "$EXP/runs/reference_uniform_cache/manifest.json" \
  --bank-audit "$AUDIT" \
  --a0-provenance "$EXP/runs/a0_full_seed1/provenance.json" \
  --output "$EXP/manifests/reference_student_caches_audit.json" \
  --calibration-dir "$EXP/configs/reference_students" \
  | tee "$EXP/runs/audit_reference_student_caches.log"

CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$EXP/runs/reference_fixed_cache/manifest.json" \
  --run-dir "$EXP/runs/reference_fixed_student_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0.05 \
  --frozen-relation-provenance "$EXP/configs/reference_students/fixed_A0_frozen_calibration.json" \
  --relation-objective centered_residual --print-every 100 --save-every 1 \
  > "$EXP/runs/reference_fixed_student_seed1.train.log" 2>&1 &
fixed_student_pid=$!
CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" -m coreset_reactor.train_expert_student \
  --data-root data --cache-manifest "$EXP/runs/reference_uniform_cache/manifest.json" \
  --run-dir "$EXP/runs/reference_uniform_student_seed1" --device cuda:0 \
  --epochs 4 --batch-size 4 --workers 4 --lr 0.0001 --seed 1 \
  --expert-embedding-dim 64 --initialize-backbone "$INIT" \
  --velocity-weight 0.05 \
  --frozen-relation-provenance "$EXP/configs/reference_students/uniform_A0_frozen_calibration.json" \
  --relation-objective centered_residual --print-every 100 --save-every 1 \
  > "$EXP/runs/reference_uniform_student_seed1.train.log" 2>&1 &
uniform_student_pid=$!
wait "$fixed_student_pid"
wait "$uniform_student_pid"

"$REACT_PYTHON" "$EXP/audit_reference_students.py" \
  --fixed "$EXP/runs/reference_fixed_student_seed1/student.pth" \
  --uniform "$EXP/runs/reference_uniform_student_seed1/student.pth" \
  --target-specific "$TARGET_STUDENT" \
  --cache-audit "$EXP/manifests/reference_student_caches_audit.json" \
  --bank-audit "$AUDIT" \
  --output "$EXP/manifests/reference_students_audit.json" \
  | tee "$EXP/runs/audit_reference_students.log"

TEACHER_EVAL="$EXP/runs/reference_teacher_banks_eval"
"$REACT_PYTHON" "$EXP/eval_reference_teacher_banks_p4_bidir.py" \
  --output "$TEACHER_EVAL" --bank-audit "$AUDIT" --prepare
CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/eval_reference_teacher_banks_p4_bidir.py" \
  --output "$TEACHER_EVAL" --shard 0 --shards 2 \
  | tee "$TEACHER_EVAL/shard0.log" &
teacher0_pid=$!
CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" "$EXP/eval_reference_teacher_banks_p4_bidir.py" \
  --output "$TEACHER_EVAL" --shard 1 --shards 2 \
  | tee "$TEACHER_EVAL/shard1.log" &
teacher1_pid=$!
wait "$teacher0_pid"
wait "$teacher1_pid"
"$REACT_PYTHON" "$EXP/eval_reference_teacher_banks_p4_bidir.py" \
  --output "$TEACHER_EVAL" --aggregate | tee "$TEACHER_EVAL/aggregate.log"

for arm in fixed uniform; do
  "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/reference_${arm}_student_seed1/eval_combined" \
    --checkpoint "$EXP/runs/reference_${arm}_student_seed1/student.pth" --prepare
done
CUDA_VISIBLE_DEVICES=0 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/reference_fixed_student_seed1/eval_combined" \
  --shard 0 --shards 1 \
  | tee "$EXP/runs/reference_fixed_student_seed1/eval_combined/shard0.log" &
fixed_eval_pid=$!
CUDA_VISIBLE_DEVICES=1 "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
  --output "$EXP/runs/reference_uniform_student_seed1/eval_combined" \
  --shard 0 --shards 1 \
  | tee "$EXP/runs/reference_uniform_student_seed1/eval_combined/shard0.log" &
uniform_eval_pid=$!
wait "$fixed_eval_pid"
wait "$uniform_eval_pid"
for arm in fixed uniform; do
  "$REACT_PYTHON" "$EXP/eval_raw_p4_bidir.py" \
    --output "$EXP/runs/reference_${arm}_student_seed1/eval_combined" --aggregate \
    | tee "$EXP/runs/reference_${arm}_student_seed1/eval_combined/aggregate.log"
done

"$REACT_PYTHON" "$EXP/assemble_reference_controls.py" \
  --teacher-eval "$TEACHER_EVAL" \
  --fixed-student-eval "$EXP/runs/reference_fixed_student_seed1/eval_combined" \
  --uniform-student-eval "$EXP/runs/reference_uniform_student_seed1/eval_combined" \
  --target-student-eval "$EXP/runs/a0_full_seed1/eval_combined" \
  --bank-audit "$AUDIT" \
  --cache-audit "$EXP/manifests/reference_student_caches_audit.json" \
  --student-audit "$EXP/manifests/reference_students_audit.json" \
  --output-dir "$EXP" | tee "$EXP/runs/assemble_reference_controls.log"
