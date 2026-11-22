#!/usr/bin/env bash
set -euo pipefail

# Run from the release directory. Device numbering is controlled by the caller's
# CUDA_VISIBLE_DEVICES. Each training stage writes into a fresh run directory.
data_root="${1:?Usage: bash scripts/train_react2025.sh DATA_ROOT [RUN_ROOT]}"
run_root="${2:-runs/react2025_seed1}"
react_python="${CORESET_PYTHON:-python}"
assignment="configs/react2025_reference_assignment.json"

if [[ -e "$run_root" ]]; then
    echo "Run directory already exists: $run_root" >&2
    exit 1
fi
mkdir -p "$run_root"

"$react_python" -m regnn.train_conditional_regnn \
    --data-dir "$data_root" --run-dir "$run_root/teacher_00" \
    --fixed-pair-manifest "$assignment" --fixed-pair-rank 0 \
    --fixed-pair-alignment-policy relative_time_masked --fixed-pair-seed 1 \
    --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
    --seed 1 --precision bf16 --max-train-steps 600 --snapshot-steps 600

teachers=("$run_root/teacher_00/conditional-step000600-seed1.pth")
initializer="${CORESET_INITIALIZER:-${teachers[0]}}"
for rank in {1..9}; do
    printf -v teacher_dir '%s/teacher_%02d' "$run_root" "$rank"
    "$react_python" -m coreset_reactor.train_mode_reference_teacher \
        --data-dir "$data_root" --run-dir "$teacher_dir" \
        --fixed-pair-manifest "$assignment" --fixed-pair-rank "$rank" \
        --fixed-pair-alignment-policy relative_time_masked --fixed-pair-seed 1 \
        --reference-temperature 0.05 --reference-effective-count 8 \
        --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
        --seed 1 --precision bf16 --max-train-steps 600 --snapshot-steps 600
    teachers+=("$teacher_dir/conditional-step000600-seed1.pth")
done

"$react_python" -m coreset_reactor.teacher_cache \
    --data-root "$data_root" --teacher-checkpoints "${teachers[@]}" \
    --output-dir "$run_root/train_cache" --device cuda:0 \
    --batch-size 4 --crop-views 3 --crop-view-mode stratified

"$react_python" -m coreset_reactor.train_expert_student \
    --data-root "$data_root" --cache-manifest "$run_root/train_cache/manifest.json" \
    --run-dir "$run_root/student" --device cuda:0 \
    --initialize-backbone "$initializer" \
    --epochs 4 --batch-size 4 --workers 4 --lr 1e-4 --seed 1 \
    --velocity-weight 0.05 --relation-grad-ratio 0.25 \
    --relation-objective centered_residual --save-every 1 --print-every 100

"$react_python" scripts/export_checkpoint.py \
    --source "$run_root/student/student.pth" \
    --output "$run_root/coreset_react2025_student.pth" \
    --normalization-dir regnn/assets
