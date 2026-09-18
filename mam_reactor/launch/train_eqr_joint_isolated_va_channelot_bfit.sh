#!/usr/bin/env bash
set -euo pipefail

# B_fit-only continuation of the Channel-OT EQR checkpoint. The ordinary EQR
# update remains intact. Beginning at EQR_VA_START_EPOCH, every main update is
# followed by one isolated micro-update whose optimizer can change only the
# existing to_residual rows/biases 15:17 (258 scalars at d_model=128).
: "${EQR_PROJECT_DIR:?Set EQR_PROJECT_DIR}"
: "${REACT_PYTHON:?Set REACT_PYTHON}"
: "${REACT_DATA_DIR:?Set REACT_DATA_DIR}"
: "${EQR_ANCHOR_CHECKPOINT:?Set EQR_ANCHOR_CHECKPOINT}"
: "${EQR_STYLE_CACHE:?Set EQR_STYLE_CACHE}"
: "${EQR_ANCHOR_INITIALIZATION:=pretrained_frozen}"
if [[ "$EQR_ANCHOR_INITIALIZATION" == pretrained_frozen ]]; then
  : "${EQR_WARMSTART_CHECKPOINT:=}"
elif [[ "$EQR_ANCHOR_INITIALIZATION" != random_joint ]]; then
  echo "EQR_ANCHOR_INITIALIZATION must be pretrained_frozen or random_joint" >&2
  exit 6
fi
: "${EQR_GPU_ID:?Set EQR_GPU_ID}"
: "${EQR_RUN_DIR:?Set EQR_RUN_DIR to a new directory}"

split_manifest="$EQR_PROJECT_DIR/regnn/configs/eqr_session_splits_seed1.json"
for path in \
  "$EQR_ANCHOR_CHECKPOINT" \
  "$EQR_STYLE_CACHE" \
  "$split_manifest"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
if [[ "$EQR_ANCHOR_INITIALIZATION" == pretrained_frozen ]]; then
  [[ -z "$EQR_WARMSTART_CHECKPOINT" || -f "$EQR_WARMSTART_CHECKPOINT" ]] || {
    echo "Missing required input: $EQR_WARMSTART_CHECKPOINT" >&2
    exit 2
  }
fi
[[ ! -e "$EQR_RUN_DIR" ]] || {
  echo "Refusing to overwrite: $EQR_RUN_DIR" >&2
  exit 3
}

gpu_state="$(nvidia-smi \
  --query-gpu=index,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  | awk -F', *' -v gpu="$EQR_GPU_ID" '$1 == gpu {print $2 "," $3 "," $4}')"
IFS=',' read -r gpu_total_mib gpu_used_mib gpu_util <<<"$gpu_state"
gpu_free_mib=$((gpu_total_mib - gpu_used_mib))
if [[ -z "$gpu_state" \
  || "$gpu_free_mib" -lt "${EQR_MIN_FREE_MIB:-24576}" \
  || "$gpu_util" -gt "${EQR_MAX_GPU_UTIL:-10}" ]]; then
  echo "GPU $EQR_GPU_ID is not safely available (total,used,util: ${gpu_state:-unknown})" >&2
  exit 4
fi

cd "$EQR_PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$EQR_GPU_ID"
export MPLCONFIGDIR="${EQR_MPLCONFIGDIR:-/tmp/mpl-eqr-joint-isolated-va}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

isolated_va_args=()
if [[ "${EQR_ENABLE_ISOLATED_VA:-1}" == "1" ]]; then
  isolated_va_args=(
    --isolated-va-calibration
    --isolated-va-start-epoch "${EQR_VA_START_EPOCH:-3}"
    --isolated-va-lr "${EQR_VA_LEARNING_RATE:-1e-5}"
    --isolated-va-tail-fraction 0.25
    --isolated-va-mean-risk-weight 0.25
    --isolated-va-proximal-weight 0.05
    --isolated-va-diversity-preservation-weight 10
    --isolated-va-variance-preservation-weight 10
    --isolated-va-preservation-ratio 0.99
    --isolated-va-gradient-clip 0.1
  )
elif [[ "${EQR_ENABLE_ISOLATED_VA}" != "0" ]]; then
  echo "EQR_ENABLE_ISOLATED_VA must be 0 or 1" >&2
  exit 5
fi

session_set_weight=${EQR_SESSION_SET_WEIGHT:-0.2}
sequence_transport_weight=${EQR_SEQUENCE_TRANSPORT_WEIGHT:-2.0}
learned_set_ccc_weight=${EQR_LEARNED_SET_CCC_WEIGHT:-0.0}
sequence_distance_mode=${EQR_SEQUENCE_DISTANCE_MODE:-ccc}
frd_transport_weight=${EQR_FRD_TRANSPORT_WEIGHT:-8.0}
query_identity_weight=${EQR_QUERY_IDENTITY_WEIGHT:-0.25}
diversity_weight=${EQR_DIVERSITY_WEIGHT:-1.0}
diversity_margin=${EQR_DIVERSITY_MARGIN:-0.05}
official_diversity_batch_weight=${EQR_OFFICIAL_DIVERSITY_BATCH_WEIGHT:-20.0}
channel_diversity_weight=${EQR_CHANNEL_DIVERSITY_WEIGHT:-20.0}
channel_diversity_budget=${EQR_CHANNEL_DIVERSITY_BUDGET:-0.15}
num_targets=${EQR_NUM_TARGETS:-10}
experiment_tag=${EQR_EXPERIMENT_TAG:-eqr_channelot_joint_isolated_va_bfit}
set_matching_strategy=${EQR_SET_MATCHING_STRATEGY:-balanced}
clip_length=${EQR_CLIP_LENGTH:-750}
supervision_tail_frames=${EQR_SUPERVISION_TAIL_FRAMES:-0}
learning_rate=${EQR_LEARNING_RATE:-2e-6}
frd_surrogate_target=${EQR_FRD_SURROGATE_TARGET:-140.0}
anchor_args=(--anchor-initialization "$EQR_ANCHOR_INITIALIZATION")
diversity_cap_args=()
crop_view_args=()
anchor_length_args=()
style_cache_contract_args=()
if [[ "${EQR_CHANNEL_DIVERSITY_NO_CAP:-0}" == 1 ]]; then
  diversity_cap_args+=(--channel-diversity-no-cap)
fi
if [[ "${EQR_VIEW_AWARE_REPLACEMENT_CROPS:-0}" == 1 ]]; then
  crop_view_args+=(--view-aware-replacement-crops)
elif [[ "${EQR_VIEW_AWARE_REPLACEMENT_CROPS:-0}" != 0 ]]; then
  echo "EQR_VIEW_AWARE_REPLACEMENT_CROPS must be 0 or 1" >&2
  exit 7
fi
if [[ "${EQR_ALLOW_ANCHOR_LENGTH_EXTENSION:-0}" == 1 ]]; then
  anchor_length_args+=(--allow-anchor-length-extension)
elif [[ "${EQR_ALLOW_ANCHOR_LENGTH_EXTENSION:-0}" != 0 ]]; then
  echo "EQR_ALLOW_ANCHOR_LENGTH_EXTENSION must be 0 or 1" >&2
  exit 8
fi
if [[ "${EQR_ALLOW_ANCHOR_LENGTH_SHRINK:-0}" == 1 ]]; then
  anchor_length_args+=(--allow-anchor-length-shrink)
elif [[ "${EQR_ALLOW_ANCHOR_LENGTH_SHRINK:-0}" != 0 ]]; then
  echo "EQR_ALLOW_ANCHOR_LENGTH_SHRINK must be 0 or 1" >&2
  exit 9
fi
if [[ "${EQR_REQUIRE_STYLE_CACHE_SESSION_MATCH:-0}" == 1 ]]; then
  style_cache_contract_args+=(--require-style-cache-session-match)
elif [[ "${EQR_REQUIRE_STYLE_CACHE_SESSION_MATCH:-0}" != 0 ]]; then
  echo "EQR_REQUIRE_STYLE_CACHE_SESSION_MATCH must be 0 or 1" >&2
  exit 10
fi
if [[ "$EQR_ANCHOR_INITIALIZATION" == pretrained_frozen ]]; then
  if [[ -n "$EQR_WARMSTART_CHECKPOINT" ]]; then
    anchor_args+=(--warmstart-emotion-checkpoint "$EQR_WARMSTART_CHECKPOINT")
  fi
else
  anchor_args+=(--anchor-lr "${EQR_ANCHOR_LR:-1e-4}")
fi

exec "$REACT_PYTHON" -m regnn.train_emotion_query_mamba \
  --anchor-checkpoint "$EQR_ANCHOR_CHECKPOINT" \
  --style-cache "$EQR_STYLE_CACHE" \
  "${style_cache_contract_args[@]}" \
  "${anchor_args[@]}" \
  --data-dir "$REACT_DATA_DIR" \
  --session-split-manifest "$split_manifest" \
  --session-split-name B_fit \
  --run-dir "$EQR_RUN_DIR" \
  --experiment-tag "$experiment_tag" \
  --epochs "${EQR_EPOCHS:-3}" \
  --batch-size "${EQR_BATCH_SIZE:-16}" \
  --workers "${EQR_WORKERS:-4}" \
  --clip-length "$clip_length" \
  --crop-stride "${EQR_CROP_STRIDE:-1}" \
  --supervision-tail-frames "$supervision_tail_frames" \
  --online-rollout-blocks "${EQR_ONLINE_ROLLOUT_BLOCKS:-1}" \
  "${anchor_length_args[@]}" \
  --num-targets "$num_targets" \
  --target-selection-mode deterministic_epoch_shuffle \
  "${crop_view_args[@]}" \
  --lr "$learning_rate" \
  --weight-decay 1e-4 \
  --query-dim 64 \
  --mamba-d-model 128 \
  --mamba-d-state 16 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --mamba-layers 2 \
  --max-residual-logit 1.5 \
  --static-style-adapter \
  --style-adapter-hidden 128 \
  --max-style-logit 0.75 \
  --compatibility-mode legacy_relative_softmax \
  --gate-temperature 1.0 \
  --gate-floor 1.0 \
  --gate-warmup-steps 0 \
  --mixture-mode predicted_support \
  --sequence-set-training \
  --anchor-ccc-weight 10.0 \
  --learned-set-ccc-weight "$learned_set_ccc_weight" \
  --mse-weight 0.2 \
  --velocity-weight 0.05 \
  --session-set-weight "$session_set_weight" \
  --sinkhorn-temperature 0.1 \
  --sinkhorn-iterations 8 \
  --sequence-transport-weight "$sequence_transport_weight" \
  --sequence-transport-temperature 0.03 \
  --sequence-distance-mode "$sequence_distance_mode" \
  --set-matching-strategy "$set_matching_strategy" \
  --query-identity-weight "$query_identity_weight" \
  --semantic-identity-mode uniform \
  --compatibility-weight 0.1 \
  --mixture-weight 0.25 \
  --diversity-weight "$diversity_weight" \
  --diversity-margin "$diversity_margin" \
  --official-diversity-weight 0.0 \
  --official-diversity-batch-weight "$official_diversity_batch_weight" \
  --official-diversity-margin 0.15 \
  --channel-diversity-weight "$channel_diversity_weight" \
  --channel-diversity-budget "$channel_diversity_budget" \
  "${diversity_cap_args[@]}" \
  --train-au-residual-multiplier 1.8 \
  --train-va-residual-multiplier 0.4 \
  --train-expression-residual-multiplier 1.6 \
  --frd-surrogate-mode diagonal \
  --frd-surrogate-weight "${EQR_FRD_SURROGATE_WEIGHT:-2.0}" \
  --frd-surrogate-target "$frd_surrogate_target" \
  --frd-surrogate-stride "${EQR_FRD_SURROGATE_STRIDE:-12}" \
  --frd-surrogate-calibration 0.8408521090473291 \
  --frd-transport-weight "$frd_transport_weight" \
  --frd-transport-temperature 0.1 \
  --frd-transport-iterations 32 \
  --residual-weight 0.005 \
  --prototype-shrinkage 20 \
  --class-balance-power 0.5 \
  --max-sample-weight 16 \
  --train-residual-scale 0.95 \
  --train-style-residual-scale 0.95 \
  "${isolated_va_args[@]}" \
  --frd-softdtw-stride 6 \
  --frd-softdtw-band-ratio 0.15 \
  --frd-softdtw-gamma 0.1 \
  --frd-softdtw-pair-chunk-size 256 \
  --seed "${EQR_SEED:-1}" \
  --save-every "${EQR_SAVE_EVERY:-1}" \
  --print-every "${EQR_PRINT_EVERY:-25}" \
  --precision bf16 \
  --listener-3dmm-weight "${EQR_LISTENER_3DMM_WEIGHT:-0}" \
  --listener-3dmm-velocity-weight "${EQR_LISTENER_3DMM_VELOCITY_WEIGHT:-0.1}" \
  --max-train-batches "${EQR_MAX_TRAIN_BATCHES:-0}" \
  --max-train-steps "${EQR_MAX_TRAIN_STEPS:-0}"
