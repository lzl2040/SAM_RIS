#!/usr/bin/env bash
set -euo pipefail

# Run from the project directory so relative model/tokenizer paths stay valid.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export TORCH_HOME="${TORCH_HOME:-${PRETRAIN_WEIGHT_ROOT:-/mnt/lizuolei/gxy/checkpoints/Pretrain}}"
REFER_DATA_ROOT="${REFER_DATA_ROOT:-/mnt/lizuolei/gxy/Data/coco}"
WORK_DIR="${WORK_DIR:-/mnt/lizuolei/gxy/checkpoints/MM_SAM/work_dir}"
LOG_PATH="${LOG_PATH:-/mnt/lizuolei/gxy/checkpoints/MM_SAM/log_path}"
BLIP2_PATH="${BLIP2_PATH:-${PRETRAIN_WEIGHT_ROOT:-/mnt/lizuolei/gxy/checkpoints/Pretrain}/instruct_blip_vicuna7b_trimmed.pth}"
ABLATION="${ABLATION:-full}"
extra_args=()
if (( ${NPROC_PER_NODE:-1} > 1 )); then
    extra_args+=(--distributed)
fi

# Set PRETRAIN_WEIGHT_ROOT in the environment to override the SAM2 checkpoint root.
torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" \
    train_with_val.py \
    --ablation "$ABLATION" \
    --tmfm_output "${TMFM_OUTPUT:-text}" \
    --seed "${SEED:-14159265}" \
    --exclude_eval_images \
    --model_cfg "${MODEL_CFG:-sam2.1_hiera_l.yaml}" \
    --num_epochs "${NUM_EPOCHS:-40}" \
    --batch_size "${BATCH_SIZE:-36}" \
    --image_size "${IMAGE_SIZE:-480}" \
    --train_visual \
    --dataset "${DATASET:-all}" \
    --refer_data_root "$REFER_DATA_ROOT" \
    --print_interval "${PRINT_INTERVAL:-20}" \
    --save_visual_interval "${SAVE_VISUAL_INTERVAL:-100}" \
    --lr "${LR:-2e-4}" \
    --work_dir "$WORK_DIR" \
    --log_path "$LOG_PATH" \
    --blip2_path "$BLIP2_PATH" \
    --task_name "${TASK_NAME:-controlled_${ABLATION}_${DATASET:-all}_${TMFM_OUTPUT:-text}}" \
    --val_interval "${VAL_INTERVAL:-1}" \
    --val_print_interval "${VAL_PRINT_INTERVAL:-100}" \
    --auto_resume \
    "${extra_args[@]}" \
    "$@"
