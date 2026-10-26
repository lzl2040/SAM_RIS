export TORCH_HOME="${TORCH_HOME:-${PRETRAIN_WEIGHT_ROOT:-/mnt/lizuolei/gxy/checkpoints/Pretrain}}"
torchrun --nproc_per_node=1 \
    train.py \
    --model_cfg sam2.1_hiera_l.yaml \
    --num_epochs 40 \
    --batch_size 36 \
    --image_size 480 \
    --train_visual \
    --dataset all \
    --refer_data_root "${REFER_DATA_ROOT:-/mnt/lizuolei/gxy/Data/coco}" \
    --print_interval 20 \
    --save_visual_interval 100 \
    --query_num 24 \
    --lr 2e-4 \
    --work_dir "${WORK_DIR:-/mnt/lizuolei/gxy/checkpoints/MM_SAM/work_dir}" \
    --log_path "${LOG_PATH:-/mnt/lizuolei/gxy/checkpoints/MM_SAM/log_path}" \
    --blip2_path "${BLIP2_PATH:-${PRETRAIN_WEIGHT_ROOT:-/mnt/lizuolei/gxy/checkpoints/Pretrain}/instruct_blip_vicuna7b_trimmed.pth}" \
    --task_name 0526_MM_SAM_img_size_480_large_full_bs_36 \
    # --distributed