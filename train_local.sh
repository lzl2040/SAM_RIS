# bs=1, 6GB
torchrun --nproc_per_node=1 \
    train.py \
    --model_cfg sam2.1_hiera_l.yaml \
    --num_epochs 40 \
    --batch_size 1 \
    --image_size 480 \
    --train_visual \
    --dataset all \
    --refer_data_root "${REFER_DATA_ROOT:-/data/Data/coco}" \
    --print_interval 40 \
    --save_visual_interval 1000 \
    --lr 1e-4 \
    --work_dir "${WORK_DIR:-/data/weights/MM_SAM/work_dir}" \
    --log_path "${LOG_PATH:-/data/weights/MM_SAM/log_path}" \
    --blip2_path "${BLIP2_PATH:-${PRETRAIN_WEIGHT_ROOT:-/data/weights/Pretrain}/instruct_blip_vicuna7b_trimmed.pth}" \
    --task_name 0526_MM_SAM_img_size_480_large_full \
    # --distributed