torchrun --nproc_per_node=1 \
    train.py \
    --model_cfg sam2.1_hiera_b+.yaml \
    --num_epochs 40 \
    --batch_size 8 \
    --image_size 1024 \
    --train_visual \
    --dataset all \
    --refer_data_root "${REFER_DATA_ROOT:-/mnt/lizuolei/Other/Data/coco}" \
    --print_interval 40 \
    --save_visual_interval 1000 \
    --lr 1e-4 \
    --work_dir "${WORK_DIR:-/mnt/lizuolei/Other/MD_SAM/work_dir}" \
    --log_path "${LOG_PATH:-/mnt/lizuolei/Other/MD_SAM/log_path}" \
    --blip2_path "${BLIP2_PATH:-${PRETRAIN_WEIGHT_ROOT:-/mnt/lizuolei/Other/MD_SAM/Pretrain}/instruct_blip_vicuna7b_trimmed.pth}" \
    --task_name 0621_MD_SAM_img_size_1024_base_wo_google \
    # --distributed