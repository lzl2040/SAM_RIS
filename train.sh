torchrun --nproc_per_node=1 \
    train.py \
    --model_cfg sam2.1_hiera_l.yaml \
    --num_epochs 40 \
    --batch_size 36 \
    --image_size 480 \
    --train_visual \
    --dataset all \
    --refer_data_root /mnt/lizuolei/Other/Data/coco \
    --print_interval 20 \
    --save_visual_interval 100 \
    --query_num 24 \
    --lr 2e-4 \
    --work_dir /mnt/lizuolei/Other/MD_SAM/work_dir \
    --log_path /mnt/lizuolei/Other/MD_SAM/log_path \
    --blip2_path /mnt/lizuolei/Other/MD_SAM/Pretrain/instruct_blip_vicuna7b_trimmed.pth \
    --task_name 0728_MD_SAM_img_size_480_large_wo_adapter_and_tmfm \
    # --distributed