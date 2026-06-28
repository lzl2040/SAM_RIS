python train_with_val_sam_ris.py \
  --sam_model_type vit_b \
  --sam_pretrain_path /data/weights/Pretrain/sam_vit_b_01ec64.pth \
  --blip2_path /data/weights/Pretrain/instruct_blip_vicuna7b_trimmed.pth \
  --refer_data_root /data/Data/coco \
  --num_epochs 40 \
  --batch_size 2 \
  --image_size 480 \
  --dataset all \
  --print_interval 40 \
  --task_name SAM_RIS_vit_b \
  --lr 1e-4 \
  --work_dir /data/weights/MM_SAM/work_dir \
  --log_path /data/weights/MM_SAM/log_path \
  --val_interval 1 \
  --val_print_interval 100 \
  --auto_resume
    