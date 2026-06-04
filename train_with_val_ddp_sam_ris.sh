export TORCH_HOME=/mnt/lizuolei/gxy/checkpoints/Pretrain
# instruct_blip_vicuna7b_trimmed.pth
# blip2_pretrained_vitL.pth
unset NCCL_DEBUG
torchrun --nproc_per_node=8 \
  train_with_val_sam_ris.py \
  --sam_model_type vit_h \
  --sam_pretrain_path /mnt/lizuolei/gxy/checkpoints/Pretrain/sam_vit_h_4b8939.pth \
  --refer_data_root /mnt/lizuolei/gxy/Data/coco \
  --num_epochs 40 \
  --batch_size 18 \
  --image_size 480 \
  --dataset all \
  --print_interval 40 \
  --task_name 0603_MM_SAM_img_size_480_large_full_8gpu_bs_144_lr_2e-4_train_full_val_refcoco_val_ab_sam_huge \
  --lr 2e-4 \
  --grad_clip_norm 1.0 \
  --logit_clip_value 32.0 \
  --work_dir /mnt/lizuolei/gxy/checkpoints/MM_SAM/work_dir \
  --log_path /mnt/lizuolei/gxy/checkpoints/MM_SAM/log_path \
  --blip2_path /mnt/lizuolei/gxy/checkpoints/Pretrain/instruct_blip_vicuna7b_trimmed.pth \
  --blip2_vit_model eva_clip_g \
  --val_interval 1 \
  --val_print_interval 100 \
  --distributed \
  --auto_resume
    
