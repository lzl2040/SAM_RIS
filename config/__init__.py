# PRETRAIN_WEIGHT_ROOT = "/home/v-zuoleili/Pretrain"
PRETRAIN_WEIGHT_ROOT = "/mnt/lizuolei/Other/MD_SAM/Pretrain"
# PRETRAIN_WEIGHT_ROOT = "/data/lizuolei/Project/SAM_RVOS/checkpoints"
config2model_path = {
    "sam2_hiera_b+.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2_hiera_base_plus.pt",
    "sam2_hiera_s.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2_hiera_small.pt",
    "sam2.1_hiera_s.yaml" : f"{PRETRAIN_WEIGHT_ROOT}/sam2.1_hiera_small.pt",
    "sam2.1_hiera_b+.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_l.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2.1_hiera_large.pt",
}