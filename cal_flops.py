from fvcore.nn import FlopCountAnalysis
from args import get_parser
import data.transforms as T
from config import config2model_path
import torch
import os
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter
from PIL import Image
from torch.nn import functional as F
import cv2
from model.SAM2_RIS import SAM2_RIS
# from model.SAM2_RIS_WO_BLIP2 import SAM2_RIS_WO_BLIP2 as SAM2_RIS
from sam2.build_sam import build_sam2

parser = get_parser()
args = parser.parse_args()
device = torch.device(0)
model_cfg = "sam2.1_hiera_s.yaml"
sam2_checkpoint = config2model_path[model_cfg]
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True).to(device)

sam_ris_model = SAM2_RIS(model=sam2_model, args=args, device=device).to(device)
imgs = torch.rand(1, 3, 480, 480).to(device)
cnn_imgs = torch.rand(1, 3, 480, 480).to(device)
imgs_wo_trans = torch.rand(1, 3, 480, 480).to(device)
blip2_imgs = torch.rand(1, 3, 224, 224).to(device)
target = torch.rand(1, 3, 480, 480).to(device)
obj_cap = ["a cat"]
step = 0
multi_mask=False
flops = FlopCountAnalysis(sam_ris_model, (imgs, cnn_imgs, imgs_wo_trans, blip2_imgs, target, obj_cap, step, 0, multi_mask))  # 传入 tuple
print(f"GFLOPs: {flops.total() / 1e9:.2f}")