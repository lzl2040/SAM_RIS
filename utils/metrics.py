import torch
import numpy as np

def computeIoU(pred, # B x C x H x W
               gt, # B x H x W
               ):
    gt = gt.unsqueeze(1)
    gt = gt.repeat_interleave(pred.size(1),dim=1)
    I = torch.sum(torch.mul(pred, gt), (3, 2)) # B x C
    U = torch.sum(torch.add(pred, gt), (3, 2)) - I # B x C
    iou = torch.full_like(U, fill_value=0.0)
    mask = (U != 0)
    iou[mask] = I[mask] / U[mask]
    return iou, I, U # B x C

def IoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U