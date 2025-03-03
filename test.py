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
from sam2.build_sam import build_sam2
from utils.logger_utils import get_logger
from utils.metrics import computeIoU, IoU
from utils.visual import draw_contours

# from referSAM
# def get_transform(args):
#     image_transforms = [T.ResizeLongestSide(args.image_size),
#                         T.Normalize(),
#                         T.ToTensor(),
#                         T.Pad(args.image_size)
#                         ]
#     target_transforms = [T.ResizeLongestSide(args.image_size // 4),
#                          T.ToTensor(),
#                          T.Pad(args.image_size // 4)
#                          ]
#
#     return T.Compose(image_transforms), T.Compose(target_transforms)

def get_transform(args):
    transforms = [T.Resize(args.image_size, args.image_size),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # only normalize image
                 ]

    cnn_transform = [T.Resize(args.cnn_img_size, args.cnn_img_size),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]

    return T.Compose(transforms), T.Compose(cnn_transform)

def get_dataset(image_set, image_transforms, target_transforms, cnn_transforms, args, eval_mode=False):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=image_transforms,
                      target_transforms=target_transforms,
                      cnn_transform=cnn_transforms,
                      eval_mode=eval_mode,
                      )
    return ds

def test_model(model, data_loader, device, logger, save_path):
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    iterations = 0
    pred_iou = 0
    miou_avg = AverageMeter()
    with torch.no_grad():
        for step, (imgs, cnn_imgs, target, captions, blip2_imgs, imgs_wo_trans, info) in enumerate(tqdm(data_loader)):
            imgs, target, blip2_imgs = imgs.to(device), target.to(device), blip2_imgs.to(device)
            captions = captions[0].split("#####")
            org_W, org_H = info["org_size"][0]
            org_mask = info["raw_masks"].unsqueeze(0)
            org_img = info["raw_imgs"]
            for obj_cap in captions:
                pred_logits, low_pred_logits, query_embeds, object_score_logits, iou_predictions = model(imgs, cnn_imgs, imgs_wo_trans, blip2_imgs,
                                                                                         target, obj_cap, step, 0, multi_mask=False)
                # ---------------- from referSAM ----------------------#
                # low_res_masks = torch.where(low_pred_logits > 0.0, 1.0, 0.0)
                # # print(low_res_masks.shape, iou_predictions.shape) # if multi_mask=True, torch.Size([1, 3, 256, 256]) torch.Size([1, 3])
                #
                # iou_gt, _, _ = computeIoU(low_res_masks, target)  # B x C
                # max_iou_pred, idx = iou_predictions.max(1)
                # _, idx_gt = iou_gt.max(1)
                # idx_choice = idx
                # all_masks = low_res_masks.cpu().data.numpy()
                # low_res_masks = low_res_masks[range(len(idx_choice)), idx_choice, :, :]
                # low_res_masks = low_res_masks.cpu().data.numpy()
                # max_iou_pred = max_iou_pred.item()
                # ---------------- from referSAM ----------------------#

                pred_logits = F.interpolate(
                    low_pred_logits,
                    size=(org_H, org_W),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_masks = torch.sigmoid(pred_logits)
                pred_masks = pred_masks.detach().cpu().numpy()
                pred_masks = np.array(pred_masks > 0.5).astype(np.uint8)
                target_ = org_mask.cpu().data.numpy()
                # print(pred_masks.shape, target_.shape) # B C=1 H E
                I, U = IoU(pred_masks, target_)
                this_iou = 0.0 if U == 0 else I * 1.0 / U
                miou_avg.update(this_iou)
                # mean_IoU.append(this_iou)

                # ---------------- visualize ----------------------#

                raw_img = org_img.detach().cpu().numpy()[0]
                masks = np.array(target_[0] > 0).astype(np.uint8)
                # 高置信区域
                sam2_pred = pred_masks[0]
                image_bgr = raw_img[:, :, ::-1].astype(np.uint8)

                image_with_contours = draw_contours(image_bgr, sam2_pred[0], masks[0])

                # 将绘制了轮廓的图像转换回RGB格式
                image_with_contours_rgb = image_with_contours[:, :, ::-1]
                # print(image_with_contours_rgb.shape)

                # 创建 PIL 图像对象
                img_with_contours = Image.fromarray(image_with_contours_rgb)

                # 保存图像
                img_save_path = os.path.join(save_path, f"{seg_total:05d}")
                os.makedirs(img_save_path, exist_ok=True)
                img_with_contours.save(os.path.join(img_save_path, f"visual.jpg"))
                # cv2.imwrite(os.path.join(img_save_path, f"uncertainty.jpg"), entropy_heat)
                with open(os.path.join(img_save_path, "caption.txt"), "w") as f:
                    f.write(obj_cap + "\n")
                # ---------------- visualize ----------------------#

                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1
                logger.info(f"Num: {seg_total} mIoU: {miou_avg.avg * 100:.4f}, oIoU: {(cum_I / cum_U) * 100:.4f}")

    # mean_IoU = np.array(mean_IoU)
    mIoU = miou_avg.avg
    oIoU = cum_I / cum_U
    res_str = ""
    res_str += 'Final results:\n'
    res_str += 'Mean IoU is %.2f\n' % (mIoU * 100.)
    res_str += 'Overall IoU is %.2f\n' % (oIoU * 100.)

    for n_eval_iou in range(len(eval_seg_iou_list)):
        precision = seg_correct[n_eval_iou] * 100. / seg_total
        res_str += f"Precision@{str(eval_seg_iou_list[n_eval_iou])}={precision:.2f}%\n"
        # logger.info(f"Precision@{str(eval_seg_iou_list[n_eval_iou])}={precision:.2f}%\n")
    logger.info(res_str)

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.log_path = os.path.join(args.log_path, f"{args.task_name}_test_e_38.txt")
    args.pretrain_model_path = config2model_path[args.model_cfg]

    logger = get_logger(args.log_path)

    device = torch.device(int(args.device))
    image_transforms, cnn_transforms = get_transform(args)

    dataset_test = get_dataset(args.split, image_transforms, None, cnn_transforms, args,
                               eval_mode=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)

    # create model
    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    sam_ris_model = SAM2_RIS(model=sam2_model, args=args, device=device).to(device)

    # load weights
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            sam_ris_model.load_state_dict(checkpoint["model"], strict=False)
    sam_ris_model.eval()
    os.makedirs(args.pred_save_dir, exist_ok=True)
    test_model(sam_ris_model, data_loader_test, device, logger, save_path = args.pred_save_dir)




if __name__ == "__main__":
    main()