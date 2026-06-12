from args import get_parser
import data.transforms as T
import torch
import os
import numpy as np
from tqdm import tqdm
try:
    from timm.utils import AverageMeter
except ModuleNotFoundError:
    class AverageMeter:
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count if self.count else 0
from torch.nn import functional as F

from model.SAM_RIS import SAM_RIS, build_sam_ris_backbone, resolve_sam_checkpoint
from utils.logger_utils import get_logger
from utils.metrics import IoU


def get_transform(args):
    image_transforms = [
        T.Resize(args.image_size, args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    cnn_transforms = [
        T.Resize(args.cnn_img_size, args.cnn_img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(image_transforms), T.Compose(cnn_transforms)


def get_dataset(image_set, image_transforms, cnn_transforms, args, eval_mode=True):
    from data.dataset_refer_bert import ReferDataset
    return ReferDataset(
        args,
        split=image_set,
        image_transforms=image_transforms,
        target_transforms=None,
        cnn_transform=cnn_transforms,
        eval_mode=eval_mode,
    )


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_model_checkpoint(model, resume_path, logger):
    if resume_path is None:
        logger.info("No --resume checkpoint was provided; testing with initialized weights.")
        return
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    logger.info(f"=> loading checkpoint '{resume_path}'")
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = strip_module_prefix(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")


def test_model(model, data_loader, device, logger, args):
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    miou_avg = AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (imgs, cnn_imgs, target, captions, blip2_imgs, imgs_wo_trans, info) in enumerate(tqdm(data_loader)):
            imgs = imgs.to(device)
            target = target.to(device)
            blip2_imgs = blip2_imgs.to(device)
            cnn_imgs = cnn_imgs.to(device)

            captions = captions[0].split("#####")
            org_W, org_H = info["org_size"][0]
            org_mask = info["raw_masks"].unsqueeze(0)

            for obj_cap in captions:
                _, low_pred_logits, _, _, _ = model(
                    imgs,
                    cnn_imgs,
                    imgs_wo_trans,
                    blip2_imgs,
                    target,
                    obj_cap,
                    step,
                    0,
                    multi_mask=False,
                )
                pred_logits = F.interpolate(
                    low_pred_logits,
                    size=(org_H, org_W),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_masks = torch.sigmoid(pred_logits)
                pred_masks = pred_masks.detach().cpu().numpy()
                pred_masks = np.array(pred_masks > 0.5).astype(np.uint8)
                target_np = org_mask.cpu().data.numpy()

                I, U = IoU(pred_masks, target_np)
                this_iou = 0.0 if U == 0 else I * 1.0 / U
                miou_avg.update(this_iou)
                cum_I += I
                cum_U += U

                for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)

                seg_total += 1
                if args.print_interval > 0 and seg_total % args.print_interval == 0:
                    o_iou = 0.0 if cum_U == 0 else cum_I / cum_U
                    logger.info(
                        f"Num: {seg_total} "
                        f"mIoU: {miou_avg.avg * 100:.4f}, "
                        f"oIoU: {o_iou * 100:.4f}"
                    )

    mIoU = miou_avg.avg
    oIoU = 0.0 if cum_U == 0 else cum_I / cum_U
    res_str = ""
    res_str += "Final results:\n"
    res_str += "Mean IoU is %.2f\n" % (mIoU * 100.)
    res_str += "Overall IoU is %.2f\n" % (oIoU * 100.)
    for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
        precision = seg_correct[n_eval_iou] * 100. / max(seg_total, 1)
        res_str += f"Precision@{eval_seg_iou}={precision:.2f}%\n"
    logger.info(res_str)
    return {"mIoU": mIoU, "oIoU": oIoU}


def main():
    parser = get_parser()
    args = parser.parse_args()

    sam_checkpoint = resolve_sam_checkpoint(args.sam_model_type, args.sam_pretrain_path)
    args.sam_pretrain_path = sam_checkpoint

    os.makedirs(args.log_path, exist_ok=True)
    args.log_path = os.path.join(args.log_path, f"{args.task_name}_test_sam_ris.txt")
    logger = get_logger(args.log_path)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA is unavailable; using CPU.")

    image_transforms, cnn_transforms = get_transform(args)

    original_dataset = args.dataset
    original_split_by = args.splitBy
    if args.dataset == "all":
        args.dataset = "refcoco"
        args.splitBy = "unc"
        logger.info("Testing dataset 'all' is mapped to refcoco/unc, matching train-time validation.")

    dataset_test = get_dataset(
        args.split,
        image_transforms=image_transforms,
        cnn_transforms=cnn_transforms,
        args=args,
        eval_mode=True,
    )
    args.dataset = original_dataset
    args.splitBy = original_split_by

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info(f"Number of test samples: {len(dataset_test)}")

    sam_model = build_sam_ris_backbone(
        args.sam_model_type,
        args.sam_pretrain_path,
        args.image_size,
    ).to(device)
    sam_ris_model = SAM_RIS(model=sam_model, args=args, device=device).to(device)
    load_model_checkpoint(sam_ris_model, args.resume, logger)

    test_model(sam_ris_model, data_loader_test, device, logger, args)


if __name__ == "__main__":
    main()
