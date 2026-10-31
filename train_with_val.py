import monai
from args import get_parser
import numpy as np
import random
import torch
import torch.distributed as distributed
import torch.nn as nn
from torch.cuda.amp import GradScaler
from einops import rearrange
from timm.utils import AverageMeter
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import logging
from contextlib import nullcontext
from torch.nn import functional as F
from model.SAM2_RIS import SAM2_RIS
# from model.SAM2_RIS_WO_BLIP2 import SAM2_RIS_WO_BLIP2 as SAM2_RIS
from sam2.build_sam import build_sam2
import data.transforms as T
from config import config2model_path
from utils.logger_utils import get_logger
from utils.metrics import IoU


# 计算为 GB（float32 -> 每个参数4字节）
def param_count_to_gb(param_count):
    return param_count * 4 / (1024 ** 3)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

def get_dataset(image_set, transform, cnn_transform, args, eval_mode=False):
    from data.dataset_refer_bert import ReferDataset
    if args.dataset == "all" and not eval_mode:
        # use refcocog umd, we suppose refcocog google contains the val images from refcocog umd
        # data_list = ["refcoco", "refcoco+", "refcocog", "refcocog"]
        # split_by = ["unc", "unc", "umd", "google"]
        data_list = ["refcoco", "refcoco+", "refcocog"]
        split_by = ["unc", "unc", "umd"]
        dataset_list = []
        original_dataset = args.dataset
        original_split_by = args.splitBy
        for i in range(len(data_list)):
            args.dataset = data_list[i]
            args.splitBy = split_by[i]
            dataset = ReferDataset(args,
                        split=image_set,
                        image_transforms=transform,
                        target_transforms=None,
                        cnn_transform=cnn_transform,
                        eval_mode=eval_mode,
                      )
            dataset_list.append(dataset)
        args.dataset = original_dataset
        args.splitBy = original_split_by
        ds = ConcatDataset(dataset_list)
    # elif args.dataset == "refcocog":
    #     # only use one
    #     data_list = ["refcocog", "refcocog"]
    #     split_by = ["umd", "google"]
    #     dataset_list = []
    #     for i in range(len(data_list)):
    #         args.dataset = data_list[i]
    #         args.splitBy = split_by[i]
    #         dataset = ReferDataset(args,
    #                     split=image_set,
    #                     image_transforms=transform,
    #                     target_transforms=None,
    #                     cnn_transform=cnn_transform
    #                   )
    #         dataset_list.append(dataset)
    #     ds = ConcatDataset(dataset_list)
    else:
        ds = ReferDataset(args,
                        split=image_set,
                        image_transforms=transform,
                        target_transforms=None,
                        cnn_transform=cnn_transform,
                        eval_mode=eval_mode,
                        )
    if image_set == "train" and args.exclude_eval_images:
        datasets = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
        held_out = {
            ref["image_id"] for dataset in datasets for ref in dataset.refer.Refs.values()
            if ref["split"] != "train"
        }
        for dataset in datasets:
            kept = [index for index, ref_id in enumerate(dataset.ref_ids)
                    if dataset.refer.Refs[ref_id]["image_id"] not in held_out]
            dataset.ref_ids = [dataset.ref_ids[index] for index in kept]
            dataset.captions = [dataset.captions[index] for index in kept]
        # ConcatDataset caches cumulative lengths at construction.
        ds = ConcatDataset(datasets) if isinstance(ds, ConcatDataset) else datasets[0]
    return ds


def build_refcoco_val_loader(args, image_transforms, cnn_transform):
    original_dataset = args.dataset
    original_split_by = args.splitBy
    args.dataset = args.val_dataset or (original_dataset if original_dataset != "all" else "refcoco")
    args.splitBy = args.val_split_by or (original_split_by if args.dataset == "refcocog" else "unc")
    try:
        val_dataset = get_dataset("val",
                                  transform=image_transforms,
                                  cnn_transform=cnn_transform,
                                  args=args,
                                  eval_mode=True)
    finally:
        args.dataset = original_dataset
        args.splitBy = original_split_by
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    return DataLoader(val_dataset,
                      batch_size=1,
                      sampler=val_sampler,
                      num_workers=args.workers,
                      pin_memory=True)


def validate_refcoco(model, data_loader, device, logger, epoch, print_interval):
    cum_I, cum_U = 0, 0
    mean_iou = AverageMeter()
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0

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
                _, low_pred_logits, _, _, _ = model(imgs, cnn_imgs, imgs_wo_trans, blip2_imgs,
                                                    target, obj_cap, step, epoch, multi_mask=False)
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
                I, U = IoU(pred_masks, target_)
                this_iou = 0.0 if U == 0 else I * 1.0 / U
                mean_iou.update(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            if local_print_step(step, print_interval):
                o_iou = 0.0 if cum_U == 0 else cum_I / cum_U
                precision_log = ", ".join(
                    f"Pr@{eval_seg_iou_list[i]:.1f}:{seg_correct[i] * 100.0 / max(seg_total, 1):.2f}"
                    for i in range(len(eval_seg_iou_list))
                )
                logger.info(f"Val Epoch: {epoch + 1} Step:{step}/{len(data_loader)} "
                            f"mIoU:{mean_iou.avg * 100:.4f}, oIoU:{o_iou * 100:.4f}, "
                            f"{precision_log}")

    m_iou = mean_iou.avg
    o_iou = 0.0 if cum_U == 0 else cum_I / cum_U
    precision_metrics = {
        f"Precision@{eval_seg_iou_list[i]:.1f}": seg_correct[i] * 100.0 / max(seg_total, 1)
        for i in range(len(eval_seg_iou_list))
    }
    precision_log = ", ".join(f"{key}: {value:.2f}%" for key, value in precision_metrics.items())
    logger.info(f"Val Final Epoch: {epoch + 1}, Samples: {seg_total}, "
                f"mIoU: {m_iou * 100:.4f}, oIoU: {o_iou * 100:.4f}, {precision_log}")
    model.train()
    return {"mIoU": m_iou, "oIoU": o_iou, **precision_metrics}


def local_print_step(step, print_interval):
    return print_interval > 0 and step % print_interval == 0


def format_validation_metrics(metrics):
    metric_text = []
    for metric_name, metric_value in metrics.items():
        if metric_name in ("mIoU", "oIoU"):
            metric_text.append(f"{metric_name}: {metric_value * 100:.4f}%")
        else:
            metric_text.append(f"{metric_name}: {metric_value:.2f}%")
    return ", ".join(metric_text)


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_resume_checkpoint(resume_path, model, optimizer, lr_scheduler, scaler, device, logger):
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    logger.info(f"=> loading resume checkpoint '{resume_path}'")
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_config") != model.experiment_config:
        raise ValueError("Resume configuration differs or is missing; start a new controlled run.")
    model.load_state_dict(checkpoint["model"], strict=False)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_to_device(optimizer, device)
    else:
        logger.info("=> resume checkpoint has no optimizer state; optimizer starts fresh")

    if "lr_sche" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_sche"])
    else:
        logger.info("=> resume checkpoint has no lr scheduler state; scheduler starts fresh")

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    else:
        logger.info("=> resume checkpoint has no GradScaler state; scaler starts fresh")

    start_epoch = checkpoint.get("epoch", 0)
    best_oIoU = checkpoint.get("best_oIoU", checkpoint.get("val_oIoU", -1.0))
    best_loss = checkpoint.get("best_loss", 1e10)
    logger.info(f"=> resumed from epoch {start_epoch}, best_oIoU={best_oIoU * 100:.4f}")
    return start_epoch, best_oIoU, best_loss


def find_latest_checkpoint(work_dir, task_name):
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return None

    run_dirs = [path for path in work_dir.glob(f"{task_name}-*") if path.is_dir()]
    if not run_dirs:
        return None

    candidates = []
    for run_dir in run_dirs:
        latest_path = run_dir / "ris_sam_model_latest.pth"
        if latest_path.is_file():
            candidates.append(latest_path)
        candidates.extend(path for path in run_dir.glob("ris_sam_model_e_*.pth") if path.is_file())

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, best_oIoU, best_loss,
                    val_metrics=None):
    model_dict = model.state_dict()
    save_dict = {k: v for k, v in model_dict.items() if 'mask_decoder' in k or 'Qformer' in k
                 or 'resizer' in k or 'fusion' in k or 'score' in k or "obj_ptr_proj" in k
                 or "lora" in k or "cnn" in k}
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    save_dict.update({key: value for key, value in model_dict.items() if key in trainable})
    checkpoint = {
        "model": save_dict,
        "optimizer": optimizer.state_dict(),
        "lr_sche": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch + 1,
        "best_oIoU": best_oIoU,
        "best_loss": best_loss,
        "experiment_config": model.experiment_config,
    }
    if val_metrics is not None:
        for metric_name, metric_value in val_metrics.items():
            checkpoint[f"val_{metric_name}"] = metric_value
    return checkpoint, save_dict


def main():
    parser = get_parser()
    args = parser.parse_args()
    sam2_checkpoint = config2model_path[args.model_cfg]
    os.makedirs(args.log_path, exist_ok=True)
    args.log_path = os.path.join(args.log_path, f"{args.task_name}.txt")

    # ---------------- start writing logs ----------------------#
    logger = get_logger(args.log_path)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    # ---------------- end writing logs ----------------------#

    # 多卡
    if args.distributed:
        distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = torch.device(local_rank)
        torch.cuda.set_device(local_rank)
        print('I am rank %d in this world of size %d!' % (local_rank, world_size))
    else:
        device = torch.device(f"cuda:{args.device}")
        local_rank = 0

    def worker_init_fn(worker_id):
        return np.random.seed(torch.initial_seed() % (2 ** 31) + worker_id + local_rank * 100)

    set_seed(args.seed)

    model_cfg = args.model_cfg
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    sam_ris_model = SAM2_RIS(model=sam2_model, args=args, device=device).to(device)

    if args.distributed:
        sam_ris_model = torch.nn.parallel.DistributedDataParallel(sam_ris_model, device_ids=[local_rank],
                                                                   output_device=local_rank, broadcast_buffers=False,
                                                                   find_unused_parameters=True)
    model_without_ddp = sam_ris_model.module if args.distributed else sam_ris_model
    sam_ris_model.train()

    if local_rank == 0:
        total_params = sum(p.numel() for p in model_without_ddp.parameters())
        trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        # blip2_qformer_params = sum(p.numel() for p in sam_ris_model.blip2_qformer.parameters())
        # qformer_params = sum(p.numel() for p in sam_ris_model.blip2_qformer.Qformer.parameters())
        image_encoder_params = sum(p.numel() for p in model_without_ddp.sam2_model.image_encoder.parameters())
        mask_decoder_params = sum(p.numel() for p in model_without_ddp.sam2_model.sam_mask_decoder.parameters())
        fusion_params_list = [
            (name, param) for name, param in model_without_ddp.sam2_model.image_encoder.named_parameters()
            if "fusion" in name
        ]
        fusion_params = sum(p.numel() for _, p in fusion_params_list)
        logger.info(f"Number of total parameters: {param_count_to_gb(total_params)} GB")
        logger.info(f"Number of trainable parameters: {param_count_to_gb(trainable_params)} GB")
        # logger.info(f"Number of blip2qformer parameters: {param_count_to_gb(blip2_qformer_params)} GB")
        # logger.info(f"Number of qformer parameters: {param_count_to_gb(qformer_params)} GB")
        logger.info(f"Number of sam image encoder parameters: {param_count_to_gb(image_encoder_params)} GB")
        logger.info(f"Number of sam mask encoder parameters: {param_count_to_gb(mask_decoder_params)} GB")
        logger.info(f"Number of fusion parameters: {param_count_to_gb(fusion_params)} GB")


    trainable_params = [p for p in sam_ris_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    scaler = GradScaler()

    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    best_oIoU = -1.0
    best_val_epoch = None
    best_val_metrics = None
    best_loss_epoch = None

    image_transforms, cnn_transform = get_transform(args)
    train_dataset = get_dataset("train",
                                       transform=image_transforms,
                                       cnn_transform=cnn_transform,
                                       args=args)
    val_loader = None
    if local_rank == 0 and args.val_interval > 0:
        val_loader = build_refcoco_val_loader(args, image_transforms, cnn_transform)

    if local_rank == 0:
        print("Number of training samples: ", len(train_dataset))
        if val_loader is not None:
            print("Number of validation samples: ", len(val_loader.dataset))
    if args.distributed:
        # sampler_train = samplers.DistributedSampler(ytvos_train_dataset)
        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank, shuffle=True)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(train_loader) * num_epochs)) ** 0.9)

    start_epoch = 0
    if args.resume is not None and (not args.resume.strip() or not os.path.isfile(args.resume)):
        logger.warning(
            f"=> invalid resume checkpoint {args.resume!r}; ignoring it and proceeding without explicit resume"
        )
        args.resume = None

    if args.resume is None and args.auto_resume:
        latest_checkpoint = find_latest_checkpoint(args.work_dir, args.task_name)
        if latest_checkpoint is not None:
            args.resume = str(latest_checkpoint)
            logger.info(f"=> auto resume found checkpoint '{args.resume}'")
        else:
            logger.info("=> auto resume found no checkpoint; starting a new run")

    if args.resume is not None:
        start_epoch, best_oIoU, best_loss = load_resume_checkpoint(
            args.resume, model_without_ddp, optimizer, lr_scheduler, scaler, device, logger
        )
        model_save_path = Path(args.resume).parent
        # sam_rvos_model.set_model_save_path(model_save_path)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
        os.makedirs(model_save_path, exist_ok=True)
    if args.pretrain is not None and args.resume is None:
        checkpoint = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        print("=> loading pretrain checkpoint '{}'".format(args.pretrain))

    model_without_ddp.set_model_save_path(model_save_path)
    # step = 0
    loss_avg = AverageMeter()
    optimizer.zero_grad()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for step, (imgs, cnn_imgs, target, captions, blip2_imgs, imgs_wo_trans, info) in enumerate(tqdm(train_loader)):
            imgs, masks, blip2_imgs = imgs.to(device), target.to(device), blip2_imgs.to(device)
            cnn_imgs = cnn_imgs.to(device)
            # print(imgs.shape, masks.shape, blip2_imgs.shape, raw_imgs.shape)
            # imgs = rearrange(imgs, "b t c h w -> (b t) c h w")
            masks = rearrange(masks, "b h w -> b 1 h w")
            # blip2_imgs = rearrange(blip2_imgs, "b c h w -> b h w c")
            should_step = (step + 1) % args.grad_accumulation_steps == 0 or (step + 1) == len(train_loader)
            is_accumulating = not should_step
            sync_context = sam_ris_model.no_sync if args.distributed and is_accumulating else nullcontext
            with sync_context():
                with torch.cuda.amp.autocast():
                    pred_logits, low_pred_logits, query_embeds, object_score_logits, iou_predictions = sam_ris_model(imgs, cnn_imgs, imgs_wo_trans, blip2_imgs, masks, captions, step, epoch)
                    loss = seg_loss(pred_logits, masks.float()) + ce_loss(pred_logits, masks.float())
                    loss_avg.update(loss.item(), imgs.shape[0])
                    epoch_loss += loss.item()
                    if step % args.print_interval == 0 and local_rank == 0:
                        logger.info(f"Epoch: {epoch}/{num_epochs} Step:{step}/{len(train_loader)} Lr:{optimizer.param_groups[0]['lr']} "
                              f"Loss:{loss_avg.val}/{loss_avg.avg}")
                    loss = loss / args.grad_accumulation_steps
                scaler.scale(loss).backward()

            if should_step:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

        # scheduler.step()
        epoch_loss /= (step + 1)
        losses.append(epoch_loss)

        if local_rank == 0:
            logger.info(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
            if val_loader is not None and (epoch + 1) % args.val_interval == 0:
                val_metrics = validate_refcoco(model_without_ddp, val_loader, device, logger, epoch,
                                               args.val_print_interval)
                val_oIoU = val_metrics["oIoU"]
                logger.info(f"Validation at epoch {epoch + 1}: mIoU={val_metrics['mIoU'] * 100:.4f}, oIoU={val_oIoU * 100:.4f}")
                if val_oIoU > best_oIoU:
                    best_oIoU = val_oIoU
                    best_val_epoch = epoch + 1
                    best_val_metrics = val_metrics.copy()
                    checkpoint, _ = make_checkpoint(
                        model_without_ddp, optimizer, lr_scheduler, scaler, epoch,
                        best_oIoU, best_loss, val_metrics=val_metrics
                    )
                    torch.save(checkpoint, os.path.join(model_save_path, "ris_sam_model_best_oiou.pth"))
                    logger.info(f"Saved best oIoU checkpoint at epoch {epoch + 1}: "
                                f"oIoU={best_oIoU * 100:.4f}")
            elif val_loader is None:
                logger.info("Validation is disabled; best oIoU checkpoint will not be updated.")
            ## save a fallback best model by training loss only when validation is disabled
            if val_loader is None and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_epoch = epoch + 1
                checkpoint, _ = make_checkpoint(
                    model_without_ddp, optimizer, lr_scheduler, scaler, epoch, best_oIoU, best_loss
                )
                torch.save(checkpoint, os.path.join(model_save_path, "ris_sam_model_best.pth"))

            checkpoint, save_dict = make_checkpoint(
                model_without_ddp, optimizer, lr_scheduler, scaler, epoch, best_oIoU, best_loss
            )
            logger.info(save_dict.keys())
            torch.save(checkpoint, os.path.join(model_save_path, "ris_sam_model_latest.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

        if args.distributed:
            distributed.barrier()
    if local_rank == 0:
        if best_val_metrics is not None:
            logger.info(
                f"Training completed. Best validation epoch: {best_val_epoch}, "
                f"{format_validation_metrics(best_val_metrics)}"
            )
        else:
            logger.info("Training completed. No validation metrics were recorded.")
        if best_loss_epoch is not None:
            logger.info(f"Best training loss epoch: {best_loss_epoch}, loss: {best_loss:.4f}")
        elif best_val_metrics is None:
            logger.info(f"Best oIoU: {best_oIoU * 100:.4f}%, best_loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()
