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
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
from model.SAM2_RIS import SAM2_RIS
from sam2.build_sam import build_sam2
import data.transforms as T
PRETRAIN_WEIGHT_ROOT = "/home/v-zuoleili/Pretrain"
config2model_path = {
    "sam2_hiera_b+.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2_hiera_base_plus.pt",
    "sam2_hiera_s.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2_hiera_small.pt",
    "sam2.1_hiera_s.yaml" : f"{PRETRAIN_WEIGHT_ROOT}/sam2.1_hiera_small.pt",
    "sam2.1_hiera_b+.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_l.yaml" : "/data/lizuolei/Project/SAM_RVOS/checkpoints/sam2.1_hiera_large.pt",
}



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
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]

    return T.Compose(transforms)

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    return ds

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.pretrain_model_path = config2model_path[args.model_cfg]

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

    set_seed(14159265)

    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    sam_ris_model = SAM2_RIS(model=sam2_model, args=args, device=device).to(device)

    if args.distributed:
        sam_ris_model = torch.nn.parallel.DistributedDataParallel(sam_ris_model, device_ids=[local_rank],
                                                                   output_device=local_rank, broadcast_buffers=False)
        sam_ris_model = sam_ris_model.module
    sam_ris_model.train()

    if local_rank == 0:
        print("Number of total parameters: ", sum(p.numel() for p in sam_ris_model.parameters()),)
        print("Number of trainable parameters: ", sum(p.numel() for p in sam_ris_model.parameters() if p.requires_grad))

        img_mask_encdec_params = list(sam_ris_model.sam2_model.sam_mask_decoder.parameters())
        print("Number of mask decoder parameters: ", sum(p.numel() for p in img_mask_encdec_params if p.requires_grad))

    trainable_params = [p for p in sam_ris_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    scaler = GradScaler()

    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10

    train_dataset = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)

    if local_rank == 0:
        print("Number of training samples: ", len(train_dataset))
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
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            start_epoch = checkpoint["epoch"] + 1
            sam_ris_model.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["sche"])
            lr_scheduler.load_state_dict(checkpoint["lr_sche"])
        model_save_path = Path(args.resume).parent
        # sam_rvos_model.set_model_save_path(model_save_path)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
        os.makedirs(model_save_path, exist_ok=True)
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain, map_location="cpu")
        sam_ris_model.load_state_dict(checkpoint["model"], strict=False)
        print("=> loading pretrain checkpoint '{}'".format(args.pretrain))

    sam_ris_model.set_model_save_path(model_save_path)
    # step = 0
    loss_avg = AverageMeter()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for step, (imgs, target, captions, blip2_imgs, raw_imgs) in enumerate(tqdm(train_loader)):
            imgs, masks, blip2_imgs = imgs.to(device), target.to(device), blip2_imgs.to(device)
            # print(imgs.shape, masks.shape, blip2_imgs.shape, raw_imgs.shape)
            # imgs = rearrange(imgs, "b t c h w -> (b t) c h w")
            masks = rearrange(masks, "b h w -> b 1 h w")
            # blip2_imgs = rearrange(blip2_imgs, "b c h w -> b h w c")
            with torch.cuda.amp.autocast():
                sam_pred, query_embeds, neg_query_embeds, gt_mask_queries, object_score_logits = sam_ris_model(imgs, raw_imgs, blip2_imgs, masks, captions, step, epoch)
                loss = seg_loss(sam_pred, masks) + ce_loss(sam_pred, masks.float())

                loss_avg.update(loss.item(), imgs.shape[0])
                if step % args.print_interval == 0 and local_rank == 0:
                    print(f"Epoch: {epoch}/{num_epochs} Step:{step}/{len(train_loader)} Lr:{optimizer.param_groups[0]['lr']} "
                          f"Loss:{loss_avg.val}/{loss_avg.avg}")
                loss = loss / args.grad_accumulation_steps

            if (step + 1) % args.grad_accumulation_steps == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        # scheduler.step()
        epoch_loss /= step
        losses.append(epoch_loss)

        if local_rank == 0:
            print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
            ## save the latest model
            model_dict = sam_ris_model.state_dict()
            save_dict = {k: v for k, v in model_dict.items() if 'mask_decoder' in k or 'Qformer' in k
                         or 'resizer' in k or 'fusion' in k or 'score' in k or "obj_ptr_proj" in k
                         or "lora" in k}
            print(save_dict.keys())
            checkpoint = {
                "model": save_dict,
                "optimizer": optimizer.state_dict(),
                "sche" : scheduler.state_dict(),
                "lr_sche" : lr_scheduler.state_dict(),
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, os.path.join(model_save_path, f"ris_sam_model_e_{epoch + 1}.pth"))
            ## save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint = {
                    "model": save_dict,
                    "optimizer": optimizer.state_dict(),
                    "sche": scheduler.state_dict(),
                    "lr_sche": lr_scheduler.state_dict(),
                    "epoch": epoch + 1,
                }
                torch.save(checkpoint, os.path.join(model_save_path, "ris_sam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

if __name__ == '__main__':
    main()