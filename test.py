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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from model.SAM2_RIS import SAM2_RIS
# from model.SAM2_RIS_WO_BLIP2 import SAM2_RIS_WO_BLIP2 as SAM2_RIS
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

def overlay_mask_on_image(raw_img, mask, color=(255, 0, 0), alpha=0.5):
    """将单个mask叠加到图像上"""
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    color_mask[mask > 0] = color
    # print(raw_img.shape, color_mask.shape)
    blended = cv2.addWeighted(raw_img, 1.0, color_mask, alpha, 0)
    return blended

def _set_encoder_focus_cache(model, enabled):
    trunk = getattr(getattr(model.sam2_model, "image_encoder", None), "trunk", None)
    if trunk is None or not hasattr(trunk, "cache_fusion_features"):
        return False
    trunk.cache_fusion_features = enabled
    if not enabled and hasattr(trunk, "fusion_feature_cache"):
        trunk.fusion_feature_cache = []
    return True

def _get_encoder_focus_cache(model):
    trunk = getattr(getattr(model.sam2_model, "image_encoder", None), "trunk", None)
    if trunk is None:
        return []
    return getattr(trunk, "fusion_feature_cache", [])

def _configure_encoder_focus_attention(model, args):
    trunk = getattr(getattr(model.sam2_model, "image_encoder", None), "trunk", None)
    if trunk is None:
        return
    if hasattr(trunk, "encoder_focus_attn_reduce"):
        trunk.encoder_focus_attn_reduce = args.encoder_focus_attn_reduce
    if hasattr(trunk, "encoder_focus_query_idx"):
        trunk.encoder_focus_query_idx = args.encoder_focus_query_idx
    if hasattr(trunk, "encoder_focus_topk"):
        trunk.encoder_focus_topk = args.encoder_focus_topk

def _safe_name(text, max_len=80):
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip())
    return text[:max_len].strip("_") or "caption"

def _raw_image_to_numpy(raw_img):
    if torch.is_tensor(raw_img):
        raw_img = raw_img.detach().cpu()
        if raw_img.ndim == 4:
            raw_img = raw_img[0]
        raw_img = raw_img.numpy()
    if raw_img.ndim == 3 and raw_img.shape[0] in (1, 3):
        raw_img = np.transpose(raw_img, (1, 2, 0))
    raw_img = np.asarray(raw_img)
    if raw_img.dtype != np.uint8:
        raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)
    if raw_img.ndim == 2:
        raw_img = np.repeat(raw_img[:, :, None], 3, axis=2)
    return raw_img[:, :, :3]

def _mask_to_numpy(mask, out_hw):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != out_hw:
        mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def _attention_heatmap(attention, out_hw):
    # attention: B,T,H,W. Average frames if T exists, then upsample to raw image size.
    heat = attention[0].mean(dim=0, keepdim=True).unsqueeze(0)
    heat = F.interpolate(heat, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    return heat.numpy()

def _normalize_heatmap(heat, low, high, keep_percent):
    heat = (heat - low) / max(high - low, 1e-6)
    heat = np.clip(heat, 0, 1)
    if keep_percent < 100:
        threshold = np.percentile(heat, 100 - keep_percent)
        heat = np.where(heat >= threshold, heat, 0)
    return np.clip(heat, 0, 1)

def _softmax_heatmap(heat, temperature):
    flat = heat.reshape(-1).astype(np.float64)
    temperature = max(float(temperature), 1e-6)
    flat = flat - np.max(flat)
    prob = np.exp(flat / temperature)
    prob = prob / max(np.sum(prob), 1e-12)
    return prob.reshape(heat.shape).astype(np.float32)

def _keep_max_connected_component(heat):
    mask = (heat > 0).astype(np.uint8)
    if mask.max() == 0:
        return heat
    max_y, max_x = np.unravel_index(np.argmax(heat), heat.shape)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    if num_labels <= 1:
        return heat
    max_label = labels[max_y, max_x]
    if max_label == 0:
        return heat
    return np.where(labels == max_label, heat, 0)

def _postprocess_attention_heatmap(attn_map, keep_percent, postprocess, softmax_temp):
    if postprocess in ("softmax", "softmax_max_cc"):
        attn_map = _softmax_heatmap(attn_map, softmax_temp)
    low, high = np.percentile(attn_map, 1), np.percentile(attn_map, 99)
    attn_heat = _normalize_heatmap(attn_map, low, high, keep_percent)
    if postprocess in ("max_cc", "softmax_max_cc"):
        attn_heat = _keep_max_connected_component(attn_heat)
    return attn_heat

def _overlay_heatmap(raw_rgb, heat, alpha=0.45):
    if np.max(heat) <= 0:
        return raw_rgb.copy()
    color_rgb = plt.get_cmap("jet")(heat)[:, :, :3]
    color_rgb = (color_rgb * 255).astype(np.uint8)
    mask = heat[:, :, None] > 0
    blended = cv2.addWeighted(raw_rgb, 1.0 - alpha, color_rgb, alpha, 0)
    overlay = np.where(mask, blended, raw_rgb)
    return overlay

def _overlay_signed_heatmap(raw_rgb, signed_heat, alpha=0.45, cmap="seismic"):
    signed_heat = np.clip(signed_heat, -1, 1)
    color_rgb = plt.get_cmap(cmap)((signed_heat + 1) / 2.0)[:, :, :3]
    color_rgb = (color_rgb * 255).astype(np.uint8)
    strength = np.abs(signed_heat)[:, :, None]
    blended = cv2.addWeighted(raw_rgb, 1.0 - alpha, color_rgb, alpha, 0)
    overlay = np.where(strength > 1e-6, blended, raw_rgb)
    return overlay

def _normalize_signed_heatmap(heat):
    bound = np.percentile(np.abs(heat), 99)
    bound = max(float(bound), 1e-6)
    return np.clip(heat / bound, -1, 1)

def _tensor_quantiles(x, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    return torch.quantile(x, torch.tensor(qs, device=x.device, dtype=x.dtype)).detach().cpu().numpy()

def _summarize_gate_tensor(gate, update_norm=None, injected_norm=None):
    gate = gate.detach().float().cpu().reshape(-1)
    abs_gate = gate.abs()
    q01, q05, q25, q50, q75, q95, q99 = _tensor_quantiles(gate)
    abs_q01, abs_q05, abs_q25, abs_q50, abs_q75, abs_q95, abs_q99 = _tensor_quantiles(abs_gate)
    injected_norm_value = float(injected_norm) if injected_norm is not None else 0.0
    update_ratio = 0.0
    if injected_norm_value > 1e-12 and update_norm is not None:
        update_ratio = float(update_norm) / injected_norm_value
    return {
        "count": int(gate.numel()),
        "mean": float(gate.mean()),
        "std": float(gate.std(unbiased=False)),
        "min": float(gate.min()),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "max": float(gate.max()),
        "abs_mean": float(abs_gate.mean()),
        "abs_q01": float(abs_q01),
        "abs_q05": float(abs_q05),
        "abs_q25": float(abs_q25),
        "abs_median": float(abs_q50),
        "abs_q75": float(abs_q75),
        "abs_q95": float(abs_q95),
        "abs_q99": float(abs_q99),
        "near_zero_005": float((abs_gate < 0.05).float().mean()),
        "near_zero_010": float((abs_gate < 0.10).float().mean()),
        "positive_ratio": float((gate > 0).float().mean()),
        "negative_ratio": float((gate < 0).float().mean()),
        "update_to_injected_norm": update_ratio,
    }

def _add_gate_records(gate_records, cache, sample_idx, caption):
    for item in cache:
        gate_cache = item.get("gate")
        if gate_cache is None:
            continue
        gate = gate_cache["gate"]
        row = _summarize_gate_tensor(
            gate,
            update_norm=gate_cache.get("update_norm"),
            injected_norm=gate_cache.get("injected_norm"),
        )
        row["sample_idx"] = sample_idx
        row["stage"] = item["stage"]
        row["caption"] = str(caption)
        gate_records.append(row)

def _write_gate_csv(rows, save_path):
    if len(rows) == 0:
        return
    keys = [
        "sample_idx", "stage", "caption", "count", "mean", "std", "min", "q01", "q05", "q25",
        "median", "q75", "q95", "q99", "max", "abs_mean", "abs_q01", "abs_q05", "abs_q25",
        "abs_median", "abs_q75", "abs_q95", "abs_q99", "near_zero_005", "near_zero_010",
        "positive_ratio", "negative_ratio", "update_to_injected_norm"
    ]
    with open(save_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, str):
                    value = '"' + value.replace('"', '""') + '"'
                values.append(str(value))
            f.write(",".join(values) + "\n")

def _save_gate_summary_plots(rows, save_root):
    if len(rows) == 0:
        return
    os.makedirs(save_root, exist_ok=True)
    _write_gate_csv(rows, os.path.join(save_root, "gate_statistics_per_sample_stage.csv"))

    stages = sorted(set(row["stage"] for row in rows))
    summary_path = os.path.join(save_root, "gate_statistics_by_stage.csv")
    metric_keys = ["mean", "abs_mean", "near_zero_005", "near_zero_010", "update_to_injected_norm", "q05", "median", "q95"]
    with open(summary_path, "w") as f:
        f.write("stage," + ",".join(metric_keys) + "\n")
        for stage in stages:
            stage_rows = [row for row in rows if row["stage"] == stage]
            vals = {key: np.mean([row[key] for row in stage_rows]) for key in metric_keys}
            f.write(str(stage) + "," + ",".join(f"{vals[key]:.6f}" for key in metric_keys) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for stage in stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        axes[0].hist([row["mean"] for row in stage_rows], bins=20, alpha=0.55, label=f"S{stage}")
        axes[1].hist([row["abs_mean"] for row in stage_rows], bins=20, alpha=0.55, label=f"S{stage}")
        axes[2].hist([row["update_to_injected_norm"] for row in stage_rows], bins=20, alpha=0.55, label=f"S{stage}")
    axes[0].set_title("Gate mean")
    axes[1].set_title("|Gate| mean")
    axes[2].set_title("||g*x|| / ||x||")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, "gate_distribution_summary.png"), dpi=220)
    plt.close(fig)

def save_gate_visuals(model, raw_img, caption, save_root, sample_idx):
    cache = _get_encoder_focus_cache(model)
    gate_items = [item for item in cache if item.get("gate") is not None]
    if len(gate_items) == 0:
        return False
    raw_rgb = _raw_image_to_numpy(raw_img)
    out_hw = raw_rgb.shape[:2]
    folder = os.path.join(save_root, f"{sample_idx:04d}_{_safe_name(caption)}")
    os.makedirs(folder, exist_ok=True)

    cols = 1 + len(gate_items)
    fig_abs, axes_abs = plt.subplots(1, cols, figsize=(3 * cols, 3))
    fig_signed, axes_signed = plt.subplots(1, cols, figsize=(3 * cols, 3))
    fig_signed_rev, axes_signed_rev = plt.subplots(1, cols, figsize=(3 * cols, 3))
    if cols == 1:
        axes_abs = [axes_abs]
        axes_signed = [axes_signed]
        axes_signed_rev = [axes_signed_rev]
    axes_abs[0].imshow(raw_rgb)
    axes_abs[0].set_title("Image")
    axes_abs[0].axis("off")
    axes_signed[0].imshow(raw_rgb)
    axes_signed[0].set_title("Image")
    axes_signed[0].axis("off")
    axes_signed_rev[0].imshow(raw_rgb)
    axes_signed_rev[0].set_title("Image")
    axes_signed_rev[0].axis("off")
    for col, item in enumerate(gate_items, start=1):
        stage = item["stage"]
        gate_map = item["gate"]["gate_abs_spatial"][0].mean(dim=0, keepdim=True).unsqueeze(0)
        gate_map = F.interpolate(gate_map, size=out_hw, mode="bilinear", align_corners=False)[0, 0].numpy()
        low, high = np.percentile(gate_map, 1), np.percentile(gate_map, 99)
        gate_heat = _normalize_heatmap(gate_map, low, high, keep_percent=100)
        gate_overlay = _overlay_heatmap(raw_rgb, gate_heat, alpha=0.45)
        Image.fromarray(gate_overlay).save(os.path.join(folder, f"stage_{stage}_gate_abs_mean.png"))
        axes_abs[col].imshow(gate_overlay)
        axes_abs[col].set_title(f"S{stage} |gate|")
        axes_abs[col].axis("off")

        signed_map = item["gate"]["gate_signed_spatial"][0].mean(dim=0, keepdim=True).unsqueeze(0)
        signed_map = F.interpolate(signed_map, size=out_hw, mode="bilinear", align_corners=False)[0, 0].numpy()
        signed_heat = _normalize_signed_heatmap(signed_map)
        signed_overlay = _overlay_signed_heatmap(raw_rgb, signed_heat, alpha=0.45)
        Image.fromarray(signed_overlay).save(os.path.join(folder, f"stage_{stage}_gate_signed_mean.png"))
        axes_signed[col].imshow(signed_overlay)
        axes_signed[col].set_title(f"S{stage} signed")
        axes_signed[col].axis("off")

        signed_rev_overlay = _overlay_signed_heatmap(raw_rgb, signed_heat, alpha=0.45, cmap="seismic_r")
        Image.fromarray(signed_rev_overlay).save(os.path.join(folder, f"stage_{stage}_gate_signed_mean_reversed.png"))
        axes_signed_rev[col].imshow(signed_rev_overlay)
        axes_signed_rev[col].set_title(f"S{stage} signed-r")
        axes_signed_rev[col].axis("off")
    fig_abs.tight_layout()
    fig_abs.savefig(os.path.join(folder, "gate_abs_mean_grid.png"), dpi=200)
    plt.close(fig_abs)
    fig_signed.tight_layout()
    fig_signed.savefig(os.path.join(folder, "gate_signed_mean_grid.png"), dpi=200)
    plt.close(fig_signed)
    fig_signed_rev.tight_layout()
    fig_signed_rev.savefig(os.path.join(folder, "gate_signed_mean_reversed_grid.png"), dpi=200)
    plt.close(fig_signed_rev)
    return True

def save_encoder_focus_visuals(
    model,
    raw_img,
    gt_mask,
    caption,
    save_root,
    sample_idx,
    keep_percent=15.0,
    heatmap_postprocess="softmax_max_cc",
    softmax_temp=0.03,
):
    cache = _get_encoder_focus_cache(model)
    if len(cache) == 0:
        return False
    keep_percent = float(np.clip(keep_percent, 1.0, 100.0))

    raw_rgb = _raw_image_to_numpy(raw_img)
    out_hw = raw_rgb.shape[:2]
    folder = os.path.join(save_root, f"{sample_idx:04d}_{_safe_name(caption)}")
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(raw_rgb).save(os.path.join(folder, "image.png"))
    gt_mask = _mask_to_numpy(gt_mask, out_hw)
    Image.fromarray(gt_mask * 255).save(os.path.join(folder, "gt_mask.png"))
    gt_overlay = overlay_mask_on_image(raw_rgb, gt_mask, color=(0, 255, 0), alpha=0.45)
    Image.fromarray(gt_overlay).save(os.path.join(folder, "gt_overlay.png"))
    with open(os.path.join(folder, "caption.txt"), "w") as f:
        f.write(str(caption) + "\n")

    grid_cols = 2 + len(cache)
    fig, axes = plt.subplots(1, grid_cols, figsize=(3 * grid_cols, 3))
    if grid_cols == 1:
        axes = [axes]
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT")
    axes[1].axis("off")

    col = 2
    for item in cache:
        stage = item["stage"]
        attn_reduce = item.get("attn_reduce", "max")
        query_idx = item.get("query_idx", 0)
        topk = item.get("topk", 3)
        reduce_label = attn_reduce
        if attn_reduce == "query":
            reduce_label = f"query_{query_idx}"
        elif attn_reduce == "topk_mean":
            reduce_label = f"top{topk}_mean"
        attn_map = _attention_heatmap(item["attention"], out_hw)
        attn_heat = _postprocess_attention_heatmap(
            attn_map,
            keep_percent=keep_percent,
            postprocess=heatmap_postprocess,
            softmax_temp=softmax_temp,
        )
        attn_overlay = _overlay_heatmap(raw_rgb, attn_heat)

        Image.fromarray(attn_overlay).save(
            os.path.join(folder, f"stage_{stage}_{reduce_label}_{heatmap_postprocess}_cross_attention.png")
        )

        axes[col].imshow(attn_overlay)
        axes[col].set_title(f"S{stage} {reduce_label}")
        axes[col].axis("off")
        col += 1

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "encoder_focus_grid.png"), dpi=200)
    plt.close(fig)
    return True

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

def test_model(model, data_loader, device, logger, save_path, args):
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    iterations = 0
    pred_iou = 0
    miou_avg = AverageMeter()
    encoder_focus_saved = 0
    gate_visual_saved = 0
    gate_stats_saved = 0
    gate_records = []
    encoder_focus_root = args.encoder_focus_dir or os.path.join(save_path, "encoder_focus")
    gate_stats_root = args.gate_stats_dir or os.path.join(save_path, "gate_stats")
    if args.visualize_encoder_focus or args.visualize_gate_stats:
        os.makedirs(encoder_focus_root, exist_ok=True)
        os.makedirs(gate_stats_root, exist_ok=True)
        if not _set_encoder_focus_cache(model, False):
            logger.info("Fusion cache is unavailable for this model.")
        _configure_encoder_focus_attention(model, args)
    with torch.no_grad():
        total_num = 0
        for step, (imgs, cnn_imgs, target, captions, blip2_imgs, imgs_wo_trans, info) in enumerate(tqdm(data_loader)):
            imgs, target, blip2_imgs = imgs.to(device), target.to(device), blip2_imgs.to(device)
            cnn_imgs = cnn_imgs.to(device)
            captions = captions[0].split("#####")
            org_W, org_H = info["org_size"][0]
            org_mask = info["raw_masks"].unsqueeze(0)
            org_img = info["raw_imgs"]
            for obj_cap in captions:
                total_num += 1
                save_encoder_focus = (
                    total_num % 5 == 0
                    and args.visualize_encoder_focus
                    # and encoder_focus_saved < args.encoder_focus_samples
                )
                save_gate_stats = (
                    total_num % args.gate_stats_interval == 0
                    and args.visualize_gate_stats
                    # and gate_stats_saved < args.gate_stats_samples
                )
                _set_encoder_focus_cache(model, save_encoder_focus or save_gate_stats)
                pred_logits, low_pred_logits, query_embeds, object_score_logits, iou_predictions = model(imgs, cnn_imgs, imgs_wo_trans, blip2_imgs, target, obj_cap, step, 0, multi_mask=False)
                # save_encoder_focus = False
                # save_gate_stats = False
                if save_encoder_focus:
                    if save_encoder_focus_visuals(
                        model,
                        org_img,
                        org_mask,
                        obj_cap,
                        encoder_focus_root,
                        encoder_focus_saved,
                        keep_percent=args.encoder_focus_keep_percent,
                        heatmap_postprocess=args.encoder_focus_heatmap_postprocess,
                        softmax_temp=args.encoder_focus_softmax_temp,
                    ):
                        encoder_focus_saved += 1
                    _set_encoder_focus_cache(model, False)
                if save_gate_stats:
                    cache = _get_encoder_focus_cache(model)
                    _add_gate_records(gate_records, cache, total_num, obj_cap)
                    gate_stats_saved += 1
                    # print(save_gate_stats, gate_stats_saved)
                    if (args.gate_visual_samples <= 0 or gate_visual_saved < args.gate_visual_samples) and save_gate_visuals(
                        model,
                        org_img,
                        obj_cap,
                        gate_stats_root,
                        gate_visual_saved,
                    ):
                        gate_visual_saved += 1
                    _set_encoder_focus_cache(model, False)
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
                probabilities = pred_masks.detach().cpu().numpy()
                pred_masks = probabilities
                pred_masks = np.array(pred_masks > 0.5).astype(np.uint8)
                target_ = org_mask.cpu().data.numpy()
                # print(pred_masks.shape, target_.shape) # B C=1 H E
                I, U = IoU(pred_masks, target_)
                this_iou = 0.0 if U == 0 else I * 1.0 / U
                miou_avg.update(this_iou)
                mean_IoU.append(this_iou)

                # ---------------- visualize ----------------------#
                # if this_iou > 0:
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
                folder_name = obj_cap.replace(" ", "_")
                img_save_path = os.path.join(save_path, f"{seg_total}_{folder_name}")
                os.makedirs(img_save_path, exist_ok=True)
                if args.controlled_visuals:
                    model_1_iou = float("inf")
                    Image.fromarray((probabilities[0, 0] * 255).astype(np.uint8)).save(
                        os.path.join(img_save_path, "probability.png"))
                    Image.fromarray(sam2_pred[0] * 255).save(os.path.join(img_save_path, "mask.png"))
                    with open(os.path.join(img_save_path, "caption.txt"), "w") as f:
                        f.write(f"{obj_cap}, IoU:{this_iou}\n")
                else:
                    with open(os.path.join(img_save_path, "caption.txt"), "r") as f:
                        res_txt = f.read().strip()
                    model_1_iou = float(res_txt.split("IoU:")[-1])

                raw_img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                raw_img_pil.save(f"{img_save_path}/image.png")
                # logger.info(f"Num: {seg_total}")

                gt_img = overlay_mask_on_image(image_bgr.copy(), masks[0], color=(0, 255, 0))
                gt_pil = Image.fromarray(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
                gt_pil.save(f"{img_save_path}/gt.png")
                if this_iou < model_1_iou:
                    pred_img = overlay_mask_on_image(image_bgr.copy(), sam2_pred[0], color=(255, 0, 0))  # red for prediction
                    pred_pil = Image.fromarray(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
                    pred_pil.save(f"{img_save_path}/pred_2.png")

                    img_with_contours.save(os.path.join(img_save_path, f"visual.jpg"))
                # cv2.imwrite(os.path.join(img_save_path, f"uncertainty.jpg"), entropy_heat)
                # cur_iou = I / U if U != 0 else 0
                # with open(os.path.join(img_save_path, "caption.txt"), "w") as f:
                #     f.write(f"{obj_cap}, IoU:{cur_iou}\n")
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
    if args.visualize_gate_stats:
        _save_gate_summary_plots(gate_records, gate_stats_root)
        logger.info(f"Saved gate statistics and visualizations to {gate_stats_root}")

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.resume is None or not os.path.isfile(args.resume):
        raise FileNotFoundError("Pass --resume with the checkpoint to evaluate.")
    if args.controlled_visuals:
        args.pred_save_dir = os.path.join(args.pred_save_dir, args.ablation, args.tmfm_output)
    os.makedirs(args.log_path, exist_ok=True)
    args.log_path = os.path.join(args.log_path, f"{args.task_name}_test_e_38.txt")
    args.pretrain_model_path = config2model_path[args.model_cfg]

    logger = get_logger(args.log_path)

    device = torch.device(f"cuda:{args.device}")
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
            saved_config = checkpoint.get("experiment_config")
            if saved_config is not None:
                for key in ("ablation", "tmfm_output", "model_cfg", "blip2_vit_model", "image_size"):
                    if saved_config[key] != sam_ris_model.experiment_config[key]:
                        raise ValueError(f"Checkpoint configuration mismatch: {key}")
            sam_ris_model.load_state_dict(checkpoint["model"], strict=False)
    sam_ris_model.eval()
    os.makedirs(args.pred_save_dir, exist_ok=True)
    test_model(sam_ris_model, data_loader_test, device, logger, save_path = args.pred_save_dir, args=args)




if __name__ == "__main__":
    main()
