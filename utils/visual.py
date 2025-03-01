import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import ToPILImage
import random
import sys
import os
from PIL import Image, ImageDraw

def draw_contours(image, mask, gt = None, mask_color=(0, 255, 0), gt_color=(0, 0, 255)):
    """
    在给定的图像上绘制掩码和GT的轮廓。

    参数:
        image (numpy.ndarray): 原始图像作为numpy数组。
        mask (numpy.ndarray): 掩码作为numpy数组。
        gt (numpy.ndarray): 地面实况作为numpy数组。
        mask_color (tuple): 掩码轮廓的颜色 (BGR)。
        gt_color (tuple): GT轮廓的颜色 (BGR)。

    返回:
        numpy.ndarray: 绘制了轮廓的图像。
    """
    # 将掩码和GT转换为uint8格式
    mask = (mask * 255).astype(np.uint8)
    if gt.any():
        gt = (gt * 255).astype(np.uint8)

    # 如果掩码和GT不是单通道的，将其转换为单通道
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if gt.any():
        if len(gt.shape) > 2:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    # 找到轮廓
    contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if gt.any():
        contours_gt, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    image_with_contours = cv2.drawContours(image.copy(), contours_mask, -1, mask_color, 2)
    if gt.any():
        image_with_contours = cv2.drawContours(image_with_contours, contours_gt, -1, gt_color, 2)

    return image_with_contours


def tensor_to_heatmap(tensor):
    # Apply softmax to the tensor along the channel dimension
    # print(tensor.shape)
    softmaxed_tensor = torch.sigmoid(tensor)
    # print(torch.max(softmaxed_tensor), torch.min(softmaxed_tensor))

    # Find the maximum probability per pixel (max along the channel dimension)
    max_probs, _ = torch.max(softmaxed_tensor, dim=1)

    # Convert to numpy for visualization
    max_probs_np = max_probs.cpu().detach().numpy()

    # Normalize the max_probs to the range [0, 1] for better visualization
    min_val = np.min(max_probs_np)
    max_val = np.max(max_probs_np)
    # print(min_val, max_val)
    normalized_probs = (max_probs_np - min_val) / (max_val - min_val)
    # print(np.max(normalized_probs), np.min(normalized_probs))

    img = np.reshape(normalized_probs, normalized_probs.shape[-2:])
    heatmap = plt.get_cmap('turbo')(img)[:, :, :3]  # Get the RGB values from the colormap
    heatmap = (heatmap * 255).astype(np.uint8)  # Convert to 8-bit image
    # print(np.max(heatmap), np.min(heatmap))
    return heatmap

'''
dino v2对self attention可视化
'''
import colorsys
from matplotlib.patches import Polygon
from skimage.measure import find_contours
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


class ToTensorWithoutNormalization(object):
    def __call__(self, pic):
        """
        Converts a numpy.ndarray (H x W x C) in the range [0, 255]
        to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        without normalizing.
        """
        img = torch.from_numpy(pic)
        return img.float()

class InverseNormalize:
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        tensor = tensor * self.std + self.mean  # Reverse the normalization
        return tensor

from torchvision.transforms import ToPILImage

def tensor_to_pil_image(tensor):
    # 先进行逆归一化
    inverse_normalize = InverseNormalize()
    tensor = inverse_normalize(tensor)

    # 将张量的值限制在合法的范围内 [0, 1]，确保不会有超出范围的像素
    tensor = torch.clamp(tensor, 0, 1)

    # 转换为 PIL 图像
    pil_image = ToPILImage()(tensor)
    return pil_image

'''
query可视化的尝试
'''
def visual_from_dino(images, attentions, threshold = 0.5, feat_size = 16, patch_size = 14, query_idx = 1, visual_root="visual"):
    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, query_idx, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, feat_size, feat_size).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    image = tensor_to_pil_image(images[0].detach().cpu())
    image = np.array(image)
    # print(image.shape)
    os.makedirs(visual_root, exist_ok=True)
    for j in range(nh):
        save_path = os.path.join(visual_root, f"head_{j}.png")
        display_instances(image, th_attn[j], fname=save_path, blur=False)
    return th_attn

def visual_query_att_on_image(query_attn_maps, images, query_idx=0, head_idx = 0, feat_size = 16):
    nh = query_attn_maps.shape[1]
    # 16 16
    attentions = query_attn_maps[0, head_idx, query_idx, 1:].reshape(feat_size, feat_size)

    k = 5  # 选择 top-k 个值
    flattened_map = attentions.flatten()
    topk_values, topk_indices = torch.topk(flattened_map, k)  # 获取 top-k 的值和扁平化索引
    top_k_positions = torch.div(topk_indices, feat_size, rounding_mode='trunc'), topk_indices % feat_size  # 转换为二维索引

    # 3. 将索引映射到 224×224 的原始图像坐标
    scale_factor = 224 // feat_size
    mapped_positions = [(x.item() * scale_factor, y.item() * scale_factor) for x, y in zip(*top_k_positions)]

    # 4. 创建原始图像 (示例：白色背景图像)
    image = tensor_to_pil_image(images[0].detach().cpu())
    image = np.array(image)
    overlay = image.copy()  # 创建一个图层用于绘制透明矩形

    # 5. 在图层上绘制带透明度的红色方块
    alpha = 0.5  # 设置透明度 (0-完全透明, 1-完全不透明)
    for x, y in mapped_positions:
        cv2.rectangle(overlay, (y, x), (y + scale_factor, x + scale_factor), (0, 0, 255), thickness=-1)

    # 6. 合并透明图层和原始图像
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 7. 保存结果到文件
    output_path = "output_image.png"  # 保存文件名
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, image)  # 使用 OpenCV 保存

def visual_query_att_on_image_all_heads(query_attn_maps, images, save_path, query_idx=0,  feat_size = 16):

    nh = query_attn_maps.shape[1]

    # Determine grid size
    rows = (nh + 1) // 2  # Arrange in rows (2 columns per row)
    cols = 2  # You can adjust the column count as needed

    # Create a subplot grid to plot each attention map
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Flatten axes to make indexing easier
    axes = axes.flatten()

    scale_factor = 224 // feat_size
    k = 5  # Select top-k values

    for head_idx in range(nh):
        attentions = query_attn_maps[0, head_idx, query_idx, 1:].reshape(feat_size, feat_size)

        flattened_map = attentions.flatten()
        topk_values, topk_indices = torch.topk(flattened_map, k)
        top_k_positions = torch.div(topk_indices, feat_size, rounding_mode='trunc'), topk_indices % feat_size

        mapped_positions = [(x.item() * scale_factor, y.item() * scale_factor) for x, y in zip(*top_k_positions)]

        # Convert image to PIL format and then to NumPy
        image = tensor_to_pil_image(images[0].detach().cpu())
        image = np.array(image)

        overlay = image.copy()  # Create a layer for overlaying rectangles
        alpha = 0.5  # Set transparency

        for x, y in mapped_positions:
            cv2.rectangle(overlay, (y, x), (y + scale_factor, x + scale_factor), (0, 0, 255), thickness=-1)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Plot on the corresponding subplot
        ax = axes[head_idx]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Head {head_idx + 1}')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined image
    save_name = os.path.join(save_path, f"query_{query_idx}.png")
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)

    print(f"Combined image with all attention heads saved at: {save_name}")

def visual_query_attn_by_heat_map(query_attn_maps, images, save_path, feat_size = 16, cmap="jet"):
    layer_id = 0
    img_h, img_w = images.shape[2:]
    image = tensor_to_pil_image(images[0].detach().cpu())
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    # print(images.shape)
    for attn_map in query_attn_maps:
        if layer_id < 5:
            layer_id += 1
            continue
        b, heads, query_num, img_token_num = attn_map.shape
        for head in range(heads):
            # if head > 3:
            #     continue
            for query_id in range(query_num):
                img_name = f"layer_{layer_id}_head_{head}_query_{query_id}.jpg"
                cur_attn_map = attn_map[0, head, query_id, 1:].reshape(feat_size, feat_size)
                cur_attn_map = cur_attn_map.detach().cpu().numpy()
                img_save_path = os.path.join(save_path, img_name)
                mask = cv2.resize(cur_attn_map, (img_h, img_w))
                normed_mask = mask / mask.max()
                normed_mask = (normed_mask * 255).astype('uint8')
                plt.imshow(image, alpha=1)
                plt.axis('off')
                plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
                plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)

        layer_id += 1

def draw_points_on_numpy_image(image_array, points, output_path):
    """
    在NumPy图像上绘制点并保存。
    :param image_array: 原始图像的NumPy数组，形状为(H, W, 3)或(H, W)。
    :param points: 映射到原始图像尺寸的点坐标。
    :param output_path: 输出图像保存路径。
    """
    # 如果图像是灰度图，将其转换为RGB以便绘制彩色点
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    # 将NumPy数组转换为Pillow图像
    img = Image.fromarray(image_array.astype('uint8'))
    draw = ImageDraw.Draw(img)

    # 绘制点
    for point in points:
        x, y = point
        radius = 5  # 点的半径
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')

    # 保存结果
    img.save(output_path)
    # print(f"结果图像已保存到: {output_path}")


def draw_box_on_numpy_image(image_array, box, output_path):
    """
    在NumPy图像上绘制矩形框并保存。
    :param image_array: NumPy图像数据 (H, W, 3) 或 (H, W)。
    :param box: 矩形框 (x0, y0, x1, y1)。
    :param output_path: 输出图片的保存路径。
    """
    # 如果是灰度图像，转换为RGB格式
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    # 将 NumPy 图像转换为 PIL 图像
    img = Image.fromarray(image_array.astype('uint8'))
    draw = ImageDraw.Draw(img)

    # 绘制矩形框
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)  # 设置边框颜色和宽度

    # 保存图片
    img.save(output_path)
    # print(f"图片已保存到: {output_path}")