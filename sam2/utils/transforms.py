# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
# import torchvision.transforms.v2.functional as Fv2
from torchvision.transforms import Normalize, Resize, ToTensor
import dataset.transforms_video as T
import random
import torchvision.transforms as T2
from typing import List, Union, Iterable
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode
import logging

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)

def resize(image, mask, size, max_size=None, square=False):
    # Helper function to determine new size
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]  # return as (h, w)
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    # Handle square resizing if required
    if square:
        size = (size, size)  # Ensure size is (h, w) if square
    else:
        cur_size = image.size  # Current image size
        size = get_size(cur_size, size, max_size)

    # Resize the image
    image = F2.resize(image, size)

    # Resize the mask
    if mask is not None:
        mask = F2.resize(mask[None, None], size, interpolation=InterpolationMode.NEAREST).squeeze()

    return image, mask


class RandomResizeAPI:
    def __init__(self, sizes, consistent_transform, max_size=None, square=False):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform

    def __call__(self, images, masks, **kwargs):
        if self.consistent_transform:
            # Select a size for all frames (consistent resize)
            size = random.choice(self.sizes)
            for i in range(len(images)):
                images[i], masks[i] = resize(
                    images[i], masks[i], size, self.max_size, square=self.square
                )
        else:
            # Random resize for each frame independently
            for i in range(len(images)):
                size = random.choice(self.sizes)
                images[i], masks[i] = resize(
                    images[i], masks[i], size, self.max_size, square=self.square
                )
        return images, masks


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, images, masks, **kwargs):
        for i in range(len(images)):
            # if self.v2:
            #     images[i] = Fv2.to_image_tensor(images[i])
            # else:
            images[i] = F2.to_tensor(images[i])

            masks[i] = torch.as_tensor(masks[i], dtype=torch.uint8)
        return images, masks


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, images, masks, **kwargs):
        for i in range(len(images)):
            # if self.v2:
            #     images[i] = Fv2.convert_image_dtype(images[i], torch.float32)
            #     images[i] = Fv2.normalize(images[i], mean=self.mean, std=self.std)
            # else:
            images[i] = F2.normalize(images[i], mean=self.mean, std=self.std)
        return images, masks

class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, masks, **kwargs):
        for t in self.transforms:
            images, masks = t(images, masks, **kwargs)
        return images, masks

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        """
        初始化随机灰度变换类
        :param consistent_transform: 是否对所有图像应用一致的变换
        :param p: 应用灰度变换的概率
        """
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T2.Grayscale(num_output_channels=3)

    def __call__(self, images: List[Union[torch.Tensor, PILImage.Image]],
                 masks: List[Union[torch.Tensor, PILImage.Image]], **kwargs):
        """
        对传入的 images 进行随机灰度变换，masks 保持不变。
        :param images: 图像列表，每个元素是 torch.Tensor 或 PIL.Image
        :param masks: 对应的 mask 列表，保持不变
        :return: 经过灰度变换后的 images 和不变的 masks
        """
        # 如果要求一致变换，则检查是否对所有图像应用灰度
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(images)):
                    images[i] = self.Grayscale(images[i])
            # 返回处理后的图像和原始的 masks
            return images, masks

        # 否则对每个图像独立判断是否应用灰度变换
        for i in range(len(images)):
            if random.random() < self.p:
                images[i] = self.Grayscale(images[i])

        # 返回处理后的图像和原始的 masks
        return images, masks

class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        """
        初始化随机颜色抖动变换类
        :param consistent_transform: 是否对所有图像应用一致的变换
        :param brightness: 亮度调节范围
        :param contrast: 对比度调节范围
        :param saturation: 饱和度调节范围
        :param hue: 色调调节范围
        """
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else [-hue, hue]

    def __call__(self, images: List[Union[torch.Tensor, PILImage.Image]],
                 masks: List[Union[torch.Tensor, PILImage.Image]], **kwargs):
        """
        对传入的 images 进行随机颜色抖动变换，masks 保持不变。
        :param images: 图像列表
        :param masks: 对应的 mask 列表（不进行处理）
        :return: 处理后的 images 和原始的 masks
        """
        if self.consistent_transform:
            # 为所有图像生成一致的抖动参数
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T2.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            for img in images:
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F2.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F2.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F2.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F2.adjust_hue(img, hue_factor)
        else:
            # 对每个图像单独生成抖动参数
            for i in range(len(images)):
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T2.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        images[i] = F2.adjust_brightness(images[i], brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        images[i] = F2.adjust_contrast(images[i], contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        images[i] = F2.adjust_saturation(images[i], saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        images[i] = F2.adjust_hue(images[i], hue_factor)

        return images, masks

class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        随机仿射变换类
        :param degrees: 旋转角度范围
        :param consistent_transform: 是否对所有图像和 mask 使用一致的仿射变换
        :param scale: 缩放范围
        :param translate: 平移范围
        :param shear: 错切角度
        :param image_mean: 填充颜色
        :param log_warning: 是否记录警告
        :param num_tentatives: 允许的尝试次数
        :param image_interpolation: 图像插值方式
        """
        self.degrees = degrees if isinstance(degrees, list) else [-degrees, degrees]
        self.scale = scale
        self.translate = translate
        self.shear = shear if isinstance(shear, list) else [-shear, shear] if shear else None
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, images: List[Union[torch.Tensor, PILImage.Image]],
                 masks: List[Union[torch.Tensor, PILImage.Image]], **kwargs):
        """
        对传入的 images 和 masks 进行随机仿射变换
        :param images: 图像列表
        :param masks: mask 列表
        :return: 变换后的 images 和 masks
        """
        for _tentative in range(self.num_tentatives):
            res = self.transform_images_and_masks(images, masks)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} attempts"
            )
        return images, masks

    def transform_images_and_masks(self, images, masks):
        # 获取图像尺寸
        _, height, width = F2.get_dimensions(images[0])
        img_size = [width, height]

        if self.consistent_transform:
            # 对所有图像和 mask 生成一致的仿射变换参数
            affine_params = T2.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )
        # print(len(images), len(masks), masks[0].shape)
        for img_idx in range(len(images)):
            if not self.consistent_transform:
                # 单独为每张图像生成仿射变换参数
                affine_params = T2.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            # 变换图像和对应的 masks
            images[img_idx] = F2.affine(
                images[img_idx],
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )

            masks[img_idx] = F2.affine(
                masks[img_idx].unsqueeze(0),
                *affine_params,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).squeeze()

        return images, masks



class SAM2Transforms_Train(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.transforms = torch.jit.script(
        #     nn.Sequential(
        #         Resize((self.resolution, self.resolution)),
        #         Normalize(self.mean, self.std),
        #     )
        # )
        # affine, resize, colorjiter, gradscale, colorjiter
        self.transforms = ComposeAPI([
            RandomAffine(degrees=25, shear=20, image_interpolation="bilinear", consistent_transform=True),
            RandomResizeAPI(sizes=resolution, square=True, consistent_transform=True),
            ColorJitter(brightness=0.1, contrast=0.03, saturation=0.03, hue=None, consistent_transform=True),
            # RandomGrayscale(p=0.05, consistent_transform=True),
            # ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=None, consistent_transform=True),
            ToTensorAPI(),
            NormalizeAPI(mean=self.mean, std=self.std)
        ])

    def __call__(self, imgs, masks):
        # imgs: PIL.Image
        # masks: torch
        return self.transforms(imgs, masks)

class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        # 在前景区域的背景
        if self.max_hole_area > 0:
            # Holes are those connected components in background with area <= self.fill_hole_area
            # (background regions are those with mask scores <= self.mask_threshold)
            mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
            labels, areas = get_connected_components(mask_flat <= self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with a small positive mask score (10.0) to change them to foreground.
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

        # 在背景区域的前景
        if self.max_sprinkle_area > 0:
            labels, areas = get_connected_components(mask_flat > self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with negative mask score (-10.0) to change them to background.
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
