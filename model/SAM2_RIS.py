import torch.nn as nn
import torch
from sam2.build_sam import build_sam2
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from einops import rearrange, repeat
import torch.nn.functional as F
import timm
from utils.visual import draw_contours


def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

def load_blip2_qformer(path):
    qformer = Blip2Qformer()
    weights = torch.load(path, map_location='cpu')
    print(weights.keys())  # dict_keys(['model', 'optimizer', 'config', 'scaler', 'iters'])
    model_weights = weights["model"]
    bert_weight = {_key: model_weights[_key] for _key in model_weights if 'bert' in _key}
    # print([_key for _key in model_weights if 'Qformer' in _key])
    # print([_key for _key in model_weights if 'bert' in _key])
    # print(get_w(bert_weight, "Qformer"))
    qformer.Qformer.load_state_dict(get_w(bert_weight, "Qformer"), strict=False)
    qformer.ln_vision.load_state_dict(get_w(model_weights, "ln_vision"))
    qformer.query_tokens.data.copy_(model_weights['query_tokens'])
    del qformer.Qformer.cls.predictions.decoder.bias
    print("load blip2 over")
    del weights
    return qformer

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class CNNTokenGenration(nn.Module):
    def __init__(self, feature_channels = [256, 512, 1024, 2048], latent_dim = 64, grid_size = 7, cnn_feature_num = 4):
        super().__init__()
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            feature_channels[i],
                            latent_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            latent_dim,
                            latent_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.AdaptiveAvgPool2d(grid_size),
                    ]
                )
                for i in range(cnn_feature_num)
            ]
        )


class SAM2_RIS(nn.Module):
    def __init__(self, model, args, target_size=1024, device="cuda:0"):
        super().__init__()
        self.device = device
        self.sam2_model = model
        self.blip2_qformer = load_blip2_qformer(args.blip2_path)
        # multi model queries
        self.query_resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=256,
            dropout=0,
        )
        # self.cnn_extractor = timm.create_model(
        #     "resnet50",
        #     pretrained=True,
        #     features_only=True,
        #     out_indices=[1, 2, 3, 4],
        # )
        # self.cnn_token_generator = CNNTokenGenration(feature_channels=[256, 512, 1024, 2048])
        self.target_size = args.image_size
        # 特征大小
        feat_sizes = [256 // 2 ** i * self.target_size // 1024 for i in range(3)]
        self.bb_feat_sizes = [(feat_sizes[0], feat_sizes[0]), (feat_sizes[1], feat_sizes[1]),
                              (feat_sizes[2], feat_sizes[2])]

        # focal loss参数
        self.focal_alpha_obj_score = 0.25
        self.focal_gamma_obj_score = 2

        self.args = args

        # freeze cnn encoder
        # for param in self.cnn_extractor.parameters():
        #     param.requires_grad = False

        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        # prompt encoder可训练
        self.sam2_model.sam_prompt_encoder.no_mask_embed.requires_grad_(True)
        # mask decoder可训练
        for name, param in self.sam2_model.sam_mask_decoder.named_parameters():
            param.requires_grad = True
        for name, param in self.sam2_model.image_encoder.named_parameters():
            if "fusion" in name:
                param.requires_grad = True
                continue
            param.requires_grad = False
        self.blip2_qformer.query_tokens.requires_grad_(True)
        # qformer可训练
        for name, param in self.blip2_qformer.named_parameters():
            if "Qformer" in name:
                # print(name)
                param.requires_grad = True
                continue
            param.requires_grad = False

    def set_model_save_path(self, model_save_path):
        self.save_path = model_save_path

    def generate_queries(self, blip2_imgs, repeat_catpipns):
        blip2_output, final_cross_attention_score = self.blip2_qformer.extract_features(
            {"image": blip2_imgs, "text_input": repeat_catpipns}, mode="multimodal")

        raw_multimodal_queries = blip2_output.multimodel_query # not fuse with text again, length=32
        multimodal_queries = blip2_output.multimodal_embeds # fuse with text again, length=1
        multi_modal_embed = self.query_resizer(multimodal_queries)

        return raw_multimodal_queries, multi_modal_embed, multimodal_queries, final_cross_attention_score

    def generate_cnn_tokens(self, feature_list):
        batch_size = feature_list[0].shape[0]
        cnn_token_list = []
        for i in range(len(feature_list)):
            cnn_image_token = self.cnn_token_generator.conv[i](feature_list[i])
            latent_dim = cnn_image_token.shape[1]
            cnn_image_token = cnn_image_token.permute(0, 2, 3, 1).reshape(
                batch_size, -1, latent_dim
            )
            cnn_token_list.append(cnn_image_token)
        return torch.cat(cnn_token_list, dim=1)

    def forward(self, image, cnn_image, raw_imgs, blip2_imgs, masks, captions, step, epoch, multi_mask=False):
        """
        image: (B, 3, 1024, 1024)
        """
        # feature_list = self.cnn_extractor(cnn_image)
        # cnn_tokens = self.generate_cnn_tokens(feature_list)
        cnn_tokens = None
        # print(cnn_tokens.shape) # 4 N=196 C=64
        raw_multimodal_queries, multi_modal_embed, multimodal_queries, final_cross_attention_score = self.generate_queries(blip2_imgs=blip2_imgs,
                                                                          repeat_catpipns=captions)

        _features, vision_feats, vision_pos_embeds, query = self._image_encoder(image, raw_multimodal_queries, cnn_tokens)

        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]

        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embed=multi_modal_embed,
            Batch_size=img_embed.shape[0]
        )

        # print(img_embed.shape, sparse_embeddings.shape)
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multi_mask,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # B 1 H W
        masks_logits = F.interpolate(low_res_masks_logits, (self.target_size, self.target_size),
                                             mode="bilinear", align_corners=False)

        # if self.args.train_visual and step % self.args.save_visual_interval == 0:
        #     visual_save_path = os.path.join(self.save_path, "visual")
        #     os.makedirs(visual_save_path, exist_ok=True)
        #     save_pred_mask = torch.zeros_like(masks_logits)
        #     print(torch.max(masks_logits[0]), torch.min(masks_logits[0]))
        #     save_pred_mask[masks_logits > 0.5] = 1
        #     save_pred_mask = save_pred_mask[1].detach().cpu().numpy()
        #     gt_mask = masks[1].detach().cpu().numpy()
        #     select_img = raw_imgs[1].detach().cpu().numpy()
        #     # print(select_img.shape)
        #     # 将图像转换为BGR格式，因为OpenCV使用BGR
        #     image_bgr = select_img[:, :, ::-1].astype(np.uint8)
        #     # print(np.max(select_img, axis=0))

        #     # 绘制轮廓
        #     image_with_contours = draw_contours(image_bgr, save_pred_mask[0], gt_mask[0])

        #     # 将绘制了轮廓的图像转换回RGB格式
        #     image_with_contours_rgb = image_with_contours[:, :, ::-1]
        #     # print(image_with_contours_rgb.shape)

        #     # 创建 PIL 图像对象
        #     img_with_contours = Image.fromarray(image_with_contours_rgb)

        #     # 保存图像
        #     save_path = os.path.join(visual_save_path, f"{epoch}-{step}")
        #     os.makedirs(save_path, exist_ok=True)
        #     img_with_contours.save(os.path.join(save_path, f"visual.jpg"))
        #     with open(os.path.join(save_path, "caption.txt"), "w") as f:
        #         for exp in captions:
        #             f.write(exp + "\n")
        #         # f.write(str(pred_mask_score))

        #     # 保存特征图
        #     # to_pil_image = transforms.ToPILImage()
        #     sample = high_res_features[0][1].to(torch.float32).detach().cpu().numpy()
        #     channel_num = [i for i in range(sample.shape[0])]
        #     random.shuffle(channel_num)

        #     channel_num = channel_num[:25]

        #     plt.figure(figsize=(10, 10))
        #     for index, channel in enumerate(channel_num):
        #         ax = plt.subplot(5, 5, index + 1, )
        #         plt.imshow(sample[channel, :, :])
        #     plt.savefig(os.path.join(save_path, "feature.jpg"), dpi=300)
        #     plt.close()

        return masks_logits, low_res_masks_logits, multi_modal_embed, object_score_logits, iou_predictions

    def _image_encoder(self, input_image, multimodal_queries=None, cnn_tokens = None, is_memory=False):
        backbone_out = self.sam2_model.forward_image(input_image, multimodal_queries, cnn_tokens)
        # print(backbone_out[0].shape, backbone_out[1].shape)
        # 选择哪几个层次的特征用来处理
        _, vision_feats, vision_pos_embeds, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed and is_memory == False:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        # bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
                    feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
                ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features, vision_feats, vision_pos_embeds, backbone_out["query"]