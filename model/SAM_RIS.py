import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry


def get_w(weights, keyword):
    return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}


def load_blip2_qformer(path, vit_model="eva_clip_g"):
    from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
    # vit_model="clip_L", "eva_clip_g"
    qformer = Blip2Qformer(vit_model=vit_model)
    weights = torch.load(path, map_location="cpu")
    model_weights = weights["model"]
    bert_weight = {_key: model_weights[_key] for _key in model_weights if "bert" in _key}
    missing_keys, unexpected_keys = qformer.Qformer.load_state_dict(
        get_w(bert_weight, "Qformer"), strict=False
    )
    print("missing_keys", missing_keys)
    print("unexpected_keys", unexpected_keys)
    qformer.ln_vision.load_state_dict(get_w(model_weights, "ln_vision"), strict=True)
    qformer.query_tokens.data.copy_(model_weights["query_tokens"])
    del qformer.Qformer.cls.predictions.decoder.bias
    print("load blip2 over")
    del weights
    return qformer


def resolve_sam_checkpoint(model_type, checkpoint):
    if checkpoint:
        return checkpoint

    candidates = {
        "vit_b": [
            "/data/weights/Pretrain/sam_vit_b_01ec64.pth",
            "/data/weights/Pretrain/sam_vit_b.pth",
        ],
        "vit_h": [
            "/data/weights/Pretrain/sam_vit_h_4b8939.pth",
            "/data/weights/Pretrain/sam_vit_h.pth",
        ],
    }
    for path in candidates.get(model_type, []):
        if os.path.isfile(path):
            return path
    return candidates.get(model_type, [None])[0]


def build_sam_ris_backbone(model_type="vit_b", checkpoint=None, image_size = 1024):
    checkpoint = resolve_sam_checkpoint(model_type, checkpoint)
    if checkpoint is None or not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint for {model_type} not found. Put it under "
            f"/data/weights/Pretrain or pass --sam_pretrain_path. Expected: {checkpoint}"
        )
    return sam_model_registry[model_type](checkpoint=checkpoint, image_size=image_size)


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return self.dropout(x)


class VisionMultiModalQueryFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, input_feat_size=768):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multi_modal_resizer = FeatureResizer(
            input_feat_size=input_feat_size,
            output_feat_size=d_model,
            dropout=dropout,
        )
        self.res_gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        memory = self.multi_modal_resizer(memory)
        bsz = memory.shape[0]
        _, channels, height, width = tgt.shape

        tgt_seq = tgt.flatten(2).permute(2, 0, 1)
        memory_seq = memory.permute(1, 0, 2)
        query2img_tgt, _ = self.multihead_attn(
            query=self.with_pos_embed(tgt_seq, query_pos),
            key=self.with_pos_embed(memory_seq, pos),
            value=memory_seq,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
        )
        q2i_tgt = tgt_seq + self.res_gate(query2img_tgt) * query2img_tgt
        q2i_tgt = q2i_tgt.permute(1, 2, 0).reshape(bsz, channels, height, width)
        return q2i_tgt, None


class SAMImageEncoderWithFusion(nn.Module):
    def __init__(self, image_encoder, input_feat_size=768):
        super().__init__()
        self.image_encoder = image_encoder
        self.img_size = image_encoder.img_size
        self.fusion_indexes = self._build_fusion_indexes(len(image_encoder.blocks))
        embed_dim = image_encoder.pos_embed.shape[-1]
        self.vision_multi_modal_fusions = nn.ModuleList(
            [
                VisionMultiModalQueryFusionModule(
                    d_model=embed_dim,
                    input_feat_size=input_feat_size,
                    nhead=8,
                    dropout=0.1,
                )
                for _ in self.fusion_indexes
            ]
        )

    @staticmethod
    def _build_fusion_indexes(depth):
        predefined = {
            12: [2, 5, 8, 11],
            24: [5, 11, 17, 23],
            32: [7, 15, 23, 31],
        }
        if depth in predefined:
            return predefined[depth]
        return sorted({max(0, min(depth - 1, round((i + 1) * depth / 4) - 1)) for i in range(4)})

    def _get_pos_embed(self, hw):
        pos_embed = self.image_encoder.pos_embed
        if pos_embed is None:
            return None
        if pos_embed.shape[1:3] == hw:
            return pos_embed
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=hw, mode="bicubic", align_corners=False)
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, x, multi_modal_queries=None):
        x = self.image_encoder.patch_embed(x)
        pos_embed = self._get_pos_embed(x.shape[1:3])
        if pos_embed is not None:
            x = x + pos_embed

        fusion_stage = 0
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if multi_modal_queries is not None and i in self.fusion_indexes:
                feats = x.permute(0, 3, 1, 2)
                feats, _ = self.vision_multi_modal_fusions[fusion_stage](
                    tgt=feats,
                    memory=multi_modal_queries,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None,
                )
                x = feats.permute(0, 2, 3, 1)
                fusion_stage += 1

        return self.image_encoder.neck(x.permute(0, 3, 1, 2))


class SAM_RIS(nn.Module):
    def __init__(self, model, args, target_size=1024, device="cuda:0"):
        super().__init__()
        self.device = device
        self.sam_model = model
        self.sam_model.image_encoder = SAMImageEncoderWithFusion(self.sam_model.image_encoder)
        self.blip2_qformer = load_blip2_qformer(args.blip2_path, args.blip2_vit_model)
        self.query_resizer = FeatureResizer(input_feat_size=768, output_feat_size=256, dropout=0)
        self.target_size = args.image_size
        self.args = args

        for param in self.sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        self.sam_model.prompt_encoder.no_mask_embed.requires_grad_(True)

        for param in self.sam_model.mask_decoder.parameters():
            param.requires_grad = True

        for name, param in self.sam_model.image_encoder.named_parameters():
            param.requires_grad = "fusion" in name

        self.blip2_qformer.query_tokens.requires_grad_(True)
        for name, param in self.blip2_qformer.named_parameters():
            param.requires_grad = "Qformer" in name

    def set_model_save_path(self, model_save_path):
        self.save_path = model_save_path

    def generate_queries(self, blip2_imgs, repeat_catpipns):
        blip2_output, final_cross_attention_score = self.blip2_qformer.extract_features(
            {"image": blip2_imgs, "text_input": repeat_catpipns}, mode="multimodal"
        )
        raw_multimodal_queries = blip2_output.multimodel_query
        multimodal_queries = blip2_output.multimodal_embeds
        multi_modal_embed = self.query_resizer(multimodal_queries)
        return raw_multimodal_queries, multi_modal_embed, multimodal_queries, final_cross_attention_score

    def _dense_no_mask_embedding(self, batch_size, image_embed):
        prompt_encoder = self.sam_model.prompt_encoder
        dense_embeddings = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size,
            -1,
            prompt_encoder.image_embedding_size[0],
            prompt_encoder.image_embedding_size[1],
        )
        return dense_embeddings.to(dtype=image_embed.dtype, device=image_embed.device)

    def _image_encoder(self, input_image, multimodal_queries=None):
        image_embed = self.sam_model.image_encoder(input_image, multimodal_queries)
        prompt_size = self.sam_model.prompt_encoder.image_embedding_size
        if image_embed.shape[-2:] != prompt_size:
            image_embed = F.interpolate(
                image_embed,
                size=prompt_size,
                mode="bilinear",
                align_corners=False,
            )
        return image_embed

    def _decode_masks(self, img_embed, sparse_embeddings, dense_embeddings, multi_mask):
        low_res_masks = []
        iou_predictions = []
        image_pe = self.sam_model.prompt_encoder.get_dense_pe().to(
            img_embed.device, dtype=img_embed.dtype
        )
        for idx in range(img_embed.shape[0]):
            sample_masks, sample_ious = self.sam_model.mask_decoder(
                image_embeddings=img_embed[idx : idx + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings[idx : idx + 1],
                dense_prompt_embeddings=dense_embeddings[idx : idx + 1],
                multimask_output=multi_mask,
            )
            low_res_masks.append(sample_masks)
            iou_predictions.append(sample_ious)
        return torch.cat(low_res_masks, dim=0), torch.cat(iou_predictions, dim=0)

    def forward(self, image, cnn_image, raw_imgs, blip2_imgs, masks, captions, step, epoch, multi_mask=False):
        raw_multimodal_queries, multi_modal_embed, multimodal_queries, final_cross_attention_score = (
            self.generate_queries(blip2_imgs=blip2_imgs, repeat_catpipns=captions)
        )
        img_embed = self._image_encoder(image, raw_multimodal_queries)

        sparse_embeddings = multi_modal_embed
        dense_embeddings = self._dense_no_mask_embedding(img_embed.shape[0], img_embed)
        low_res_masks_logits, iou_predictions = self._decode_masks(
            img_embed,
            sparse_embeddings,
            dense_embeddings,
            multi_mask,
        )

        masks_logits = F.interpolate(
            low_res_masks_logits,
            (self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        object_score_logits = None
        return masks_logits, low_res_masks_logits, multi_modal_embed, object_score_logits, iou_predictions
