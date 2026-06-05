# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union, Optional

from einops import rearrange
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP
import math

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


from timm.models.vision_transformer import Mlp
class LocalTokenAdapter(nn.Module):
    def __init__(self, sam_dim, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.sam_feat_down_proj = MLP(input_dim=sam_dim,
                    hidden_dim=dim,
                    output_dim=dim,
                    num_layers=2)
        self.sam_feat_up_proj = MLP(input_dim=dim,
                    hidden_dim=sam_dim,
                    output_dim=sam_dim,
                    num_layers=2)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            # nn.Sigmoid()
            # nn.Tanh()
            nn.Tanh()
        )

    def forward(self, query, image_token):
        # print(query.shape)
        down_query = self.sam_feat_down_proj(query)
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = down_query.shape
        q = (
            self.q(down_query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        update_down_query = down_query + self.gate(x) * x
        up_query = self.sam_feat_up_proj(update_down_query)
        query = query + up_query
        return query

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

class VisionMultiModalQueryFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, input_feat_size=768, need_query_resizer = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multi_modal_resizer = FeatureResizer(input_feat_size=input_feat_size, output_feat_size=d_model, dropout=dropout)
        self.res_gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False),
            # nn.Sigmoid()
            # nn.Tanh()
            nn.Tanh()
        )


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # print(memory.shape)
        memory = self.multi_modal_resizer(memory) # B L C
        # print(memory.shape)
        # clip
        # B = memory.shape[0] // T
        # single image
        B = memory.shape[0]
        T = tgt.shape[0] // B

        memory = memory.permute(1, 0, 2) # L B C
        # print(memory.shape, pos.shape, tgt.shape)
        H, W = tgt.shape[2:4]
        tgt = rearrange(tgt, "(b t) c h w -> (t h w) b c", b=B, t=T)
        memory = rearrange(memory, "l (b t) c -> (l t) b c", b=B, t=T)
        query2img_tgt, _ = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)
        q2i_tgt = tgt + (self.res_gate(query2img_tgt) * query2img_tgt)
        # q2i_tgt = tgt + query2img_tgt
        q2i_tgt = rearrange(q2i_tgt, "(t h w) b c -> (b t) c h w", t=T, h = H, w = W)
        i2q_tgt = None
        ###************************###
        # query2img_tgt, _ = self.query2img(query=self.with_pos_embed(tgt, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=None,
        #                            key_padding_mask=memory_key_padding_mask)
        #
        # # add img2query to update query
        # img2query_tgt, _ = self.img2query(query=self.with_pos_embed(memory, query_pos),
        #                                        key=self.with_pos_embed(tgt, pos),
        #                                        value=tgt, attn_mask=None,
        #                                        key_padding_mask=memory_key_padding_mask)
        #
        # q2i_tgt = tgt + (self.res_gate_img(query2img_tgt) * query2img_tgt)
        # q2i_tgt = rearrange(q2i_tgt, "(t h w) b c -> (b t) c h w", t=T, h = H, w = W)
        # i2q_tgt = memory + (self.res_gate_query(img2query_tgt) * img2query_tgt)
        # i2q_tgt = rearrange(i2q_tgt, "(l t) b c -> (b t) l c", t=T)
        ###************************###
        return q2i_tgt, i2q_tgt

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()
        self.vision_multi_modal_fusions = nn.ModuleList()

        # self.vision_text_fusions = nn.ModuleList()
        pre_dim = 768
        sam2_dims = []
        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                sam2_dims.append(dim_out)
                # 视觉-语言融合，只在后三个层次的，选所有层次的多出600M内存
                # if cur_stage > 1:
                # multi-modal queries
                # fusion_module = VisionMultiModalQueryFusionModule(d_model=dim_out, input_feat_size=768, nhead=8, dropout=0.1, need_query_resizer = (cur_stage == 1))
                # self.vision_multi_modal_fusions.append(fusion_module)

                # 每经过一个阶段，通道数翻倍 for base: 112 -> 224 -> 448 -> 896
                dim_out = int(embed_dim * dim_mul)
                pre_dim = embed_dim
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )
            # every layer
            # multi-modal queries
            # fusion_module = VisionMultiModalQueryFusionModule(d_model=dim_out, input_feat_size=768, nhead=8, dropout=0.1, need_query_resizer = (cur_stage == 1))
            # self.vision_multi_modal_fusions.append(fusion_module)

            embed_dim = dim_out
            self.blocks.append(block)

        # 最后一层
        fusion_module = VisionMultiModalQueryFusionModule(d_model=dim_out, input_feat_size = 768, nhead=8, dropout=0.1)
        self.vision_multi_modal_fusions.append(fusion_module)
        sam2_dims.append(dim_out)
        print(sam2_dims)

        # self.cnn2sam_fusion = nn.ModuleList([
        #     LocalTokenAdapter(sam_dim=sam2_dim, dim=64, num_heads=4) for sam2_dim in sam2_dims
        # ])

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed


    def forward(self, x: torch.Tensor, multi_modal_queries = None, cnn_tokens = None) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        stage = 0
        early_layer_idx = len(self.blocks) // 8
        mid_layer_idx = len(self.blocks) // 2
        last_layer_idx = len(self.blocks) - 1
        layer_idx = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # every layer
            if multi_modal_queries != None and i == early_layer_idx:
                feats = x.permute(0, 3, 1, 2)
                feats, _ = self.vision_multi_modal_fusions[layer_idx](tgt=feats,
                                            memory=multi_modal_queries,
                                            memory_key_padding_mask=None,
                                            pos=None,
                                            query_pos=None)
                x = feats.permute(0, 2, 3, 1)
                layer_idx += 1
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                # reshape feats to B N C
                B, H, W, C = x.shape
                # print(x.shape)
                # b c h w
                feats = x.permute(0, 3, 1, 2)
                
                # if stage > 1:
                # 进行融合 multi-modal
                # if multi_modal_queries != None:
                #     feats, _ = self.vision_multi_modal_fusions[stage](tgt=feats,
                #                              memory=multi_modal_queries,
                #                              memory_key_padding_mask=None,
                #                              pos=None,
                #                              query_pos=None)
                #     outputs.append(feats)
                #     stage += 1
                #     # 这一步显存要求很大
                #     x = feats.permute(0, 2, 3, 1)
                # else:
                outputs.append(feats)
                stage += 1

        '''outputs组成: 1024
            torch.Size([9, 96, 256, 256])
            torch.Size([9, 192, 128, 128])
            torch.Size([9, 384, 64, 64])
            torch.Size([9, 768, 32, 32])
        '''
        return outputs, multi_modal_queries
