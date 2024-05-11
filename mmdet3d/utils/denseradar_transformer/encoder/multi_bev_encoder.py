import copy
import warnings
from mmcv.cnn.bricks.transformer import (
    TRANSFORMER_LAYER_SEQUENCE,
    ATTENTION,
    TransformerLayerSequence,
    build_attention,
)
from mmcv.cnn import xavier_init
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MultiBEVFusion(TransformerLayerSequence):
    def __init__(
        self,
        return_intermediate=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.return_intermediate = return_intermediate

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        *args,
        **kwargs,
    ):
        if len(query.shape) == 4:
            query = query.reshape(*query.shape[:2], -1).permute(2, 0, 1)
        if len(key.shape) == 4:
            key = key.reshape(*key.shape[:2], -1).permute(2, 0, 1)

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                *args,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@ATTENTION.register_module()
class MultiBEVCrossAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        radar_embed_dims=64,
        num_heads=4,
        d_head=32,
        dropout=0.1,
        bias=True,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=256,
            num_points=8,
            num_levels=1,
        ),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(init_cfg, **kwargs)
        self.embed_dims = embed_dims
        self.radar_embed_dims = radar_embed_dims
        self.num_heads = num_heads
        self.d_head = d_head
        self.dropout = dropout
        self.bias = bias

        self.atten_scale = self.d_head**-0.5
        self.dropout = nn.Dropout(self.dropout)

        self.batch_first = batch_first

        self.init_layers()
        self.init_weight()

    def init_layers(self):
        self.query_proj = nn.Linear(self.embed_dims, self.num_heads * self.d_head)
        self.key_proj = nn.Linear(self.radar_embed_dims, self.num_heads * self.d_head)
        self.value_proj = nn.Linear(self.radar_embed_dims, self.num_heads * self.d_head)

        self.output_proj = nn.Linear(self.num_heads * self.d_head, self.embed_dims)

    def init_weight(self):
        xavier_init(self.query_proj, distribution="uniform", bias=0)
        xavier_init(self.key_proj, distribution="uniform", bias=0)
        xavier_init(self.value_proj, distribution="uniform", bias=0)
        xavier_init(self.output_proj, distribution="uniform", bias=0)
        # nn.init.normal_(self.query_proj.weight,mean=0,std=np.sqrt(2.0 / (d_model + d_k))))

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
    ):
        """MultiBEVCrossAttention forward computing
        Args:
            query (torch.tensor): [num_query, bs, embed_dims], query stands for img_lss_bev.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is"
                        f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # computing the multi-head attention between query and (key & value).
        n_query, bs, _ = query.shape
        n_key, _, _ = key.shape
        query = (
            self.query_proj(query.permute(1, 0, 2))
            .reshape(bs, n_query, self.num_heads, self.d_head)
            .permute(2, 0, 1, 3)
        )  # bs, num_query, embed_dims -> bs, num_query, num_heads  d_head

        key = (
            self.key_proj(key.permute(1, 0, 2))
            .reshape(bs, n_key, self.num_heads, self.d_head)
            .permute(2, 0, 1, 3)
        )
        value = (
            self.value_proj(value.permute(1, 0, 2))
            .reshape(bs, n_key, self.num_heads, self.d_head)
            .permute(2, 0, 1, 3)
        )

        attention = self.atten_scale * torch.einsum(
            "h b q d, h b k d -> h b q k",
            query,
            key,
        )
        attention = F.softmax(attention, dim=-1)

        output = torch.einsum("h b q k, h b k d -> h b q d", attention, value)
        output = rearrange(output, "h b q d -> b q (h d)")

        output = self.output_proj(output)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiBEVCrossAttention_V2(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        radar_embed_dims=64,
        num_heads=4,
        d_head=32,
        dropout=0.1,
        bias=True,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=256, num_levels=1
        ),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(init_cfg, **kwargs)
        self.embed_dims = embed_dims
        self.radar_embed_dims = radar_embed_dims
        self.num_heads = num_heads
        self.d_head = d_head
        self.dropout = dropout
        self.bias = bias
        self.deformable_attention = deformable_attention

        self.atten_scale = self.d_head**-0.5
        self.dropout = nn.Dropout(self.dropout)

        self.batch_first = batch_first

        self.init_layers()
        self.init_weight()

    def init_layers(self):
        self.query_proj = nn.Linear(
            self.embed_dims, self.num_heads * self.d_head, bias=self.bias
        )
        self.key_proj = nn.Linear(
            self.radar_embed_dims, self.num_heads * self.d_head, bias=self.bias
        )
        self.value_proj = nn.Linear(
            self.radar_embed_dims, self.num_heads * self.d_head, bias=self.bias
        )

        self.output_proj = nn.Linear(
            self.num_heads * self.d_head, self.embed_dims, bias=self.bias
        )

        self.deformable_attention = build_attention(self.deformable_attention)

    def init_weight(self):
        xavier_init(self.query_proj, distribution="uniform", bias=0)
        xavier_init(self.key_proj, distribution="uniform", bias=0)
        xavier_init(self.value_proj, distribution="uniform", bias=0)
        xavier_init(self.output_proj, distribution="uniform", bias=0)
        # nn.init.normal_(self.query_proj.weight,mean=0,std=np.sqrt(2.0 / (d_model + d_k))))

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
    ):
        """MultiBEVCrossAttention forward computing
        Args:
            query (torch.tensor): [num_query, bs, embed_dims], query stands for img_lss_bev.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is"
                        f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        return identity + None
