import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

# import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init, build_activation_layer
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    TRANSFORMER_LAYER_SEQUENCE,
    FEEDFORWARD_NETWORK,
)
from mmcv.cnn.bricks.transformer import (
    TransformerLayerSequence,
    MultiheadAttention,
    build_dropout,
)
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (
    ConfigDict,
    build_from_cfg,
    deprecated_api_warning,
    to_2tuple,
)

from torch.utils.checkpoint import checkpoint as cp

# from .checkpoint import checkpoint as cp

from mmcv.utils import ext_loader
from ..attention.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32,
    MultiScaleDeformableAttnFunction_fp16,
)

from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)
from mmdet3d.core.bbox.utils import decode_bbox

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                    reference_points[..., 2:3]
                )

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    @deprecated_api_warning(
        {"residual": "identity"}, cls_name="MultiScaleDeformableAttention"
    )
    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        # if not self.batch_first:
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DenseRadarTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        transformerlayers=None,
        num_layers=None,
        return_intermediate=False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(transformerlayers, num_layers, init_cfg, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        reg_branches=None,
        cls_branches=None,
        level_start_index=None,
        **kwargs,
    ):
        """Decoder forward

        Args:
            query (torch.tensor, Embedding): [bs, num_query_len, embedd_dims] object query embedding
            key (_type_, optional): _description_. Defaults to None.
            value (_type_, optional): _description_. Defaults to None.
            query_pos (_type_, optional): _description_. Defaults to None.
            reference_points (_type_, optional): _description_. Defaults to None.
            reg_branches (_type_, optional): _description_. Defaults to None.
            cls_branches (_type_, optional): _description_. Defaults to None.
            level_start_index (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                reference_points=reference_points,
                level_start_index=level_start_index,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                # tmp: (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
                new_reference_points = torch.zeros_like(reference_points)

                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                    reference_points[..., 2:3]
                )

                # new_reference_points[..., :2] = tmp[..., :2] + reference_points[..., :2]

                # new_reference_points[..., 2:3] = (
                #     tmp[..., 4:5] + reference_points[..., 2:3]
                # )

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class DNObjQuerySelfAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        pc_range=[],
        batch_first=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self.attention = MultiheadAttention(
            embed_dims, num_heads, dropout, batch_first=True
        )
        self.gen_tau = nn.Linear(embed_dims, num_heads)

        self.batch_first = batch_first

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(
                centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2),
                dim=-1,
            )
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist

    # def forward(self, query_bbox, query_feat, pre_attn_mask):
    def _forward(
        self,
        query,
        query_pos=None,
        reference_points=None,
        # query_bbox=None,
        # query_feat=None,
        pre_attn_mask=None,
    ):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """

        if query_pos is not None:
            query = query + query_pos

        # reference_points = reference_points.permute(1, 0, 2)
        query = query.permute(1, 0, 2)

        dist = self.calc_bbox_dists(reference_points)
        tau = self.gen_tau(query)  # [B, Q, 8]

        # if DUMP.enabled:
        #     torch.save(
        #         tau,
        #         "{}/sasa_tau_stage{}.pth".format(
        #             DUMP.out_dir, DUMP.stage_count
        #         ),
        #     )

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]
        if pre_attn_mask is not None:
            attn_mask[:, :, pre_attn_mask] = float("-inf")

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]

        output = self.attention(query, attn_mask=attn_mask)
        output = output.permute(1, 0, 2)

        return output

    def forward(
        self,
        query,
        *args,
        query_pos=None,
        reference_points=None,
        # query_bbox=None,
        # query_feat=None,
        pre_attn_mask=None,
        **kwargs,
    ):
        if self.training and query.requires_grad:
            return cp(
                self._forward,
                query,
                query_pos,
                reference_points,
                # query_bbox=None,
                # query_feat=None,
                pre_attn_mask,
            )
        else:
            return self._forward(
                query,
                query_pos,
                reference_points,
                # query_bbox=None,
                # query_feat=None,
                pre_attn_mask,
            )


@ATTENTION.register_module()
class CamRadarCrossAtten(BaseModule):
    def __init__(
        self,
        radar_num_points=4,
        num_points=1,
        num_heads=8,
        num_levels=1,
        dense_radar_dims_in=128,
        embed_dims=256,
        embed_dims_in=sum([128, 256]),
        embed_dims_out=256,
        dropout=0.1,
        im2col_step=64,
        batch_first=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.radar_num_points = radar_num_points
        self.im2col_step = im2col_step

        self.embed_dims = embed_dims
        self.embed_dims_in = embed_dims_in
        self.embed_dims_out = embed_dims_out

        self.rad_sampling_offsets = nn.Linear(
            embed_dims, num_heads * radar_num_points * 2
        )

        self.rad_value_proj = nn.Linear(dense_radar_dims_in, dense_radar_dims_in)
        self.rad_attention_weights = nn.Linear(embed_dims, num_heads * radar_num_points)
        self.rad_output_proj = nn.Linear(dense_radar_dims_in, embed_dims)

        self.embedding_fusion_seq = nn.Sequential(
            nn.Linear(self.embed_dims_in, self.embed_dims_out),
            nn.LayerNorm(self.embed_dims_out),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims_out, self.embed_dims_out),
            nn.LayerNorm(self.embed_dims_out),
        )
        self.img_bev_attention_weights = nn.Linear(embed_dims, self.num_points)
        self.img_bev_sample_out_proj = nn.Linear(embed_dims, embed_dims)
        self.ouput_proj = nn.Linear(embed_dims_in, embed_dims)

        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first

        self.img_bev_feats_sample_pos_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims // 2),
            nn.LayerNorm(embed_dims // 2),
            nn.ReLU(inplace=True),
        )

        # self.img_bev_feat_level_embed = nn.Parameter(1, self.embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""

        constant_init(self.img_bev_attention_weights, val=0.0, bias=0.0)
        xavier_init(self.ouput_proj, distribution="uniform", bias=0.0)

        # device = next(self.parameters()).device
        # thetas = torch.arange(self.num_heads, dtype=torch.float32, device=device) * (
        #     2.0 * math.pi / self.num_heads
        # )

        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (
        #     (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        #     .view(self.num_heads, 1, 1, 2)
        #     .repeat(1, self.num_levels, self.radar_num_points, 1)
        # )

        # for i in range(self.num_points):
        #     grid_init[:, :, i, :] *= i + 1

        constant_init(self.rad_sampling_offsets, 0.0)
        # self.rad_sampling_offsets.bias.data = grid_init[:, 0].reshape(-1)
        constant_init(self.rad_attention_weights, val=0.0, bias=0.0)
        xavier_init(self.rad_value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.rad_output_proj, distribution="uniform", bias=0.0)

        self._is_init = True

    def img_bev_feats_sample(
        self, query, img_bev_feats, reference_points, ref_point_embed=False
    ):
        query = query.permute(1, 0, 2)
        attention_weights = self.img_bev_attention_weights(query)
        attention_weights = attention_weights.sigmoid()
        attention_weights = attention_weights.unsqueeze(-2)

        # img_bev_feats [b c h w]
        reference_points_voxel = (reference_points.sigmoid() - 0.5) * 2
        reference_points_voxel = reference_points_voxel.view(
            -1, 1, 1, *reference_points_voxel.shape[-2:]
        )  # b num_query 3 -> b 1 1 num_query 3

        img_bev_embed = F.grid_sample(
            img_bev_feats,
            reference_points_voxel.reshape(-1, *reference_points_voxel.shape[-3:])[
                ..., :2
            ],
        )  # b embed_dims num_points num_query

        img_bev_embed = img_bev_embed.reshape(
            len(query), -1, img_bev_embed.shape[1], img_bev_embed.shape[-1]
        ).permute(
            0, 3, 2, 1
        )  # b, num_query, embed_dims, num_points

        img_bev_embed = img_bev_embed * attention_weights
        img_bev_embed = img_bev_embed.sum(-1)



        img_bev_embed = self.img_bev_sample_out_proj(img_bev_embed)

        if ref_point_embed:
            outputs_pos_embed = self.img_bev_feats_sample_pos_encoder(reference_points)

            return img_bev_embed + outputs_pos_embed
        else:
            return img_bev_embed

    def simple_img_bev_feats_sample(
        self, query, img_bev_feats, reference_points, ref_point_embed=False
    ):
        query = query.permute(1, 0, 2)

        # img_bev_feats [b c h w]
        # reference_points_voxel = (reference_points.sigmoid() - 0.5) * 2
        reference_points_voxel = reference_points.sigmoid()
        reference_points_voxel = reference_points_voxel.view(
            -1, 1, 1, *reference_points_voxel.shape[-2:]
        )  # b num_query 3 -> b 1 1 num_query 3

        img_bev_embed = F.grid_sample(
            img_bev_feats,
            reference_points_voxel.reshape(-1, *reference_points_voxel.shape[-3:])[
                ..., :2
            ],
        )  # b embed_dims 1 num_query

        img_bev_embed = img_bev_embed.reshape(len(query), -1, query.shape[1]).permute(
            0, 2, 1
        )  # b, embed_dims, num_query

        return img_bev_embed

    def dense_radar_feats_sample(
        self,
        query,
        radar_feats,
        reference_points,
        feats_shapes,
        level_start_index,
        batch_first=False,
    ):
        reference_points = reference_points.sigmoid()
        query = query.permute(1, 0, 2)
        value = radar_feats
        if batch_first:
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (feats_shapes[:, 0] * feats_shapes[:, 1]).sum() == num_value

        value = self.rad_value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.rad_sampling_offsets(query).view(
            bs, num_query, self.num_heads, 1, self.radar_num_points, 2
        )
        attention_weights = self.rad_attention_weights(query).view(
            bs, num_query, self.num_heads, self.radar_num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, 1, self.radar_num_points
        )
        ref_points = reference_points.unsqueeze(2).expand(-1, -1, 1, -1)
        # ref_points = reference_points
        ref_points = ref_points[..., :2]
        if ref_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [feats_shapes[..., 1], feats_shapes[..., 0]], -1
            )

            sampling_locations = (
                ref_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2, but get {reference_points.shape[-1]} instead."
            )
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                feats_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, feats_shapes, sampling_locations, attention_weights
            )

        radar_output = self.rad_output_proj(output)

        return radar_output

    def embedding_fusion(self, embedding_1, embedding_2, style="cat"):
        """embedding fusion block
        Args
            embedding_1 (torch.tensor): [num_len, bs, embed_dims] or permute(1,0,2)
            embedding_2 (torch.tensor): [num_len, bs, embed_dims] or permute(1,0,2)
        """
        if embedding_1 is None:
            return embedding_2

        if embedding_2 is None:
            return embedding_1

        assert len(embedding_1.shape) == 3
        assert len(embedding_2.shape) == 3
        if style == "compact":
            new_embeddings = torch.cat([embedding_1, embedding_2], dim=-1)
            new_embeddings = self.embedding_fusion_seq(new_embeddings)
        elif style == "cat":
            new_embeddings = torch.cat([embedding_1, embedding_2], dim=-1)

        return new_embeddings

    def forward(
        self,
        query,
        key,
        value,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        img_bev_flatten_level_embed=None,
        **kwargs,
    ):
        """
        Args:
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, 3s),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
        """
        assert isinstance(value, list)

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        if value[0] is not None:
            # img_bev_embedding = self.img_bev_feats_sample(
            #     query=query,
            #     img_bev_feats=value[0],
            #     reference_points=reference_points,
            # )

            # img_bev_embed = value[0].flatten(2).permute(
            #     0, 2, 1
            # ) + img_bev_flatten_level_embed.view(1, 1, -1)
            img_bev_embed = value[0]
            img_bev_embedding = self.simple_img_bev_feats_sample(
                query=query,
                img_bev_feats=img_bev_embed,
                reference_points=reference_points,
            )
        else:
            img_bev_embedding = None

        if value[1] is not None:
            dense_radar_embedding = self.dense_radar_feats_sample(
                query=query,
                radar_feats=value[1],
                reference_points=reference_points,
                feats_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        else:
            dense_radar_embedding = None

        outputs = self.embedding_fusion(
            img_bev_embedding, dense_radar_embedding, style="cat"
        )

        outputs = outputs.permute(1, 0, 2)

        outputs = self.ouput_proj(outputs)

        return self.dropout(outputs) + identity

    def _forward(
        self,
        query,
        key,
        value,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        img_bev_flatten_level_embed=None,
        **kwargs,
    ):
        if self.training and query.requires_grad:
            return cp(
                self._forward,
                query,
                key,
                value,
                identity,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                img_bev_flatten_level_embed,
            )
        else:
            return self._forward(
                query,
                key,
                value,
                identity,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                img_bev_flatten_level_embed,
            )


@FEEDFORWARD_NETWORK.register_module()
class FFNMoe(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    @deprecated_api_warning(
        {"dropout": "ffn_drop", "add_residual": "add_identity"}, cls_name="FFN"
    )
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        num_experts=8,
        filter_experts=2,
        experts_ffn_cfg=None,
        act_cfg_experts=dict(type="ReLU", inplace=True),
        **kwargs,
    ):
        super(FFNMoe, self).__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        self.num_experts = num_experts
        self.filter_experts = filter_experts
        self.experts_ffn_cfg = experts_ffn_cfg
        self.act_cfg_experts = act_cfg_experts

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels

        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        )
        self.add_identity = add_identity

        self._init_layers()

    def _init_layers(self):
        self.experts_gated = nn.Linear(self.embed_dims, self.filter_experts)

        experts_layers = nn.ModuleList()
        for _ in range(self.num_experts):
            experts_layers.append()

    @deprecated_api_warning({"residual": "identity"}, cls_name="FFN")
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
