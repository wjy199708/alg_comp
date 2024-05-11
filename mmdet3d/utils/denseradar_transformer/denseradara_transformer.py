import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_

# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import (
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    build_transformer_layer_sequence,
)

# from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
from .decoder.decoder import CamRadarCrossAtten
from typing import Optional


@TRANSFORMER.register_module()
class DenseRadarPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        dn_enabled=False,
        encoder=None,
        decoder=None,
        embed_dims=256,
        **kwargs,
    ):
        super(DenseRadarPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = (
            build_transformer_layer_sequence(encoder) if encoder is not None else None
        )
        self.decoder_config = decoder
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.dn_enabled = dn_enabled

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

        if self.dn_enabled:
            self.query_pos_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
        self.img_bev_flatten_level_embed = nn.Parameter(
            torch.Tensor(1, self.embed_dims)
        )

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(
                m, CamRadarCrossAtten
            ):
                m.init_weights()
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        normal_(self.img_bev_flatten_level_embed)

    @auto_fp16(
        apply_to=(
            "mlvl_feats",
            "bev_queries",
            "object_query_embed",
            "prev_bev",
            "bev_pos",
            "dense_radar_feats",
            "img_bev_feats",
        )
    )
    def forward(
        self,
        img_bev_feats: Optional[torch.tensor],
        dense_radar_feats: Optional[torch.tensor],
        object_query_embed,
        mlvl_feats,
        feats_shapes,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformer`.
        Args:
            img_bev_feats (torch.tensor): [bs embed_dims h w].
            dense_radar_feats (torch.tensor): New dense features after fusing radar features to bev query [num_len, bs, embed_dims//2].
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bs = mlvl_feats[0].size(0)

        if self.dn_enabled:
            query_pos = self.query_pos_encoder(object_query_embed[..., :3])

            reference_points = kwargs["query_bbox"][..., :3]
            query = object_query_embed
            init_reference_out = reference_points
        else:
            query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)

            # reference_points = self.reference_points(query_pos).sigmoid()
            reference_points = self.reference_points(query_pos)  # B Q 3
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        # print(query.shape, "=[][][===]", query_pos.shape)  # 1240 b 256

        if (
            self.decoder_config.transformerlayers.attn_cfgs[1]["type"]
            == "CustomMSDeformableAttention"
        ):
            # At this time, dense_radar_feats only contains bev_query
            # value = dense_radar_feats
            value = img_bev_feats.flatten(2).permute(2, 0, 1)
        elif img_bev_feats is not None:
            value = [img_bev_feats, dense_radar_feats]

        if type(value) is not list and len(value.shape) == 4:
            value = value.flatten(2).permute(2, 0, 1)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=feats_shapes,
            reg_branches=reg_branches,
            level_start_index=torch.tensor([0], device=query.device),
            img_bev_flatten_level_embed=self.img_bev_flatten_level_embed,
            **kwargs,
        )

        inter_references_out = inter_references
        # inter_references_out = inter_references.sigmoid()

        # print(inter_references_out)

        return inter_states, init_reference_out, inter_references_out
