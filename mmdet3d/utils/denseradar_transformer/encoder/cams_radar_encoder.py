import copy
import warnings
from mmcv.cnn.bricks.transformer import (
    TRANSFORMER_LAYER_SEQUENCE,
    ATTENTION,
    build_activation_layer,
    TransformerLayerSequence,
    build_attention,
)
from mmcv.cnn import xavier_init
from mmcv.runner import force_fp32, auto_fp16
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import BACKBONES
from mmdet3d.models.necks.fpn import UpSampleBlock
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential


class Query_Decoder_UpSample(BaseModule):
    def __init__(self, dim, blocks, residual=True, factor=2, init_cfg=None):
        super().__init__(init_cfg)
        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = UpSampleBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CamsRadarEncoder(TransformerLayerSequence):
    def __init__(
        self,
        return_intermediate=False,
        only_ret_bev_query=False,
        bev_query_decoder=False,
        bev_query_decoder_layers=[256, 256, 64],
        bev_shape=None,
        bev_downsample=8,
        upsample_residual=True,
        upsample_factor=2,
        query_channel=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate
        self.only_ret_bev_query = only_ret_bev_query
        self.bev_query_decoder = bev_query_decoder
        self.bev_query_decoder_layers = bev_query_decoder_layers
        self.bev_shape = bev_shape
        self.bev_downsample = bev_downsample

        self.query_bev_upsample = Query_Decoder_UpSample(
            dim=query_channel,
            blocks=bev_query_decoder_layers,
            residual=upsample_residual,
            factor=upsample_factor,
        )

    def unified_shape_size(self, bev_query, bevshape):
        RH, RW = bevshape
        bev_query = F.interpolate(bev_query, (RH, RW), mode="bilinear")

        return bev_query

    def forward(self, query_bev, radar_feats, *args, **kwargs):
        bev_query = None
        key = None
        value = None
        bev_pos = None
        key_pos = None

        query_bev = (
            rearrange(
                query_bev,
                "b (h w) d -> b d h w",
                h=(self.bev_shape[0] // self.bev_downsample),
                w=(self.bev_shape[1] // self.bev_downsample),
            )
            if len(query_bev.shape) == 3
            else query_bev
        )

        _, _, q_H, q_w = query_bev.shape

        if q_H != self.bev_shape[0]:
            query_bev = self.query_bev_upsample(query_bev)

            query_bev = self.unified_shape_size(
                bev_query=query_bev, bevshape=self.bev_shape
            )

        if self.only_ret_bev_query:
            return query_bev

        new_query = torch.cat([query_bev, radar_feats], dim=1)

        new_query = rearrange(new_query, "b d h w -> (h w) b d")

        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                new_query,
                new_query,
                new_query,
                *args,
                bev_pos=bev_pos,
                key_pos=key_pos,
                **kwargs,
            )

            new_query = output

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@BACKBONES.register_module()
class CamsRadarEncoderConv(BaseModule):
    def __init__(
        self,
        only_ret_bev_query=False,
        bev_shape=None,
        bev_downsample=8,
        query_channel=256,
        bev_query_decoder_layers=[256, 256, 64],
        upsample_residual=True,
        upsample_factor=2,
        fusion_in_channel=128,
        fusion_out_channel=128,
        init_cfg=None,
    ):
        """The camera bev upsampled query features and radar features are directly fused using convolution.

        Args:
            only_ret_bev_query (bool, optional): _description_. Defaults to False.
            bev_shape (_type_, optional): _description_. Defaults to None.
            bev_downsample (int, optional): _description_. Defaults to 8.
            query_channel (int, optional): _description_. Defaults to 256.
            bev_query_decoder_layers (list, optional): _description_. Defaults to [256, 256, 64].
            upsample_residual (bool, optional): _description_. Defaults to True.
            upsample_factor (int, optional): _description_. Defaults to 2.
            fusion_in_channel (int, optional): _description_. Defaults to 128.
            fusion_out_channel (int, optional): _description_. Defaults to 128.
            init_cfg (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(init_cfg)

        self.only_ret_bev_query = only_ret_bev_query
        self.bev_shape = bev_shape
        self.bev_downsample = bev_downsample

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channel, fusion_out_channel, kernel_size=1),
            nn.BatchNorm2d(fusion_out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_out_channel, fusion_out_channel, kernel_size=1),
            nn.BatchNorm2d(fusion_out_channel),
        )

        self.query_bev_upsample = Query_Decoder_UpSample(
            dim=query_channel,
            blocks=bev_query_decoder_layers,
            residual=upsample_residual,
            factor=upsample_factor,
        )

    def unified_shape_size(self, bev_query, radar_feats):
        _, _, RH, RW = radar_feats.shape
        bev_query = F.interpolate(bev_query, (RH, RW), mode="bilinear")

        return bev_query

    def forward(
        self,
        query_bev,
        radar_feats,
    ):
        B, r_ch, r_H, r_W = radar_feats.shape
        query_bev = (
            rearrange(
                query_bev,
                "b (h w) d -> b d h w",
                h=(self.bev_shape[0] // self.bev_downsample),
                w=(self.bev_shape[1] // self.bev_downsample),
            )
            if len(query_bev.shape) == 3
            else query_bev
        )
        _, _, q_H, q_w = query_bev.shape

        if r_H != q_H:
            query_bev = self.query_bev_upsample(query_bev)

            query_bev = self.unified_shape_size(
                bev_query=query_bev, radar_feats=radar_feats
            )

        if self.only_ret_bev_query:
            return query_bev

        new_query = torch.cat([query_bev, radar_feats], dim=1)

        new_query = self.fusion(new_query)

        new_query = rearrange(new_query, "b d h w -> (h w) b d")

        return new_query
