import copy
import warnings
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmcv.cnn.bricks.transformer import (
    TransformerLayerSequence,
    build_positional_encoding,
    build_attention,
)
from mmdet3d.utils.point_generator import MlvlPointGenerator
from ..custom_base_transformer_layer import MyCustomBaseTransformerLayer
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn import xavier_init, ConvModule, caffe2_xavier_init, normal_init
from mmdet.models.builder import BACKBONES
import numpy as np
import torch
import cv2 as cv
from mmcv.utils import TORCH_VERSION, digit_version, ext_loader
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint as cp

# from .checkpoint import checkpoint as cp

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        *args,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        dataset_type="nuscenes",
        **kwargs,
    ):
        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device="cuda",
        dtype=torch.float,
    ):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5,
                    Z - 0.5,
                    num_points_in_pillar,
                    dtype=dtype,
                    device=device,
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)  # D H W 3
            # D 3 H W -> D 3 HW -> D HW 3
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # B D HW  3
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=("reference_points", "img_metas"))
    def point_sampling(self, reference_points, pc_range, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )  # D B N n_query 4 1

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )  # D B N n_query 4 4

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(
            -1
        )  # D B N n_query 4
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        # D B N n_query 4 -> N B n_query D 4
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)  # ncams B bev_query D

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs["img_metas"]
        )

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            # temporal self attention
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DenseRadarBEVEncoder(TransformerLayerSequence):
    def __init__(
        self,
        data_config=None,
        bev_shape=(200, 200),
        downsample=8,
        is_multi_scale=False,
        img_feats_idx=0,
        bev_embedding=None,
        in_channels=256,
        num_views=6,
        return_intermediate=False,
        pc_range=None,
        extra_bev_pos=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.bev_embedding = bev_embedding
        self.data_config = data_config
        self.bev_shape = bev_shape
        self.downsample = downsample
        self.num_views = num_views
        self.pc_range = pc_range
        self.is_multi_scale = is_multi_scale
        self.img_feats_idx = img_feats_idx
        self.extra_bev_pos = extra_bev_pos

        self.return_intermediate = return_intermediate

        # if bev_embedding.bev_pos_type.bev_pos_encode in ["Learn", "Sin"]:
        #     self.bev_pos_embedding = build_positional_encoding(
        #         bev_embedding.bev_pos_type
        #     )
        # else:
        #     self.bev_pos_embedding, _ = self.get_query_and_key_pos_embedding()

        # self.register_buffer("")

        self.init_layers()

    def init_layers(self):
        self.extra_bev_pos_embedding = (
            build_positional_encoding(self.extra_bev_pos)
            if self.extra_bev_pos is not None
            else None
        )

        if self.bev_embedding.type == "LearningBEVEmbedding":
            self.bev_query = nn.Embedding(
                self.bev_shape[0] * self.bev_shape[1], embedding_dim=256
            )
        elif self.bev_embedding.type == "MatualBEVEmbedding":
            self.bev_query = nn.Parameter(
                self.bev_embedding.sigma
                * torch.randn(
                    self.bev_embedding.embed_dim,
                    self.bev_shape[0] // self.downsample,
                    self.bev_shape[1] // self.downsample,
                )
            )
        else:
            raise NotImplemented

        self.query_pos_encoder = nn.Conv2d(
            2, out_channels=self.in_channels, kernel_size=1, bias=False
        )

        self.key_pos_embed_encoder_1 = nn.Conv2d(4, self.in_channels, 1, bias=False)
        # self.key_pos_embed_encoder_2 = nn.Conv2d(4, self.in_channels, 1, bias=False)
        self.key_pos_embed_encoder_2 = nn.Conv2d(
            3, self.in_channels, 1, bias=False
        )  # for version2
        self.key_feats_embedding_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
        )

        self.value_embedding_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
        )

    def create_range_indices(
        self,
        range_indices_hw,
    ):
        range_indices_h, range_indices_w = range_indices_hw

        xs = torch.linspace(0, 1, range_indices_w)
        ys = torch.linspace(0, 1, range_indices_h)

        indices = torch.stack(torch.meshgrid((xs, ys)), 0)  # 2 w h
        indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 w h
        indices = indices[None]  # 1 3 w h

        return indices

    def create_bev_grid(
        self, bev_shape, downsample, pc_range=[-50, -50, -5, 50, 50, 3]
    ):
        bev_points = self.create_range_indices(
            np.array(self.bev_shape) // downsample
        ).squeeze(0)

        pc_range = np.array(pc_range)
        center = (pc_range[-3:] - pc_range[:3])[:2]

        bev_points[0] = self.bev_shape[1] * bev_points[0]
        bev_points[1] = self.bev_shape[0] * bev_points[1]

        h = bev_shape[0]
        w = bev_shape[1]

        sh = h / center[0]
        sw = w / center[1]

        trans_matrix = torch.FloatTensor(
            [
                [0.0, -sw, w / 2.0],
                [-sh, 0.0, h * 0.0 + h / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ).inverse()

        bev_points = (
            (trans_matrix @ bev_points.flatten(1))
            .reshape(-1, h // downsample, w // downsample)[:2]
            .unsqueeze(0)
        )  # 1 2 h w

        return bev_points

    def get_extrin_intrin_cams_params(self, img_inputs, img_metas):
        _, _, _, intrins, _, _, _ = img_inputs

        lidar2cams = []
        for lidar2cam in img_metas:
            lidar2cams.append(lidar2cam["lidar2cams"])
        lidar2cams = torch.stack(lidar2cams)

        return intrins, lidar2cams

    def get_query_and_key_pos_embedding(
        self,
        extrins: torch.tensor,
        intrins: torch.tensor,
        img_feats_shape: tuple,
        device,
        post_rots: torch.tensor = None,
        post_trans: torch.tensor = None,
    ):
        """get_query_and_key_pos_embedding_v2

        Args:
            extrins (torch.tensor): lidar2cams calibration.
            intrins (torch.tensor): cams intrins calibration, make cooridination to pixels.
            img_feats_shape (tuple): img_feats_shape
            post_rots (torch.tensor): cams augmentation execuation.
            post_trans (torch.tensor):  cams augmentation execuation.
            device (_type_): default is GPU device.

        Returns:
            query_pos_embedding (torch.tensor): [bn, c, h, w]
            key_pos_embedding (torch.tensor): [bn, c, h, w]
        """
        B, N, _, _ = intrins.shape
        extrins_inverse = torch.inverse(extrins).to(device)
        cameras_params_embedding = extrins_inverse[..., -1:]
        cameras_params_embedding = rearrange(
            cameras_params_embedding, "b n ... -> (b n) ..."
        ).unsqueeze(-1)
        cameras_params_embedding = self.key_pos_embed_encoder_1(
            cameras_params_embedding
        )

        img_pixel_points = self.create_range_indices(img_feats_shape).unsqueeze(
            0
        )  # 1 1 3 h w
        img_pixel_points[:, :, 0] *= self.data_config["input_size"][1]
        img_pixel_points[:, :, 1] *= self.data_config["input_size"][0]
        img_pixel_points = img_pixel_points.permute(0, 1, 2, 4, 3)
        _, _, _, H, W = img_pixel_points.shape

        # parts added in version 2
        img_pixel_points = (
            rearrange(img_pixel_points, "... h w -> ... (h w)")
            .permute(0, 1, 3, 2)
            .repeat(B, N, 1, 1)
        ).to(
            intrins.device
        )  # 1 1 hw 3

        if post_trans is not None:
            img_pixel_points -= post_trans.view(B, N, 1, 3)

        if post_rots is not None:
            # b ncams 1 3 3 * b ncams hw 3 1 ->  b ncams hw 3
            img_pixel_points = (
                torch.inverse(post_rots)
                .view(B, N, 1, 3, 3)
                .matmul(img_pixel_points.unsqueeze(-1))
            ).squeeze(-1)

        img_pixel_points = img_pixel_points.permute(0, 1, 3, 2)

        # pixel to camera coordinate system transformation.
        # b ncams 3 3 * b ncams 3 hw
        cams_points = torch.inverse(intrins) @ img_pixel_points  # b n 3 hw
        cams_points = F.pad(
            cams_points, (0, 0, 0, 1, 0, 0, 0, 0), value=1
        )  #  b n 4 hw11
        cams_points = extrins_inverse @ cams_points
        cams_points = rearrange(cams_points, "b n d (h w) -> (b n) d h w", h=H, w=W)
        cams_points = self.key_pos_embed_encoder_2(cams_points)

        key_pos_embedding = cams_points - cameras_params_embedding
        key_pos_embedding = key_pos_embedding / (
            key_pos_embedding.norm(dim=1, keepdim=True) + 1e-7
        )

        # generate bev embedding token features
        bev_points = self.create_bev_grid(
            bev_shape=self.bev_shape, downsample=self.downsample
        ).to(intrins.device)
        bev_points = self.query_pos_encoder(bev_points)
        query_pos_embedding = bev_points - cameras_params_embedding
        query_pos_embedding = query_pos_embedding / (
            query_pos_embedding.norm(dim=1, keepdim=True) + 1e-7
        )

        return query_pos_embedding, key_pos_embedding

    def get_query_and_key_pos_embedding_v2(
        self,
        extrins: torch.tensor,
        intrins: torch.tensor,
        img_feats_shape: tuple,
        device,
        post_rots: torch.tensor = None,
        post_trans: torch.tensor = None,
        rots: torch.tensor = None,
        trans: torch.tensor = None,
    ):
        """get_query_and_key_pos_embedding_v2

        Args:
            extrins (torch.tensor): lidar2cams calibration.
            intrins (torch.tensor): cams intrins calibration, make cooridination to pixels.
            img_feats_shape (tuple): img_feats_shape
            post_rots (torch.tensor): cams augmentation execuation.
            post_trans (torch.tensor):  cams augmentation execuation.
            device (_type_): default is GPU device.

        Returns:
            query_pos_embedding (torch.tensor): [bn, c, h, w]
            key_pos_embedding (torch.tensor): [bn, c, h, w]
        """
        B, N, _, _ = intrins.shape
        extrins_inverse = torch.inverse(extrins).to(device)
        cameras_params_embedding = extrins_inverse[..., -1:]
        cameras_params_embedding = rearrange(
            cameras_params_embedding, "b n ... -> (b n) ..."
        ).unsqueeze(-1)
        cameras_params_embedding = self.key_pos_embed_encoder_1(
            cameras_params_embedding
        )

        img_pixel_points = self.create_range_indices(img_feats_shape).unsqueeze(
            0
        )  # 1 1 3 h w
        img_pixel_points[:, :, 0] *= self.data_config["input_size"][1]
        img_pixel_points[:, :, 1] *= self.data_config["input_size"][0]
        img_pixel_points = img_pixel_points.permute(0, 1, 2, 4, 3)
        _, _, _, H, W = img_pixel_points.shape

        # parts added in version 2
        img_pixel_points = (
            rearrange(img_pixel_points, "... h w -> ... (h w)")
            .permute(0, 1, 3, 2)
            .repeat(B, N, 1, 1)
        ).to(
            intrins.device
        )  # 1 1 hw 3

        if post_trans is not None:
            img_pixel_points -= post_trans.view(B, N, 1, 3)

        if post_rots is not None:
            # b ncams 1 3 3 * b ncams hw 3 1 ->  b ncams hw 3
            img_pixel_points = (
                torch.inverse(post_rots)
                .view(B, N, 1, 3, 3)
                .matmul(img_pixel_points.unsqueeze(-1))
            ).squeeze(-1)

        # pixel to camera coordinate system transformation.
        # b ncams 1 3 3 * b ncams hw 3 1
        combine = rots.matmul(torch.inverse(intrins))
        cams_points = (
            combine.view(B, N, 1, 3, 3)
            .matmul(img_pixel_points.unsqueeze(-1))
            .squeeze(-1)
        )
        cams_points += trans.view(B, N, 1, 3)  # b ncams hw 3

        cams_points = rearrange(cams_points, "b n (h w) d -> (b n) d h w", h=H, w=W)
        cams_points = self.key_pos_embed_encoder_2(cams_points)

        key_pos_embedding = cams_points - cameras_params_embedding
        key_pos_embedding = key_pos_embedding / (
            key_pos_embedding.norm(dim=1, keepdim=True) + 1e-7
        )

        # generate bev embedding token features
        bev_points = self.create_bev_grid(
            bev_shape=self.bev_shape, downsample=self.downsample
        ).to(intrins.device)
        bev_points = self.query_pos_encoder(bev_points)
        query_pos_embedding = bev_points - cameras_params_embedding
        query_pos_embedding = query_pos_embedding / (
            query_pos_embedding.norm(dim=1, keepdim=True) + 1e-7
        )

        return query_pos_embedding, key_pos_embedding

    def get_value_embedding(self, img_feats):
        assert len(img_feats.shape) == 5
        B, _, _, _, _ = img_feats.shape
        img_feats_encoder = self.value_embedding_encoder(
            rearrange(img_feats, "b n c h w -> (b n) c h w")
        )

        return rearrange(
            img_feats_encoder, "(b n) c h w -> b n c h w", b=B, n=self.num_views
        )

    def forward(
        self,
        mlvls_cams_feats=None,
        img_inputs=None,
        img_meats=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            mlvls_cams_feats: Single layer image features, multi-scale reserved.
        """
        _, rots, trans, _, post_rots, post_trans, _ = img_inputs

        if not self.is_multi_scale:
            mlvls_cams_feats = mlvls_cams_feats[self.img_feats_idx]

        B, ncams, C, H, W = mlvls_cams_feats.shape
        img_feats_shape = (H, W)

        bev_query = self.bev_query.unsqueeze(0).repeat(B, 1, 1, 1)  # b emb_dim h w
        _, _, q_h, q_w = bev_query.shape

        key = self.key_feats_embedding_encoder(
            mlvls_cams_feats.view(-1, C, H, W)
        ).reshape(B, ncams, C, H, W)

        value = self.get_value_embedding(mlvls_cams_feats)

        intrins, extrins = self.get_extrin_intrin_cams_params(
            img_inputs=img_inputs, img_metas=img_meats
        )

        query_pos, key_pos = self.get_query_and_key_pos_embedding_v2(
            extrins,
            intrins,
            img_feats_shape,
            device=mlvls_cams_feats.device,
            post_rots=post_rots,
            post_trans=post_trans,
            rots=rots,
            trans=trans,
        )

        if self.extra_bev_pos_embedding is not None:
            extra_bev_pos_mask = torch.zeros(
                B,
                self.extra_bev_pos.row_num_embed,
                self.extra_bev_pos.col_num_embed,
            ).to(bev_query.device)
            extra_bev_pos = self.extra_bev_pos_embedding(extra_bev_pos_mask).to(
                bev_query.device
            )
        else:
            extra_bev_pos = None

        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                query_pos=query_pos,
                extra_query_pos=extra_bev_pos,
                key_pos=key_pos,
                **kwargs,
            )

            bev_query = rearrange(output, "b (h w) d -> b d h w", h=q_h, w=q_w)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@ATTENTION.register_module()
class CrossViewBEVAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        heads=8,
        dim_head=32,
        qkv_bias=True,
        num_views=None,
        norm=nn.LayerNorm,
        batch_first=False,
        **kwargs,
    ):
        super(CrossViewBEVAttention, self).__init__(**kwargs)
        if embed_dims % heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {heads}"
            )

        self.embed_dims = embed_dims
        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head

        self.num_views = num_views

        self.to_q = nn.Sequential(
            norm(embed_dims),
            nn.Linear(embed_dims, heads * dim_head, bias=qkv_bias),
        )
        self.to_k = nn.Sequential(
            norm(embed_dims),
            nn.Linear(embed_dims, heads * dim_head, bias=qkv_bias),
        )
        self.to_v = nn.Sequential(
            norm(embed_dims),
            nn.Linear(embed_dims, heads * dim_head, bias=qkv_bias),
        )

        self.output_proj = nn.Linear(heads * dim_head, embed_dims)
        # self.prenorm = norm(dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        # )
        # self.postnorm = norm(dim)

        self.batch_first = batch_first

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        extra_query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        B, _, H, W = query.shape

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

        if extra_query_pos is not None:
            query = query + extra_query_pos

        if query_pos is not None:
            # query = query + query_pos
            query = query.unsqueeze(1) + rearrange(
                query_pos, "(b n) ... -> b n ...", b=B, n=self.num_views
            )
        if key_pos is not None:
            key = key + rearrange(
                key_pos, "(b n) ... -> b n ...", b=B, n=self.num_views
            )

        assert len(query.shape) == 5 and len(key.shape) == 5 and len(value.shape) == 5

        query = rearrange(
            self.to_q(rearrange(query, "b n c h w -> b n (h w) c")),
            "b n q (h d) -> (b h) n q d",
            h=self.heads,
            d=self.dim_head,
        )
        key = rearrange(
            self.to_k(rearrange(key, "b n c h w -> b n (h w) c")),
            "b n k (h d) -> (b h) n k d",
            h=self.heads,
            d=self.dim_head,
        )
        value = rearrange(
            self.to_v(rearrange(value, "b n c h w -> b (n h w) c")),
            "b v (h d) -> (b h) v d",
            h=self.heads,
            d=self.dim_head,
        )

        attention_weights = self.scale * torch.einsum(
            "b n q d, b n k d -> b n q k", query, key
        )
        attention_weights = rearrange(
            attention_weights, "b n q k -> b q (n k)"
        ).softmax(dim=-1)

        output = torch.einsum("b q v, h v d -> b q d", attention_weights, value)
        output = rearrange(
            output, "(b h) q d -> b q (h d)", h=self.heads, d=self.dim_head
        )
        output = self.output_proj(output) + rearrange(identity, "b d h w -> b (h w) d")

        return output


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DenseRadarBEVEncoder_V2(BEVFormerEncoder):
    def __init__(
        self,
        data_config=None,
        num_feature_levels=4,
        pc_range=None,
        num_points_in_pillar=4,
        num_layers=None,
        transformerlayers=None,
        return_intermediate=False,
        use_cams_embeds=True,
        init_cfg=None,
    ):
        super().__init__(transformerlayers, num_layers, init_cfg)

        self.data_config = data_config
        self.use_cams_embeds = use_cams_embeds
        self.num_feature_levels = num_feature_levels
        self.num_cams = data_config["Ncams"]

        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar

        self.return_intermediate = return_intermediate

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))

    def _init_weights(self):
        pass

    # This function must use fp32!!!
    @force_fp32(apply_to=("reference_points", "img_metas", "img_inputs"))
    def point_sampling(self, reference_points, pc_range, img_metas, img_inputs):
        """Adjust the mapping method and align the mapping point position of the LSS BEV feature.

        Args:
            reference_points (_type_): _description_
            pc_range (_type_): _description_
            img_metas (_type_): _description_
            img_inputs (list): cams_imgs rots trans intrins post_rots post_trans projected_depth.
        Returns:
            _type_: _description_
        """

        lidar2imgs = []
        for img_meta in img_metas:
            lidar2imgs.append(img_meta["lidar2imgs"])
        lidar2img = torch.stack(lidar2imgs).to(reference_points.device)
        # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        # ncams B bev_query D
        # B D HW 3 -> D B HW 3
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )  # D B ncams n_query 4 1

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )  # D B ncams n_query 4 4

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(
            -1
        )  # D B ncams n_query 4
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps  # D B ncams bev_qeuery 1

        # Image enhancement for demapping lss.
        _, _, _, _, post_rots, post_trans, _ = img_inputs
        post_rots = post_rots.view(1, B, num_cam, 1, 3, 3).repeat(
            D, 1, 1, num_query, 1, 1
        )[..., :2, :2]

        reference_points_cam[..., 0:2] = post_rots.matmul(
            reference_points_cam[..., 0:2].unsqueeze(-1)
        ).squeeze(
            -1
        )  # D B ncams bev_query 2

        reference_points_cam[..., 0:2] += post_trans.view(1, B, num_cam, 1, 3).repeat(
            D, 1, 1, num_query, 1
        )[..., 0:2]

        # Mappinng point normalization
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= self.data_config["input_size"][1]
        reference_points_cam[..., 1] /= self.data_config["input_size"][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        # D B N n_query 4 -> N B n_query D 4
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(
            -1
        )  # D B ncams bev_query -> ncams B bev_query D 1 -> ncams B bev_query D

        return reference_points_cam, bev_mask

    def forward(
        self,
        bev_query,
        mlvl_feats: list,
        query_pos=None,
        img_inputs=None,
        img_metas=None,
        bev_h=None,
        bev_w=None,
        prev_bev=None,
    ):
        """DenseRadarBEVEncoder_V2 forward

        Args:
            bev_query (_type_): bev_query
            key (optional): mlvls_imgs_feats.
            value (_type_): mlvls_imgs_feats.
            query_pos (_type_, optional): _description_. Defaults to None.
            key_pos (_type_, optional): _description_. Defaults to None.
            attn_masks (_type_, optional): _description_. Defaults to None.
            query_key_padding_mask (_type_, optional): _description_. Defaults to None.
            key_padding_mask (_type_, optional): _description_. Defaults to None.
            bev_h (_type_, optional): _description_. Defaults to None.
            bev_w (_type_, optional): _description_. Defaults to None.
            prev_bev (_type_, optional): _description_. Defaults to None.

        Returns:
            outputs: bev features learned from multiple images
        """

        B = mlvl_feats[0].shape[0]

        bev_query = bev_query.unsqueeze(0).repeat(B, 1, 1)

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=B,
            device=bev_query.device,
            dtype=bev_query.dtype,
        )  # [bs, num_d_pillar, num_query, 3]

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas, img_inputs
        )

        # bs multi-levels imgs feats
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # b ncams c h w -> b ncams c hw -> ncams b hw c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  # n b hw*4 c

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_query.device
        )

        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                feat_flatten,
                feat_flatten,
                bev_pos=query_pos,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

    def _forward(
        self,
        bev_query,
        mlvl_feats,
        query_pos=None,
        img_inputs=None,
        bev_h=None,
        bev_w=None,
        prev_bev=None,
        **kwargs,
    ):
        img_metas = kwargs["img_metas"]
        if self.training and bev_query.requires_grad:
            return cp(
                self._forward,
                bev_query,
                mlvl_feats,
                query_pos,
                img_inputs,
                img_metas,
                bev_h,
                bev_w,
                prev_bev,
            )
        else:
            return self._forward(
                bev_query,
                mlvl_feats,
                query_pos,
                img_inputs=img_inputs,
                img_metas=kwargs["img_metas"],
                bev_h=bev_h,
                bev_w=bev_w,
                prev_bev=prev_bev,
            )

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Deformable_Sampling_SelfEncoder(TransformerLayerSequence):
    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_outs=3,
        num_layers=6,
        embed_dims=256,
        conv_cfg=dict(type="Conv3d"),
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=dict(type="ReLU"),
        transformerlayers=None,
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        ret_ms=False,
        init_cfg=None,
        **kwargs,
    ):

        super().__init__(
            transformerlayers=transformerlayers,
            num_layers=num_layers,
            init_cfg=init_cfg,
        )

        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = transformerlayers.attn_cfgs[0].num_levels
        assert self.num_encoder_levels >= 1

        # build input conv for channel adapation
        # from top to down (low to high resolution)
        input_conv_list = []
        for i in range(
            self.num_input_levels - 1,
            self.num_input_levels - self.num_encoder_levels - 1,
            -1,
        ):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=None,
                bias=True,
            )
            input_conv_list.append(input_conv)

        self.input_convs = ModuleList(input_conv_list)
        # self.encoder = build_transformer_layer_sequence(encoder)
        self.postional_encoding = build_positional_encoding(positional_encoding)
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=None,
            )

            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
            )

            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = nn.Conv3d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)
        self.ret_ms = ret_ms

    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv, gain=1, bias=0, distribution="uniform"
            )

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # for layer in self.layers:
        #     for attn in layer.attentions:
        #         if isinstance(attn, POP_Deformable_3D_Attention):
        #             attn.init_weights()

    def sparse_pts_feats_to_dense(self, pts_feats: list) -> list:
        ret_lists = []
        for i, feat in enumerate(pts_feats):
            ret_lists.append(feat.dense())

        return ret_lists

    def forward(self, feats: list, **kwagrs):
        feats = self.sparse_pts_feats_to_dense(feats)
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []

        for i in range(self.num_encoder_levels):
            # 从最后一层输入开始
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]

            feat_projected = self.input_convs[i](feat)
            X, Y, Z = feat.shape[-3:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size,) + feat.shape[-3:], dtype=torch.bool
            )
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]

            level_pos_embed = level_embed.view(1, -1, 1, 1, 1) + pos_embed

            # (h_i * w_i * d_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-3:], level_idx, device=feat.device
            )

            # normalize points to [0, 1]
            factor = feat.new_tensor([[Z, Y, X]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, x_i, y_i, z_i) -> (x_i * y_i * z_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-3:])
            reference_points_list.append(reference_points)

        # shape (batch_size, total_num_query),
        # total_num_query=sum([., x_i * y_i * z_i, .])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=0)

        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device
        )
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1
        )
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2)
        )

        # shape (num_total_query, batch_size, c)
        for layer in self.layers:
            memory = layer(
                query=encoder_inputs,
                key=None,
                value=None,
                query_pos=level_positional_encodings,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                query_key_padding_mask=padding_masks,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_radios=valid_radios,
            )

        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        if self.ret_ms:

            # from low resolution to high resolution
            num_query_per_level = [e[0] * e[1] * e[2] for e in spatial_shapes]
            outs = torch.split(memory, num_query_per_level, dim=-1)
            outs = [
                x.reshape(
                    batch_size,
                    -1,
                    spatial_shapes[i][0],
                    spatial_shapes[i][1],
                    spatial_shapes[i][2],
                )
                for i, x in enumerate(outs)
            ]

            # build FPN path
            indice1 = [
                i
                for i in range(
                    self.num_input_levels - self.num_encoder_levels - 1, -1, -1
                )
            ]
            indice2 = [
                i for i in range(0, self.num_input_levels - self.num_encoder_levels)
            ]
            for i, j in zip(indice1, indice2):
                x = feats[j]
                cur_feat = self.lateral_convs[i](x)

                y = cur_feat + F.interpolate(
                    outs[-1],
                    size=cur_feat.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

                y = self.output_convs[i](y)
                outs.append(y)

            outs[-1] = self.mask_feature(outs[-1])

        else:

            return memory
