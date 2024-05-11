import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import multi_apply, multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models import HEADS, build_backbone
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.utils import normalize_bbox, encode_bbox
from mmcv.cnn.bricks.transformer import (
    build_positional_encoding,
    build_transformer_layer_sequence,
)
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import force_fp32, auto_fp16

import numpy as np
import math
import cv2 as cv
from termcolor import colored
from typing import Optional


@HEADS.register_module()
class DenseRadarHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        transformer=None,
        bev_query_encoder=None,
        dense_radar_encoder=None,
        multi_bev_fusion=None,
        decoder=None,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ]

        self.bev_query_encoder = bev_query_encoder
        self.dense_radar_encoder = dense_radar_encoder
        self.multi_bev_fusion = multi_bev_fusion

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        super(DenseRadarHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False,
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_query_embed = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims
            )
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

        if self.bev_query_encoder is not None:
            self.bev_query_encoder = build_transformer_layer_sequence(
                self.bev_query_encoder
            )
        else:
            self.bev_query_encoder = None

        if self.dense_radar_encoder is not None:
            if "Conv" in self.dense_radar_encoder.type:
                self.dense_radar_encoder = build_backbone(self.dense_radar_encoder)
            else:
                self.dense_radar_encoder = build_transformer_layer_sequence(
                    self.dense_radar_encoder
                )
        else:
            self.dense_radar_encoder = None

        if self.multi_bev_fusion is not None:
            self.multi_bev_fusion_encoder = build_transformer_layer_sequence(
                self.multi_bev_fusion
            )
        else:
            self.multi_bev_fusion_encoder = None

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
                
    
    def ref_point_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(
            pts.shape[0], self.map_num_vec, self.map_num_pts_per_vec, 2
        )
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == "minmax":
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape


    @auto_fp16(apply_to=("mlvl_feats"))
    def forward(
        self,
        img_inputs,
        mlvl_feats,
        img_metas,
        img_bev: Optional[torch.tensor],
        radar_feats: Optional[torch.tensor],
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        B, num_cam, _, _, _ = mlvl_feats[0].shape

        dtype = mlvl_feats[0].dtype

        object_query_embeds = self.query_embedding.weight.to(dtype)
        # bev_queries = self.bev_embedding.weight.to(dtype)
        # bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # Multi-scale bev query for BEV feas generation.
        bev_query_embed = self.bev_query_embed.weight.to(dtype)
        bev_mask = torch.zeros(
            (B, self.bev_h, self.bev_w), device=bev_query_embed.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if self.bev_query_encoder is not None:
            outputs_bev_query = self.bev_query_encoder(
                bev_query_embed,
                mlvl_feats,
                bev_pos,
                img_inputs=img_inputs,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                img_metas=img_metas,
            )  # b bev_shape//downsample embed_dims
        else:
            outputs_bev_query = None

        if self.dense_radar_encoder is not None:
            outputs_bev_query = self.dense_radar_encoder(
                outputs_bev_query, radar_feats
            )  # (128 128) b 128
            if len(outputs_bev_query.shape) == 4:
                outputs_bev_query = outputs_bev_query.reshape(
                    *outputs_bev_query.shape[:2], -1
                ).permute(2, 0, 1)

        if self.multi_bev_fusion is not None:
            dense_radar_bev = self.multi_bev_fusion_encoder(
                query=img_bev,
                key=radar_feats,
                value=None,
            )

        # object query deocder transformer
        outputs = self.transformer(
            img_bev,
            outputs_bev_query,
            object_query_embeds,
            mlvl_feats,
            feats_shapes=torch.as_tensor(
                [(self.bev_h, self.bev_w)],
                dtype=torch.long,
                device=object_query_embeds.device,
            ),
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        hs, init_reference, inter_references = outputs

        hs = hs.permute(
            0, 2, 1, 3
        )  # levels num_query bs embed_dims -> levels bs num_query embed_dims

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
                
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs

    def _get_target_single(
        self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_bboxes_ignore=None
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
        img_metas=None,
    ):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return ret_list


@HEADS.register_module()
class DenseRadarHead_BEVQuery(DenseRadarHead):
    def forward(
        self,
        img_inputs,
        mlvl_feats,
        img_metas,
        img_bev,
        radar_feats,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        B, num_cam, _, _, _ = mlvl_feats[0].shape

        dtype = mlvl_feats[0].dtype

        object_query_embeds = self.query_embedding.weight.to(dtype)
        # bev_queries = self.bev_embedding.weight.to(dtype)
        # bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outputs_bev_query = self.bev_query_encoder(
            mlvl_feats, img_inputs, img_metas
        )  # b bev_shape//downsample embed_dims

        if self.dense_radar_encoder is not None:
            outputs_bev_query = self.dense_radar_encoder(
                outputs_bev_query, radar_feats
            )  # (128 128) b 128
            if len(outputs_bev_query.shape) == 4:
                outputs_bev_query = outputs_bev_query.reshape(
                    *outputs_bev_query.shape[:2], -1
                ).permute(2, 0, 1)

        if self.multi_bev_fusion is not None:
            dense_radar_bev = self.multi_bev_fusion_encoder(
                query=img_bev,
                key=radar_feats,
                value=None,
            )

        # object query deocder transformer
        outputs = self.transformer(
            img_bev,
            outputs_bev_query,
            object_query_embeds,
            mlvl_feats,
            feats_shapes=torch.as_tensor(
                [(self.bev_h, self.bev_w)],
                dtype=torch.long,
                device=object_query_embeds.device,
            ),
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        hs, init_reference, inter_references = outputs

        hs = hs.permute(
            0, 2, 1, 3
        )  # levels num_query bs embed_dims -> levels bs num_query embed_dims

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs


@HEADS.register_module()
class DenseRadarHead_RadarQuery(DenseRadarHead):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self._init_layers()

    def _init_layers(self):
        self.geo_position_embedding = nn.Sequential(
            nn.Conv1d(2, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1),
        )

        self.geo_fusion_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dims + self.embed_dims // 4, self.embed_dims, 1, 1),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims + self.embed_dims // 4, self.embed_dims, 1, 1),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(),
        )

    def create_2d_grid(self, x, y):
        """create_2d_grid

        Returns:
            coords: [1,L,2]
        """
        coord_x = torch.linspace(0, x - 1, x)
        coord_y = torch.linspace(0, y - 1, y)
        xx, yy = torch.meshgrid(coord_x, coord_y)
        xx = xx + 0, 5
        yy = yy + 0.5

        coords = torch.cat(xx[None], yy[None], dim=0)  # 2 H W
        coords = coords.flatten(1)[None].permute(0, 2, 1)

        return coords

    def radar_query_initial(
        self, radar_feats, fusion_feats=None, query_initial_ops="cat"
    ):
        radar_points = self.create_2d_grid(radar_feats.dim(2), radar_feats.dim(3))
        radar_pos_embed = self.geo_position_embedding(radar_points.permute(0, 2, 1))
        if fusion_feats is not None and query_initial_ops == "cat":
            new_flatten_feats = self.geo_fusion_encoder(
                torch.cat([radar_feats, fusion_feats], dim=1)
            )
            radar_flatten = new_flatten_feats.flatten(2).permute(0, 2, 1)  # B Q C
        else:
            radar_flatten = radar_feats.flatten(2).permute(0, 2, 1)  # B Q C

        return radar_flatten, radar_pos_embed

    def forward(
        self,
        img_inputs,
        mlvl_feats,
        img_metas,
        img_bev,
        radar_feats,
    ):
        radar_query_flatten, radar_pos_embed = self.radar_query_initial(
            radar_feats=radar_feats, fusion_feats=img_bev
        )
        bev_query = None

        # object query deocder transformer
        outputs = self.transformer(
            img_bev,
            bev_query,
            self.query_embedding.weight.to(radar_pos_embed.device),
            mlvl_feats,
            feats_shapes=torch.as_tensor(
                [(self.bev_h, self.bev_w)],
                dtype=torch.long,
                device=radar_pos_embed.device,
            ),
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        hs, init_reference, inter_references = outputs

        hs = hs.permute(
            0, 2, 1, 3
        )  # levels num_query bs embed_dims -> levels bs num_query embed_dims

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs


@HEADS.register_module()
class DN_DenseRadarHead(DenseRadarHead):
    def __init__(
        self,
        query_denoising=True,
        query_denoising_groups=10,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.dn_enabled = query_denoising
        self.dn_group_num = query_denoising_groups
        self.dn_weight = 1.0
        self.dn_bbox_noise_scale = 0.5
        self.dn_label_noise_scale = 0.5

        self.bev_query_embed = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        self.query_embedding = nn.Embedding(
            self.num_query, 10
        )  # (x, y, z, w, l, h, sin, cos, vx, vy)
        self.label_enc = nn.Embedding(
            self.num_classes + 1, self.embed_dims - 1
        )  # DAB-DETR

        nn.init.zeros_(self.query_embedding.weight[:, 2:3])
        nn.init.zeros_(self.query_embedding.weight[:, 8:10])
        nn.init.constant_(self.query_embedding.weight[:, 5:6], 1.5)

        # add for dn-detr, When not using denoising, please comment.
        grid_size = int(math.sqrt(self.num_query))
        assert grid_size * grid_size == self.num_query
        x = y = torch.arange(grid_size)
        # xx, yy = torch.meshgrid(x, y, indexing="ij")  # [0, grid_size - 1]
        xx, yy = torch.meshgrid(x, y)
        xy = torch.cat([xx[..., None], yy[..., None]], dim=-1)
        xy = (xy + 0.5) / grid_size  # [0.5, grid_size - 0.5] / grid_size ~= (0, 1)
        with torch.no_grad():
            self.query_embedding.weight[:, :2] = xy.reshape(-1, 2)  # [Q, 2]

    def forward(self, img_inputs, mlvl_feats, img_metas, img_bev, radar_feats):
        B, _, _, _, _ = mlvl_feats[0].shape
        
        dtype = mlvl_feats[0].dtype

        bev_query_embed = self.bev_query_embed.weight.to(dtype)
        bev_mask = torch.zeros(
            (B, self.bev_h, self.bev_w), device=bev_query_embed.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # bev feature initialization
        outputs_bev_query = self.bev_query_encoder(
            bev_query_embed,
            mlvl_feats,
            bev_pos,
            img_inputs=img_inputs,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            img_metas=img_metas,
        )  # b bev_shape//downsample embed_dims

        # outputs_bev_query = self.dense_radar_encoder(
        #     outputs_bev_query, radar_feats
        # )  # (128 128) b 128

        if outputs_bev_query.dim() == 4:
            outputs_bev_query = outputs_bev_query.flatten(2).permute(2, 0, 1)

        # dense_radar_bev = self.multi_bev_fusion_encoder(
        #     query=img_bev,
        #     key=radar_feats,
        #     value=None,
        # )

        # Detect decoding initialization
        object_query_embeds = self.query_embedding.weight.to(dtype)
        # bev_queries = self.bev_embedding.weight.to(dtype)
        # bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)

        (
            object_query_embeds,
            query_feat,
            attn_mask,
            mask_dict,
        ) = self.prepare_for_dn_input(B, object_query_embeds, self.label_enc, img_metas)

        # object query deocder transformer
        outputs = self.transformer(
            img_bev,
            outputs_bev_query,
            query_feat,
            mlvl_feats,
            feats_shapes=torch.as_tensor(
                [(self.bev_h, self.bev_w)],
                dtype=torch.long,
                device=object_query_embeds.device,
            ),
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            query_bbox=object_query_embeds,
            pre_attn_mask=attn_mask,
        )

        hs, init_reference, inter_references = outputs

        hs = hs.permute(
            0, 2, 1, 3
        )  # levels num_query bs embed_dims -> levels bs num_query embed_dims

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if mask_dict is not None and mask_dict["pad_size"] > 0:
            output_known_cls_scores = outputs_classes[:, :, : mask_dict["pad_size"], :]
            output_known_bbox_preds = outputs_coords[:, :, : mask_dict["pad_size"], :]
            output_cls_scores = outputs_classes[:, :, mask_dict["pad_size"] :, :]
            output_bbox_preds = outputs_coords[:, :, mask_dict["pad_size"] :, :]
            mask_dict["output_known_lbs_bboxes"] = (
                output_known_cls_scores,
                output_known_bbox_preds,
            )
            outs = {
                "all_cls_scores": output_cls_scores,
                "all_bbox_preds": output_bbox_preds,
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
                "dn_mask_dict": mask_dict,
            }
        else:
            outs = {
                "all_cls_scores": outputs_classes,
                "all_bbox_preds": outputs_coords,
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
            }

        # outs = {
        #     "all_cls_scores": outputs_classes,
        #     "all_bbox_preds": outputs_coords,
        #     "enc_cls_scores": None,
        #     "enc_bbox_preds": None,
        # }

        return outs

    def prepare_for_dn_input(self, batch_size, init_query_bbox, label_enc, img_metas):
        device = init_query_bbox.device
        indicator0 = torch.zeros([self.num_query, 1], device=device)
        init_query_feat = label_enc.weight[self.num_classes].repeat(self.num_query, 1)
        init_query_feat = torch.cat([init_query_feat, indicator0], dim=1)

        if self.training and self.dn_enabled:
            targets = [
                {
                    "bboxes": torch.cat(
                        [
                            m["gt_bboxes_3d"].gravity_center,
                            m["gt_bboxes_3d"].tensor[:, 3:],
                        ],
                        dim=1,
                    ).cuda(),
                    "labels": m["gt_labels_3d"].cuda().long(),
                }
                for m in img_metas
            ]

            known = [torch.ones_like(t["labels"], device=device) for t in targets]
            known_num = [sum(k) for k in known]

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t["labels"] for t in targets]).clone()
            bboxes = torch.cat([t["bboxes"] for t in targets]).clone()
            batch_idx = torch.cat(
                [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
            )

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # add noise
            known_indice = known_indice.repeat(self.dn_group_num, 1).view(-1)
            known_labels = labels.repeat(self.dn_group_num, 1).view(-1)
            known_bid = batch_idx.repeat(self.dn_group_num, 1).view(-1)
            known_bboxs = bboxes.repeat(self.dn_group_num, 1)  # 9
            known_labels_expand = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the box
            if self.dn_bbox_noise_scale > 0:
                wlh = known_bbox_expand[..., 3:6].clone()
                rand_prob = torch.rand_like(known_bbox_expand) * 2 - 1.0
                known_bbox_expand[..., 0:3] += (
                    torch.mul(rand_prob[..., 0:3], wlh / 2) * self.dn_bbox_noise_scale
                )
                # known_bbox_expand[..., 3:6] += (
                #     torch.mul(rand_prob[..., 3:6], wlh)
                #     * self.dn_bbox_noise_scale
                # )
                # known_bbox_expand[..., 6:7] += (
                #     torch.mul(rand_prob[..., 6:7], 3.14159)
                #     * self.dn_bbox_noise_scale
                # )

            known_bbox_expand = encode_bbox(known_bbox_expand, self.pc_range)
            known_bbox_expand[..., 0:3].clamp_(min=0.0, max=1.0)
            # nn.init.constant(known_bbox_expand[..., 8:10], 0.0)

            # noise on the label
            if self.dn_label_noise_scale > 0:
                p = torch.rand_like(known_labels_expand.float())
                chosen_indice = torch.nonzero(p < self.dn_label_noise_scale).view(
                    -1
                )  # usually half of bbox noise
                new_label = torch.randint_like(
                    chosen_indice, 0, self.num_classes
                )  # randomly put a new one here
                known_labels_expand.scatter_(0, chosen_indice, new_label)

            known_feat_expand = label_enc(known_labels_expand)
            indicator1 = torch.ones(
                [known_feat_expand.shape[0], 1], device=device
            )  # add dn part indicator
            known_feat_expand = torch.cat([known_feat_expand, indicator1], dim=1)

            # construct final query
            dn_single_pad = int(max(known_num))
            dn_pad_size = int(dn_single_pad * self.dn_group_num)
            dn_query_bbox = torch.zeros(
                [dn_pad_size, init_query_bbox.shape[-1]], device=device
            )
            dn_query_feat = torch.zeros([dn_pad_size, self.embed_dims], device=device)
            input_query_bbox = torch.cat(
                [dn_query_bbox, init_query_bbox], dim=0
            ).repeat(batch_size, 1, 1)
            input_query_feat = torch.cat(
                [dn_query_feat, init_query_feat], dim=0
            ).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat(
                    [torch.tensor(range(num)) for num in known_num]
                )  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [
                        map_known_indice + dn_single_pad * i
                        for i in range(self.dn_group_num)
                    ]
                ).long()

            if len(known_bid):
                input_query_bbox[known_bid.long(), map_known_indice] = known_bbox_expand
                input_query_feat[
                    (known_bid.long(), map_known_indice)
                ] = known_feat_expand

            total_size = dn_pad_size + self.num_query
            attn_mask = torch.ones([total_size, total_size], device=device) < 0

            # match query cannot see the reconstruct
            attn_mask[dn_pad_size:, :dn_pad_size] = True
            for i in range(self.dn_group_num):
                if i == 0:
                    attn_mask[
                        dn_single_pad * i : dn_single_pad * (i + 1),
                        dn_single_pad * (i + 1) : dn_pad_size,
                    ] = True
                if i == self.dn_group_num - 1:
                    attn_mask[
                        dn_single_pad * i : dn_single_pad * (i + 1),
                        : dn_single_pad * i,
                    ] = True
                else:
                    attn_mask[
                        dn_single_pad * i : dn_single_pad * (i + 1),
                        dn_single_pad * (i + 1) : dn_pad_size,
                    ] = True
                    attn_mask[
                        dn_single_pad * i : dn_single_pad * (i + 1),
                        : dn_single_pad * i,
                    ] = True

            mask_dict = {
                "known_indice": torch.as_tensor(known_indice).long(),
                "batch_idx": torch.as_tensor(batch_idx).long(),
                "map_known_indice": torch.as_tensor(map_known_indice).long(),
                "known_lbs_bboxes": (known_labels, known_bboxs),
                "pad_size": dn_pad_size,
            }
        else:
            input_query_bbox = init_query_bbox.repeat(batch_size, 1, 1)
            input_query_feat = init_query_feat.repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return input_query_bbox, input_query_feat, attn_mask, mask_dict

    def prepare_for_dn_loss(self, mask_dict):
        cls_scores, bbox_preds = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_bboxs = mask_dict["known_lbs_bboxes"]
        map_known_indice = mask_dict["map_known_indice"].long()
        known_indice = mask_dict["known_indice"].long()
        batch_idx = mask_dict["batch_idx"].long()
        bid = batch_idx[known_indice]
        num_tgt = known_indice.numel()

        if len(cls_scores) > 0:
            cls_scores = cls_scores.permute(1, 2, 0, 3)[
                (bid, map_known_indice)
            ].permute(1, 0, 2)
            bbox_preds = bbox_preds.permute(1, 2, 0, 3)[
                (bid, map_known_indice)
            ].permute(1, 0, 2)

        return known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt

    def dn_loss_single(
        self,
        cls_scores,
        bbox_preds,
        known_bboxs,
        known_labels,
        num_total_pos=None,
    ):
        # Compute the average number of gt boxes accross all gpus
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1.0).item()

        # cls loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        loss_cls = self.loss_cls(
            cls_scores,
            known_labels.long(),
            label_weights,
            avg_factor=num_total_pos,
        )

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )

        loss_cls = self.dn_weight * torch.nan_to_num(loss_cls)
        loss_bbox = self.dn_weight * torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=("preds_dicts"))
    def calc_dn_loss(self, loss_dict, preds_dicts, num_dec_layers):
        (
            known_labels,
            known_bboxs,
            cls_scores,
            bbox_preds,
            num_tgt,
        ) = self.prepare_for_dn_loss(preds_dicts["dn_mask_dict"])

        all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
        all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]

        dn_losses_cls, dn_losses_bbox = multi_apply(
            self.dn_loss_single,
            cls_scores,
            bbox_preds,
            all_known_bboxs_list,
            all_known_labels_list,
            all_num_tgts_list,
        )

        loss_dict["loss_cls_dn"] = dn_losses_cls[-1]
        loss_dict["loss_bbox_dn"] = dn_losses_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1], dn_losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls_dn"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox_dn"] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
        img_metas=None,
    ):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        if "dn_mask_dict" in preds_dicts and preds_dicts["dn_mask_dict"] is not None:
            loss_dict = self.calc_dn_loss(loss_dict, preds_dicts, num_dec_layers)

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict


@HEADS.register_module()
class DN_MSBEVQuery_DenseRadarHead(DN_DenseRadarHead):
    def __init__(self, query_denoising=True, query_denoising_groups=10, **kwargs):
        super().__init__(query_denoising, query_denoising_groups, **kwargs)

        self.bev_query_embed = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def forward(self, img_inputs, mlvl_feats, img_metas, img_bev, radar_feats):
        """DN_MSBEVQuery_DenseRadarHead detection head forward computing.

        Args:
            img_inputs (list[torch.tensor]): original_imgs, rots, trans, intrins, post_rots, post_trans, lidar_to_img_depth
            mlvl_feats (list[torch.tensor]):
            img_metas (list[dict]): img_meats of all the total parameters.
            img_bev (torch.tensor): BEV features with model of LSS.
            radar_feats (torch.tensoer): get the radar feats from  the projet of the LiDAR points.

        Returns:
            outs (dict): forward computation results, and wrap the primearily detection results.
        """
        # print(radar_feats, "==========[][][]")

        B, num_cam, _, _, _ = mlvl_feats[0].shape
        _, _, radar_shape_h, radar_shape_w = radar_feats.shape
        dtype = mlvl_feats[0].dtype

        # Multi-scale bev query for BEV feas generation.
        bev_query_embed = self.bev_query_embed.weight.to(dtype)
        bev_mask = torch.zeros(
            (B, self.bev_h, self.bev_w), device=bev_query_embed.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outputs_bev_query = self.bev_query_encoder(
            bev_query_embed,
            mlvl_feats,
            bev_pos,
            img_inputs=img_inputs,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            img_metas=img_metas,
        )  # b bev_shape//downsample embed_dims

        if self.dense_radar_encoder is not None:
            outputs_bev_query = self.dense_radar_encoder(
                outputs_bev_query, radar_feats
            )  # (128 128) b 128
            if len(outputs_bev_query.shape) == 4:
                outputs_bev_query = outputs_bev_query.reshape(
                    *outputs_bev_query.shape[:2], -1
                ).permute(2, 0, 1)
        else:
            pass

        # dense_radar_bev = self.multi_bev_fusion_encoder(
        #     query=img_bev,
        #     key=radar_feats,
        #     value=None,
        # )

        # Detect decoding initialization
        # Use denoising detr mode initialization, learning detection target and initialization reference point
        object_query_embeds = self.query_embedding.weight.to(dtype)

        (
            object_query_embeds,
            query_feat,
            attn_mask,
            mask_dict,
        ) = self.prepare_for_dn_input(B, object_query_embeds, self.label_enc, img_metas)

        # object query deocder transformer
        outputs = self.transformer(
            img_bev,
            outputs_bev_query,
            query_feat,
            mlvl_feats,
            feats_shapes=torch.as_tensor(
                [(self.bev_h, self.bev_w)],
                dtype=torch.long,
                device=object_query_embeds.device,
            ),
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            query_bbox=object_query_embeds,
            pre_attn_mask=attn_mask,
        )

        hs, init_reference, inter_references = outputs

        hs = hs.permute(
            0, 2, 1, 3
        )  # levels num_query bs embed_dims -> levels bs num_query embed_dims

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if mask_dict is not None and mask_dict["pad_size"] > 0:
            output_known_cls_scores = outputs_classes[:, :, : mask_dict["pad_size"], :]
            output_known_bbox_preds = outputs_coords[:, :, : mask_dict["pad_size"], :]
            output_cls_scores = outputs_classes[:, :, mask_dict["pad_size"] :, :]
            output_bbox_preds = outputs_coords[:, :, mask_dict["pad_size"] :, :]
            mask_dict["output_known_lbs_bboxes"] = (
                output_known_cls_scores,
                output_known_bbox_preds,
            )
            outs = {
                "all_cls_scores": output_cls_scores,
                "all_bbox_preds": output_bbox_preds,
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
                "dn_mask_dict": mask_dict,
            }
        else:
            outs = {
                "all_cls_scores": outputs_classes,
                "all_bbox_preds": outputs_coords,
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
            }

        # outs = {
        #     "all_cls_scores": outputs_classes,
        #     "all_bbox_preds": outputs_coords,
        #     "enc_cls_scores": None,
        #     "enc_bbox_preds": None,
        # }

        return outs
