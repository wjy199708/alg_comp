# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy
from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from .paconv_regularization_loss import PAConvRegularizationLoss
from .denseradar_loss import DepthLossForImgBEV
from .distill_loss import (
    FeatureLoss,
    Dc_ResultDistill,
    SimpleL1,
    SelfLearningMFD,
    FeatureLoss_Affinity,
    FeatureLoss_Coefficient,
    QualityFocalLoss_,
    FeatureLoss_InnerClip,
    Relevance_Distillation,
    InfoMax,
    Radar_MSDistilll,
)

__all__ = [
    "FocalLoss",
    "SmoothL1Loss",
    "binary_cross_entropy",
    "ChamferDistance",
    "chamfer_distance",
    "axis_aligned_iou_loss",
    "AxisAlignedIoULoss",
    "PAConvRegularizationLoss",
    "DepthLossForImgBEV",
    "FeatureLoss",
    "Dc_ResultDistill",
    "SimpleL1",
    "SelfLearningMFD",
    "FeatureLoss_Affinity",
    "FeatureLoss_Coefficient",
    "QualityFocalLoss_",
    "FeatureLoss_InnerClip",
    "Relevance_Distillation",
    "InfoMax",
    "Radar_MSDistilll",
]
