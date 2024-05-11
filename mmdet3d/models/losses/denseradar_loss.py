import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class DepthLossForImgBEV(nn.Module):
    def __init__(self, grid_config, loss_depth_weight) -> None:
        super().__init__()
        self.grid_config = grid_config
        self.loss_depth_weight = loss_depth_weight
        self.D = int(len(torch.arange(*grid_config["dbound"])))

    def forward(self, depth_gt, depth):
        B, N, H, W = depth_gt.shape
        loss_weight = (
            (~(depth_gt == 0)).reshape(B, N, 1, H, W).expand(B, N, self.D, H, W)
        )
        depth_gt = (depth_gt - self.grid_config["dbound"][0]) / self.grid_config[
            "dbound"
        ][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0, self.D).to(torch.long)
        depth_gt_logit = F.one_hot(depth_gt.reshape(-1), num_classes=self.D)
        depth_gt_logit = (
            depth_gt_logit.reshape(B, N, H, W, self.D)
            .permute(0, 1, 4, 2, 3)
            .to(torch.float32)
        )
        depth = depth.sigmoid().view(B, N, self.D, H, W)

        loss_depth = F.binary_cross_entropy(depth, depth_gt_logit, weight=loss_weight)
        loss_depth = self.loss_depth_weight * loss_depth
        return loss_depth
