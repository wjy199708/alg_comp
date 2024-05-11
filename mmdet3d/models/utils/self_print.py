from matplotlib import pyplot as plt
import cv2
import os
import torch
import numpy as np


def print2file(content, suffix="", end=".json", mode="w"):
    f = open("/mnt/data/exps_logs/out_" + suffix + end, mode=mode)
    print(content, file=f)


# def feats_to_img(feats, base_path, suffix="out", **kwargs):

#     base_path = os.path.join(base_path, suffix)

#     base_path = os.path.join(base_path, suffix)
#     if os.path.exists(base_path):
#         pass
#     else:
#         # os.mkdir(base_path)
#         os.makedirs(base_path)

#     bs, c, h, w = feats.shape
#     assert bs >= 1
#     # feats = feats[0].detach().cpu().numpy()
#     feats = feats[0]
#     if "boxes" in kwargs.keys():
#         gt_boxes = kwargs["boxes"]


#     for idx, feat in enumerate(feats):

#         heatmapshow = None
#         # heatmapshow = cv2.normalize(
#         #     feats,
#         #     heatmapshow,
#         #     alpha=0,
#         #     beta=255,
#         #     norm_type=cv2.NORM_MINMAX,
#         #     dtype=cv2.CV_8U,
#         # )

#         heatmapshow = (
#             feats.mul(255)
#             .add_(0.5)
#             .clamp_(0, 255)
#             .permute(1, 2, 0)
#             .to("cpu", torch.uint8)
#             .numpy()
#         )
#         heatmapshow = heatmapshow.astype(np.uint8)
#         heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

#         for box in gt_boxes:
#             cv2.circle(heatmapshow, box[:2] / 8, 3, "black", 0)

#         cv2.imwrite(
#             f"{base_path}/gray_scale_tensor{idx}.png",
#             heatmapshow,
#         )


def feats_to_img(feats, base_path, suffix="out", **kwargs):
    feats = feats[0] if isinstance(feats, list) else feats
    import os

    base_path = os.path.join(base_path, suffix)
    if os.path.exists(base_path):
        pass
    else:
        # os.mkdir(base_path)
        os.makedirs(base_path)

    bs, c, h, w = feats.shape
    assert bs >= 1
    feats = feats[0].detach().cpu().numpy()

    for idx, feat in enumerate(feats):

        # print()
        plt.imsave(
            f"{base_path}/gray_scale_tensor{idx}.png",
            feat,
            cmap="gray",
        )


def feats_to_img_boxes(feats, base_path, suffix="out", **kwargs):
    feats = feats[0] if isinstance(feats, list) else feats
    import os

    boxes = kwargs["boxes"]
    points = [(box[0] / 8, box[1] / 8) for box in boxes]

    base_path = os.path.join(base_path, suffix)
    if os.path.exists(base_path):
        pass
    else:
        # os.mkdir(base_path)
        os.makedirs(base_path)

    bs, c, h, w = feats.shape
    assert bs >= 1
    feats = feats[0].detach().cpu().numpy()

    for idx, feat in enumerate(feats):

        for point in points:
            plt.plot(point[0], point[1], "bo")  # 'bo'表示蓝色圆点

        # print()
        plt.imsave(
            f"{base_path}/gray_scale_tensor{idx}.png",
            feat,
            cmap="gray",
        )
