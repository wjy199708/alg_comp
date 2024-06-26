# Copyright (c) OpenMMLab. All rights reserved.
import os
import io

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results["occ_gt_path"]
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels["semantics"]
        mask_lidar = occ_labels["mask_lidar"]
        mask_camera = occ_labels["mask_camera"]

        results["voxel_semantics"] = semantics
        results["mask_lidar"] = mask_lidar
        results["mask_camera"] = mask_camera

        return results


@PIPELINES.register_module()
class PointToMultiViewDepth_D3PD(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (
            (coor[:, 0] >= 0)
            & (coor[:, 0] < width)
            & (coor[:, 1] >= 0)
            & (coor[:, 1] < height)
            & (depth < self.grid_config["depth"][1])
            & (depth >= self.grid_config["depth"][0])
        )
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.0).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = ranks[1:] != ranks[:-1]
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results["points"]
        imgs, rots, trans, intrins = results["img_inputs"][:4]
        post_rots, post_trans, bda = results["img_inputs"][4:]
        # post_rots, post_trans = results["img_inputs"][4:]
        depth_map_list = []
        for cid in range(len(results["cam_names"])):
            cam_name = results["cam_names"][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results["curr"]["lidar2ego_rotation"]
            ).rotation_matrix
            lidar2lidarego[:3, 3] = results["curr"]["lidar2ego_translation"]
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results["curr"]["ego2global_rotation"]
            ).rotation_matrix
            lidarego2global[:3, 3] = results["curr"]["ego2global_translation"]
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results["curr"]["cams"][cam_name]["sensor2ego_rotation"]
            ).rotation_matrix
            cam2camego[:3, 3] = results["curr"]["cams"][cam_name][
                "sensor2ego_translation"
            ]
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results["curr"]["cams"][cam_name]["ego2global_rotation"]
            ).rotation_matrix
            camego2global[:3, 3] = results["curr"]["cams"][cam_name][
                "ego2global_translation"
            ]
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego)
            )
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T
            ) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1
            )
            points_img = (
                points_img.matmul(post_rots[cid].T) + post_trans[cid : cid + 1, :]
            )
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results["gt_depth"] = depth_map
        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        file_client_args=dict(backend="disk"),
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def img_transform(
        self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config["Ncams"] < len(self.data_config["cams"]):
            cam_names = np.random.choice(
                self.data_config["cams"], self.data_config["Ncams"], replace=False
            )
        else:
            cam_names = self.data_config["cams"]
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config["input_size"]
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config["resize"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.data_config["crop_h"])) * newH) - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config["flip"] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config["rot"])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get("resize_test", 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config["crop_h"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info["cams"][cam_name]["sensor2ego_rotation"]
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info["cams"][cam_name]["sensor2ego_translation"]
        )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info["cams"][cam_name]["ego2global_rotation"]
        ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info["cams"][cam_name]["ego2global_translation"]
        )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def read_img(self, fname):
        value = self.file_client.get(fname)
        buff = io.BytesIO(value)
        img = Image.open(buff)
        return img

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results["cam_names"] = cam_names
        canvas = []

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        for cam_name in cam_names:
            cam_data = results["curr"]["cams"][cam_name]
            filename = cam_data["data_path"]
            # img = Image.open(filename)
            img = self.read_img(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data["cam_intrinsic"])

            sensor2ego, ego2global = self.get_sensor_transforms(
                results["curr"], cam_name
            )
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale
            )
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert "adjacent" in results
                for adj_info in results["adjacent"]:
                    filename_adj = adj_info["cams"][cam_name]["data_path"]
                    # img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.read_img(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results["adjacent"]:
                post_trans.extend(post_trans[: len(cam_names)])
                post_rots.extend(post_rots[: len(cam_names)])
                intrins.extend(intrins[: len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = self.get_sensor_transforms(
                        adj_info, cam_name
                    )
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results["canvas"] = canvas
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results["img_inputs"] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf["rot_lim"])
            scale_bda = np.random.uniform(*self.bda_aug_conf["scale_lim"])
            flip_dx = np.random.uniform() < self.bda_aug_conf["flip_dx_ratio"]
            flip_dy = np.random.uniform() < self.bda_aug_conf["flip_dy_ratio"]
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        scale_mat = torch.Tensor(
            [[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]
        )
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(
                -1
            )
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results["ann_infos"]
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(
            gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy
        )

        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results["gt_bboxes_3d"] = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5)
        )
        results["gt_labels_3d"] = gt_labels
        imgs, rots, trans, intrins = results["img_inputs"][:4]
        post_rots, post_trans = results["img_inputs"][4:]
        results["img_inputs"] = (
            imgs,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            bda_rot,
        )

        if "voxel_semantics" in results:
            if flip_dx:
                results["voxel_semantics"] = results["voxel_semantics"][
                    ::-1, ...
                ].copy()
                results["mask_lidar"] = results["mask_lidar"][::-1, ...].copy()
                results["mask_camera"] = results["mask_camera"][::-1, ...].copy()
            if flip_dy:
                results["voxel_semantics"] = results["voxel_semantics"][
                    :, ::-1, ...
                ].copy()
                results["mask_lidar"] = results["mask_lidar"][:, ::-1, ...].copy()
                results["mask_camera"] = results["mask_camera"][:, ::-1, ...].copy()
        return results


@PIPELINES.register_module()
class LoadAnnotations_D3PD(object):
    def __call__(self, results):
        gt_boxes, gt_labels = results["ann_infos"]
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results["gt_bboxes_3d"] = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5)
        )
        results["gt_labels_3d"] = gt_labels
        return results


@PIPELINES.register_module()
class BEVAug(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf["rot_lim"])
            scale_bda = np.random.uniform(*self.bda_aug_conf["scale_lim"])
            flip_dx = np.random.uniform() < self.bda_aug_conf["flip_dx_ratio"]
            flip_dy = np.random.uniform() < self.bda_aug_conf["flip_dy_ratio"]
            translation_std = self.bda_aug_conf.get("tran_lim", [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(
        self, gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy, tran_bda
    ):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        scale_mat = torch.Tensor(
            [[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]
        )
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(
                -1
            )
            gt_boxes[:, :3] = gt_boxes[:, :3] + tran_bda
        return gt_boxes, rot_mat

    def __call__(self, results):

        gt_boxes = results["gt_bboxes_3d"].tensor
        gt_boxes[:, 2] = gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]
        (
            rotate_bda,
            scale_bda,
            flip_dx,
            flip_dy,
            tran_bda,
        ) = self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(
            gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda
        )

        if "points" in results:
            points = results["points"].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:, :3] = points_aug + tran_bda
            points = results["points"].new_point(points)
            results["points"] = points
        if "radar" in results:
            points = results["radar"].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:, :3] = points_aug + tran_bda
            points = results["radar"].new_point(points)
            results["radar"] = points

        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results["gt_bboxes_3d"] = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5)
        )
        if "img_inputs" in results:
            imgs, rots, trans, intrins = results["img_inputs"][:4]
            post_rots, post_trans = results["img_inputs"][4:]
            results["img_inputs"] = (
                imgs,
                rots,
                trans,
                intrins,
                post_rots,
                post_trans,
                # bda_mat,
                bda_rot,
            )

        if "voxel_semantics" in results:
            if flip_dx:
                results["voxel_semantics"] = results["voxel_semantics"][
                    ::-1, ...
                ].copy()
                results["mask_lidar"] = results["mask_lidar"][::-1, ...].copy()
                results["mask_camera"] = results["mask_camera"][::-1, ...].copy()
            if flip_dy:
                results["voxel_semantics"] = results["voxel_semantics"][
                    :, ::-1, ...
                ].copy()
                results["mask_lidar"] = results["mask_lidar"][:, ::-1, ...].copy()
                results["mask_camera"] = results["mask_camera"][:, ::-1, ...].copy()
        return results
