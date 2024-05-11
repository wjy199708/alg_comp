_base_ = [
    # "../datasets/custom_nus-3d.py",
    "../../_base_/datasets/nus-3d.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/cyclic_24e.py",
]
#
# plugin = True
# plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# voxel_size = [0.2, 0.2, 8]
voxel_size = [0.1, 0.1, 0.2]
# Model
grid_config = {
    "xbound": [-51.2, 51.2, 0.8],
    "ybound": [-51.2, 51.2, 0.8],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

data_config = {
    "cams": [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    # Augmentation
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.04,
}

# radar configuration,  x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 8, 9, 18]
radar_voxel_size = [0.8, 0.8, 8]
# radar_voxel_size = [0.6, 0.6, 8]
radar_max_voxels_times = 3


img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True,
)

_dim_ = 256
numC_Trans = _dim_ // 4

bev_h_ = 128
bev_w_ = 128
queue_length = 4  # each sequence contains `queue_length` frames.

loss_depth_weight = 100
downsample = 16

teacher_pretrained = "/mnt/data/exps/DenseRadar/ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"

# swin_pretrained = "/mnt/data/exps/DenseRadar/ckpts/bevdet-stereo-geomim-512x1408.pth"

model = dict(
    type="D3PD_V1",
    teacher_pretrained=teacher_pretrained,
    # swin_pretrained=swin_pretrained,
    # ====================imgs feature processing=======================
    img_backbone=dict(
        pretrained="torchvision://resnet50",
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style="pytorch",
    ),
    img_neck=dict(
        type="FPNForBEVDet",
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    # ====================img-bev processing=======================
    img_view_transformer=dict(
        type="ViewTransformerLSSBEVDepth",
        loss_depth_weight=loss_depth_weight,
        grid_config=grid_config,
        data_config=data_config,
        numC_input=_dim_ * 2,
        numC_Trans=numC_Trans,
        downsample=downsample,
        extra_depth_net=dict(
            type="ResNetForBEVDet",
            numC_input=_dim_,
            num_layer=[
                3,
            ],
            num_channels=[
                256,
            ],
            stride=[
                1,
            ],
        ),
    ),
    img_bev_encoder_backbone=dict(type="ResNetForBEVDet", numC_input=numC_Trans),
    img_bev_encoder_neck=dict(
        type="FPN_LSS",
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256,  # 256
    ),
    # ====================radar feature processing=======================
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(
            30000 * radar_max_voxels_times,
            40000 * radar_max_voxels_times,
        ),
        point_cloud_range=point_cloud_range,
        deterministic=False,
    ),
    radar_pillar_encoder=dict(
        type="PillarFeatureNet",
        in_channels=6,
        feat_channels=[64],
        with_distance=False,
        voxel_size=radar_voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    radar_middle_encoder=dict(
        type="PointPillarsScatter", in_channels=64, output_shape=(128, 128)
    ),
    radar_backbone=dict(
        type="SECOND",
        in_channels=64,
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        layer_nums=[3, 5],
        layer_strides=[1, 2],
        out_channels=[64, 128],
    ),
    radar_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        out_channels=[128, 128],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    # ====================lidar processing=======================
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000),
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=5),
    pts_middle_encoder=dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    rc_bev_fusion=dict(
        type="RC_BEV_Fusion_Sampling",
        low_feats_channels=256,
        hight_feats_channels=256,
        process=[dict(type="SpatialProbAtten"), dict(type="DualWeight_Fusion")],
    ),
    # dict(
    #     type="SamplingWarpFusion",
    #     features=256,
    #     reduce_ch=dict(in_channels=320, in_channels2=512),
    #     ret_sampling_feats=True,
    #     bi_dir=dict(
    #         in_channels=512,
    #         bi_weight=True,
    #         bi_weight_fusion=dict(
    #             type="BiDirectionWeightFusion", img_channels=256, radar_channels=64
    #         ),
    #     ),
    # ),
    distillation=dict(
        sparse_feats_distill=dict(type=None),
        sampling_pos_distill=dict(type="SimpleL1"),
        sampling_feats_distill=dict(
            type="FeatureLoss",
            student_channels=256,
            teacher_channels=256,
            name="sampling_disitll",
            alpha_mgd=0.00002,
            lambda_mgd=0.65,
        ),
        det_result_distill=dict(
            type="Dc_ResultDistill",
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_scale=8,
            ret_sum=False,
            loss_weight_reg=10,
            loss_weight_cls=1,
        ),
        mask_bev_feats_distill=dict(
            type="SelfLearningMFD",
            loss_weight=1e-2,
        ),
    ),
    x_sample_num=24,
    y_sample_num=24,
    embed_channels=[256, 512],
    inter_keypoint_weight=100.0,
    enlarge_width=1.6,
    inner_depth_weight=1.0,
    inter_channel_weight=10.0,
    # ====================detection head processing=======================
    pts_bbox_head=dict(
        type="CenterHead",
        task_specific=True,
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    pts_bbox_head_tea=dict(
        type="CenterHead",
        task_specific=True,
        in_channels=256 * 2,
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            # Scale-NMS
            nms_type=["rotate", "rotate", "rotate", "circle", "rotate", "rotate"],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0,
                [0.7, 0.7],
                [0.4, 0.55],
                1.1,
                [1.0, 1.0],
                [4.5, 9.0],
            ],
        )
    ),
)

# dataset_type = "CustomNuScenesDataset"
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")


train_pipeline = [
    dict(
        type="LoadRadarPointsMultiSweeps",
        load_dim=18,
        sweeps_num=6,
        use_dim=radar_use_dims,
        max_num=1200,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="LoadMultiViewImageFromFiles_DenseRadar", data_config=data_config),
    # dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="PointToMultiViewDepth",
        grid_config=grid_config,
        downsample=downsample,
    ),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    # dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D_W_Radar", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=["gt_bboxes_3d", "gt_labels_3d", "img_inputs", "radar", "points"],
    ),
]

test_pipeline = [
    dict(
        type="LoadRadarPointsMultiSweeps",
        load_dim=18,
        sweeps_num=6,
        use_dim=radar_use_dims,
        max_num=1200,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type="LoadPointsFromMultiSweeps",
    #     sweeps_num=10,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    # ),
    # dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadMultiViewImageFromFiles_DenseRadar", data_config=data_config),
    dict(
        type="PointToMultiViewDepth",
        grid_config=grid_config,
        downsample=downsample,
    ),
    # dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D_W_Radar",
                class_names=class_names,
                with_label=False,
            ),
            dict(type="CustomCollect3D", keys=["img_inputs", "radar", "points"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            box_type_3d="LiDAR",
            img_info_prototype="bevdet",
        ),
    ),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        img_info_prototype="bevdet",
    ),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        img_info_prototype="bevdet",
    ),
)



checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, pipeline=test_pipeline)

load_from = teacher_pretrained


work_dir = "d3pd/v1/r50-all-distill"
