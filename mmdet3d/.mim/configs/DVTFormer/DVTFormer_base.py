_base_ = [
    # "../datasets/custom_nus-3d.py",
    "../_base_/datasets/nus-3d.py",
    "../_base_/default_runtime.py",
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
    use_lidar=False, use_camera=True, use_radar=True, use_map=False, use_external=True
)

_dim_ = 256
numC_Trans = _dim_ // 4
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4  # each sequence contains `queue_length` frames.

loss_depth_weight = 100
downsample = 8

model = dict(
    type="DVTFormer",
    use_grid_mask=True,
    # video_test_mode=True,
    # img encoder
    img_backbone=dict(
        type="ResNet",
        # depth=101,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),
    # img_neck=dict(
    #     type="FPNForBEVDet",
    #     in_channels=[512, 1024, 2048],
    #     out_channels=512,
    #     num_outs=1,
    #     start_level=0,
    #     out_ids=[0],
    # ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    img_view_transformer=dict(
        type="ViewTransformerLSSBEVDepth",
        loss_depth_weight=loss_depth_weight,
        grid_config=grid_config,
        data_config=data_config,
        numC_input=_dim_,
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
        depth_loss=dict(
            type="DepthLossForImgBEV",
            grid_config=grid_config,
            loss_depth_weight=loss_depth_weight,
        ),
    ),
    img_bev_encoder_backbone=dict(type="ResNetForBEVDet", numC_input=numC_Trans),
    img_bev_encoder_neck=dict(
        type="FPN_LSS", in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256
    ),
    # radar encoder
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(30000 * radar_max_voxels_times, 40000 * radar_max_voxels_times),
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
    pts_bbox_head=dict(
        # NOTE input of pts_bbox_head is [mlvls_img_feats, img_metas, pre_bev_fetas, is_only_bev?, radar_pillar_bev_feats, cam_lss_bev_feats/cam_fast_bev_feats_point_sampling]
        type="DenseRadarHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        num_views=len(data_config["cams"]),
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        bev_query_encoder=dict(
            type="DenseRadarBEVEncoder",
            data_config=data_config,
            downsample=downsample,
            bev_shape=(bev_h_, bev_w_),
            bev_embedding=dict(
                type="MatualBEVEmbedding",  # MatualBEVEmbedding, LearningBEVEmbedding
                sigma=1.0,
                embed_dim=_dim_,
                bev_pos_embedding=dict(
                    type="SinePositionalEncoding3D", num_feats=_pos_dim_
                ),
            ),
            num_layers=2,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=[
                    dict(
                        type="CrossViewBEVAttention",
                        embed_dims=_dim_,
                        num_views=len(data_config["cams"]),
                        heads=8,
                        qkv_bias=True,
                        dim_head=_dim_ // 8,
                        # dropout=0.1,
                    ),
                ],
                ffn_cfgs=dict(
                    type="FFN",
                    embed_dims=256,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type="GELU"),
                ),
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=("cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        dense_radar_encoder=dict(
            type="CamsRadarEncoder",
            num_layers=1,
            return_intermediate=False,
            bev_query_decoder=True,
            bev_query_decoder_layers=[256, 256, 64],
            upsample_residual=True,
            upsample_factor=2,
            bev_shape=(bev_h_, bev_w_),
            bev_downsample=downsample,
            query_channel=_dim_,
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=[
                    dict(
                        type="MultiheadAttention",
                        embed_dims=_dim_ // 2,
                        num_heads=1,
                        dropout=0.1,
                    ),
                ],
                ffn_cfgs=dict(
                    type="FFN",
                    embed_dims=_dim_ // 2,
                    feedforward_channels=_dim_,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
                operation_order=("cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        transformer=dict(
            type="DenseRadarPerceptionTransformer",
            embed_dims=_dim_,
            decoder=dict(
                type="DenseRadarTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CamRadarCrossAtten",
                            num_heads=8,
                            num_levels=1,
                            radar_num_points=4,
                            num_points=1,
                            dense_radar_dims_in=_dim_ // 2,
                            embed_dims=_dim_,
                            embed_dims_in=sum([128, 256]),
                            embed_dims_out=_dim_,
                            # num_sweeps=cam_sweep_num,
                            # fp16_enabled=fp16_enabled,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=256,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    norm_cfg=dict(type="LN"),
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
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
        sweeps_num=4,
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
    dict(type="LoadMultiViewImageFromFiles_DenseRadar", data_config=data_config),
    # dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="PointToMultiViewDepth", grid_config=grid_config, downsample=downsample),
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
        keys=["gt_bboxes_3d", "gt_labels_3d", "img_inputs", "radar"],
    ),
]

test_pipeline = [
    dict(
        type="LoadRadarPointsMultiSweeps",
        load_dim=18,
        sweeps_num=4,
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
    # dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadMultiViewImageFromFiles_DenseRadar", data_config=data_config),
    dict(type="PointToMultiViewDepth", grid_config=grid_config, downsample=downsample),
    # dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D_W_Radar",
                class_names=class_names,
                with_label=False,
            ),
            dict(type="CustomCollect3D", keys=["img_inputs", "radar"]),
        ],
    ),
]

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
#     dataset_scale=4,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + "nuscenes_infos_temporal_train.pkl",
#         pipeline=train_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=False,
#         use_valid_flag=True,
#         bev_size=(bev_h_, bev_w_),
#         queue_length=queue_length,
#         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#         box_type_3d="LiDAR",
#     ),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + "nuscenes_infos_temporal_val.pkl",
#         pipeline=test_pipeline,
#         bev_size=(bev_h_, bev_w_),
#         classes=class_names,
#         modality=input_modality,
#         samples_per_gpu=1,
#     ),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + "nuscenes_infos_temporal_val.pkl",
#         pipeline=test_pipeline,
#         bev_size=(bev_h_, bev_w_),
#         classes=class_names,
#         modality=input_modality,
#     ),
#     shuffler_sampler=dict(type="DistributedGroupSampler"),
#     nonshuffler_sampler=dict(type="DistributedSampler"),
# )

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    dataset_scale=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        img_info_prototype="bevdet",
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_infos_val.pkl",
        img_info_prototype="bevdet",
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_infos_val.pkl",
        img_info_prototype="bevdet",
    ),
)


optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
load_from = "ckpts/fcos3d_r101_dcn.pth"
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

checkpoint_config = dict(interval=1)
