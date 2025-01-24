_base_ = [
    '../_base_/datasets/eider_ducks_sliced_filtered.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]
batch_size = 8

train_dataloader = dict(
    batch_size=batch_size,)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

vis_backends = [
    dict(
        type='LocalVisBackend'
    ),
    # dict(
    #     type='WandbVisBackend',
    #     init_kwargs=dict(
    #         entity='mmwhale',
    #         project='animal_slicing',
    #         name='{{fileBasenameNoExtension}}',
    #     ),
    #     define_metric_cfg=None,
    #     commit=True,
    #     log_code_name=None,
    #     watch_kwargs=None
    # ),
]

visualizer = dict(vis_backends=vis_backends)
# since fileBasenameNoExtension refers to the current config file we need to keep that field in the final(leaf) config file
val_evaluator = dict(
    outfile_prefix=f'./work_dirs/{{fileBasenameNoExtension}}/prediction')
test_evaluator = dict(
    outfile_prefix=f'./work_dirs/{{fileBasenameNoExtension}}/prediction')
pickle_file = f'./work_dirs/{{fileBasenameNoExtension}}/prediction'
pr_curve = f'./work_dirs/{{fileBasenameNoExtension}}/pr_curve.jpg'
confusion_matrix = f'./work_dirs/{{fileBasenameNoExtension}}/confusion_matrix.jpg'

# base_batch_size = (2 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# load_from = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dirs/deformable_DETR_1024/best_coco_bbox_mAP_iter_20000.pth'

# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/convnext-tiny_32xb128-noema_in1k_20221208-5d4509c7.pth',
            prefix='backbone.')),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.75, 1.0, 1.5],
            strides=[2, 4, 8, 16, 32],),
            # type='AnchorGenerator',
            # scales=[8],
            # ratios=[0.5, 1.0, 2.0],
            # strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=5000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=1024,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            # nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            # nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=1500)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=10)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

load_from = "checkpoints/faster-rcnn_r50_fpn_1x_coco.pth"