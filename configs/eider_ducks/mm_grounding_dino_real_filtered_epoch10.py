_base_ = '../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

# data_root = '/home/m32patel/scratch/animal_patches/eider_duck_patches/train'

data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/non_overlapping_slices/'

class_name = ("female duck","male duck","Ice","Juvenile duck","duck",)
train_ann_file = 'train_slice_filtered.json_coco.json'
test_ann_file = 'patch_annotations.json'
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (220, 20, 60),   # Female duck
    (34, 139, 34),   # Male duck
    (30, 144, 255),  # Ice
    (205, 133, 63),  # Juvenile duck
    (0, 191, 255),   # duck
])

train_patch_size = (512, 512)
test_patch_size = (1024,1024)
patch_overlap_ratio = 0
merge_iou_thr = 1

slice_configuration = dict(enable=True,
         slice_height=train_patch_size[0], slice_width=train_patch_size[1], overlap_height_ratio=0, overlap_width_ratio=0,mix_slices_with_full_images=False, save_only_positive_slices=False)

model = dict(
    sliding_window_inference = dict(enable=True, patch_size=test_patch_size[0], batch_size=1,  slice_batch_size = 4,
                                patch_overlap_ratio=patch_overlap_ratio, merge_nms_type='nms', merge_iou_thr=merge_iou_thr),
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        frozen_stages=-1,
        init_cfg=None,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=True, log_scale='auto', max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=5,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    language_model=dict(
        add_pooling_layer=False,
        max_tokens=256,
        name='checkpoints/bert/bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=2000,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=2000),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO',
    with_box_refine=True)


train_pipeline2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file=train_ann_file,
        data_prefix=dict(img='')))

# big pipeline
test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    # dict(
    #     type='FixScaleResize',
    #     scale=(800, 1333),
    #     keep_ratio=True,
    #     backend='pillow'),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    num_workers=8,
    batch_size=32,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        pipeline=test_pipeline,
        ann_file=test_ann_file,
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + test_ann_file,outfile_prefix=f'./work_dirs/{{fileBasenameNoExtension}}/prediction_val',
)
test_evaluator = val_evaluator

test_evaluator = dict(ann_file=data_root + '/' + test_ann_file,outfile_prefix=f'./work_dirs/{{fileBasenameNoExtension}}/prediction_test')

max_epoch = 50

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=10, save_best=None),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=10)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))

# work_dir = 'work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}'
load_from = 'work_dir_grounding_dino/grouding_dino_swin-t_vis_caption/epoch_10.pth'  # noqapy
# load_from = 'work_dirs/mm_grounding_dino_clustered_epoch10/epoch_10.pth'  # noqapy