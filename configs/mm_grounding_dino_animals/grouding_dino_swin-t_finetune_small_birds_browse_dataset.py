_base_ = '../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

lang_model_name = 'checkpoints/bert/bert-base-uncased'
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), 
                            (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)
                            ],
                    keep_ratio=True)
            ],
            # [
            #     dict(
            #         type='RandomChoiceResize',
            #         # The radio of all image in train dataset < 7
            #         # follow the original implement
            #         scales=[(400, 4200), (500, 4200), (600, 4200)],
            #         keep_ratio=True),
            #     dict(
            #         type='RandomCrop',
            #         crop_type='absolute_range',
            #         crop_size=(384, 600),
            #         allow_negative_crop=True),
            #     dict(
            #         type='RandomChoiceResize',
            #         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
            #                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
            #                 (736, 1333), (768, 1333), (800, 1333)],
            #         keep_ratio=True)
            # ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]


hayes_od_dataset = dict(
    type='ODVGDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_hayes_ALREADY_USED',
    ann_file='train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

michigan_od_dataset = dict(
    type='ODVGDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan',
    ann_file='train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
penguins_od_dataset = dict(
    type='ODVGDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins',
    ann_file='train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

pfeifer_od_dataset = dict(
    type='ODVGDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_pfeifer',
    ann_file='train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

seabirdwatch_od_dataset = dict(
    type='ODVGDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_seabirdwatch',
    ann_file='train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

v3d_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        # change this
        label_map_file='data/V3Det/annotations/v3det_2023_v1_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]


train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type='CustomSampleSizeSampler',
        dataset_size=[-1, -1, -1, -1, -1]),
    dataset=dict(datasets=[
        hayes_od_dataset, michigan_od_dataset, penguins_od_dataset, pfeifer_od_dataset,
        seabirdwatch_od_dataset
    ]))

class_name = ('penguin', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Qian_penguin/coco',
        ann_file='test.json',
        data_prefix=dict(img=''),
        test_mode=True,))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Qian_penguin/coco' + '/test.json',
    metric='bbox',
    format_only=False,)
test_evaluator = val_evaluator

lang_model_name = 'checkpoints/bert/bert-base-uncased'
# Model related settings
model = dict(language_model=dict(
    type='BertModel',
    name=lang_model_name))


optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.00004, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1))),
    type='OptimWrapper')

train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)

vis_backends = [
    dict(
        type='LocalVisBackend'
    ),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='mmwhale',
            project='MM_grounding_DINO',
            name='{{fileBasenameNoExtension}}',
        ),
        define_metric_cfg=None,
        commit=True,
        log_code_name=None,
        watch_kwargs=None
    ),
]


load_from = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
