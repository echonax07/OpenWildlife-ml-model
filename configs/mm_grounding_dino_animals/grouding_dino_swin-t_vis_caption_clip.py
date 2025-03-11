_base_ = '../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

lang_model_name = 'checkpoints/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'

patch_size = (1024, 1024)

# big pipeline
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=patch_size, crop_type='absolute',
         allow_negative_crop=True, recompute_bbox=True, bbox_clip_border=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        # update_pad_shape=False,
        skip_img_without_anno=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(3, 3)),
    dict(
        type='RandomSamplingNegPos_with_caption',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text', 'label_map' ,
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

# train pipeline for small images
train_pipeline2 = [
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
        type='RandomSamplingNegPos_with_caption',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text', 'label_map',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]


# BIRDS
# B1
# Big pipeline
Aerial_seabird_westafrica_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/Aerial_Seabirds_West_Africa/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B2
# Big pipeline
birds_izembek_lagoon_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/birds_Izembek_Lagoon_Waterfowl/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B3
# Big pipeline
michigan_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B4
# Big pipeline
monash_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_monash',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B5
# Big pipeline
new_mexico_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_newmexico',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B6
# Big pipeline
palmyra_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_palmyra',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B7
# Normal pipeline
penguins_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline2,
    return_classes=True,
    backend_args=None,
)
# B8
# Normal pipeline
pfeifer_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_pfeifer',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline2,
    return_classes=True,
    backend_args=None,
)
# B9
# Normal pipeline
seabirdwatch_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_seabirdwatch',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline2,
    return_classes=True,
    backend_args=None,
)
# B10
# Big pipeline
birds_poland_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_poland',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# B11
# Normal pipeline
qian_od_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='coco'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline2,
    return_classes=True,
    backend_args=None,
)

# LAND ANIMALS
# L1
# Big pipeline
aerial_livestock_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/Aerial-livestock-dataset/train/',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

# # Big pipeline
# SAVMAP_dataset = dict(
#     type='ODVGCaptionDataset',
#     data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin',
#     ann_file='.json',
#     label_map_file='o365v1_label_map.json',
#     data_prefix=dict(img=''),
#     filter_cfg=dict(filter_empty_gt=False),
#     pipeline=_base_.train_pipeline,
#     return_classes=True,
#     backend_args=None,
# )

# L2
# Big pipeline
WAID_livestock_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline2,
    return_classes=True,
    backend_args=None,
)

# L3
# Big pipeline
AED_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/AED/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# L4
# Big pipeline
Eikelboom_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/Eikelboom/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

# Ocean Animals
# O1
# Big pipeline
NOAA_sealion_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/NOAA_sea_lion_blackout/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# O2
# Big pipeline
turtle_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/turtle/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# O3
# Big pipeline
NOAA_artic_seal_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/NOAA_arctic_seal/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# O4
# Big pipeline
Beluga_2014_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/2014_Beluga/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

# O5
# Big pipeline
Beluga_2015_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/2015_Beluga/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# O6
# Big pipeline
Narwhal_2016_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/2016_Narwhal/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
# O7
# Big pipeline
Beluga_2017_dataset = dict(
    type='ODVGCaptionDataset',
    data_root='/home/m32patel/scratch/animal_patches/2017_Beluga/train',
    ann_file='train_od_vis_desc_grounded.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)
train_dataloader = dict(
    batch_size=2,
    sampler=dict(
        _delete_=True,
        type='CustomSampleSizeSampler',
        # dataset_size=[-1, 10000, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        dataset_size=[-1]),
        
    dataset=dict(datasets=[
        # Aerial_seabird_westafrica_od_dataset,
        # birds_izembek_lagoon_od_dataset,
        # michigan_od_dataset,
        # # monash_od_dataset,
        # # new_mexico_od_dataset,
        # # palmyra_od_dataset,
        # # penguins_od_dataset,
        # pfeifer_od_dataset,
        # # seabirdwatch_od_dataset,
        # birds_poland_od_dataset,
        qian_od_dataset,
        # aerial_livestock_dataset,
        # WAID_livestock_dataset,
        # AED_dataset,
        # Eikelboom_dataset,
        # NOAA_sealion_dataset,
        # turtle_dataset,
        # NOAA_artic_seal_dataset,
        # Beluga_2014_dataset,
        # Beluga_2015_dataset,
        # Narwhal_2016_dataset
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
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins',
        ann_file='test.json',
        data_prefix=dict(img=''),
        test_mode=True,))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins' + '/test.json',
    metric='bbox',
    format_only=False,)
test_evaluator = val_evaluator

# Model related settings
model = dict(language_model=dict(
    _delete_ = True, 
    type='CLIPModel',
    name=lang_model_name,
    max_tokens=77,
    pad_to_max=True,
    use_sub_sentence_represent=True,
    special_tokens_list=[
        '<|startoftext|>',
        '<|endoftext|>',
        '.',
        '?'
    ],
    num_layers_of_embedded=4,  # Average last 4 layers
    use_checkpoint=False,  # Enable if using gradient checkpointing
))

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

# visualizer = dict(vis_backends=vis_backends)

work_dir='work_dir_grounding_dino/{{fileBasenameNoExtension}}'

load_from = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
