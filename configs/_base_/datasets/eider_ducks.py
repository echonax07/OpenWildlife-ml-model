# dataset settings
# dataset_type = 'WhaleDataset'
dataset_type = 'CocoDataset'

# Sliced data_root 
# This is the path where patches are saved!
data_root_slice = '/home/m32patel/scratch/eider_patches/1024/'
# whole images data_root
data_root_whole = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export'

class_name = ("Female duck","Male duck","Ice","Juvenile duck","duck",)
# num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (220, 20, 60),   # Female duck
    (34, 139, 34),   # Male duck
    (30, 144, 255),  # Ice
    (205, 133, 63),  # Juvenile duck
    (0, 191, 255),   # duck
])

backend_args = None
# scale = (7360, 4912) # this scale pertains to image scale
ratio = 1
batch_size = 8

slice_configuration = dict(enable=True,
         slice_height=1024, slice_width=1024, overlap_height_ratio=0, overlap_width_ratio=0, save_only_positive_slices=True)


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
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(1024,1024), crop_type='absolute',
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
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    # dict(type='Resize', scale=ratio, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


dataset_real=dict(
        type='CocoDataset',
        # batch_size=4,
        data_root='/home/m32patel/scratch/animal_patches/eider_duck_patches/train',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='train_slice_filtered.json_coco.json',
        data_prefix=dict(img=''))

dataset_sparse=dict(
        type='CocoDataset',
        # batch_size=4,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/sparse',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations.json',
        data_prefix=dict(img=''))

dataset_dense=dict(
        type='CocoDataset',
        # batch_size=4,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/dense',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations.json',
        data_prefix=dict(img=''))

dataset_mixed=dict(
        type='CocoDataset',
        # batch_size=4,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/mixed',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations.json',
        data_prefix=dict(img=''))

dataset_clustered=dict(
        type='CocoDataset',
        # batch_size=4,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/clustered',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations.json',
        data_prefix=dict(img=''))

dataset_circular=dict(
        type='CocoDataset',
        # batch_size=4,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/circular',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations.json',
        data_prefix=dict(img=''))


dataset_type = 'CocoDataset'
dataset_combined=dict(
            type='ConcatDataset',
            ignore_keys=['dataset_type'],
            # datasets=[ dataset_real, dataset_sparse,dataset_dense, dataset_mixed, dataset_clustered, dataset_circular
            # ]
            datasets=[ dataset_real, dataset_sparse, dataset_dense, dataset_circular
            ]
            )


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='BalancedInfiniteSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_whole,
        metainfo=metainfo,
        ann_file='train/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_whole,
        metainfo=metainfo,
        ann_file='val/val_positives.json',
        # ann_file='train.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_whole,
        metainfo=metainfo,
        ann_file='test/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_whole + 'val/val_positives.json',
    metric='bbox',
    format_only=False,
    proposal_nums=(100, 1000, 2000),
    outfile_prefix=f'./work_dirs/{{fileBasenameNoExtension}}/prediction_val',
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_whole + 'test/test.json',
    metric='bbox',
    format_only=True,
    proposal_nums=(100, 1000, 2000),
    outfile_prefix=f'./work_dirs/{{fileBasenameNoExtension}}/prediction',
    backend_args=backend_args)

pickle_file = f'./work_dirs/{{fileBasenameNoExtension}}/prediction'
pr_curve = f'./work_dirs/{{fileBasenameNoExtension}}/pr_curve.jpg'
confusion_matrix = f'./work_dirs/{{fileBasenameNoExtension}}/confusion_matrix.jpg'
# base_batch_size = (2 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

custom_imports = dict(
    imports=['mmdet.datasets.samplers.balanced_sampler'],
    allow_failed_imports=False)
