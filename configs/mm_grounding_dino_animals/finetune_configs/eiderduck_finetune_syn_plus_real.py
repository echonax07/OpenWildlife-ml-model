
_base_ = '../../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

# data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project'
data_root = '/home/m32patel/scratch/animal_patches/eider_duck_patches/train'
# data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/dense'


class_name = ('female duck', 'male duck', 'Ice', 'Juvenile', 'Unknown',)
# train_ann_file = ''
# test_ann_file = 'coco_test.json'
test_ann_file='train_slice_filtered.json_coco.json'
# test_ann_file = 'annotations.json'

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (220, 20, 60),   # Alcelaphinae
    (34, 139, 34),   # Buffalo
    (30, 144, 255),  # Kob
    (205, 133, 63),  # Warthog
    (0, 191, 255),   # Waterbuck
])

backend_args = None
model = dict(bbox_head=dict(num_classes=num_classes),
    num_queries=2000,
    test_cfg=dict(max_per_img=2000),
    )

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
    batch_size=2,
    dataset=(dataset_combined)
    )



val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + test_ann_file,
                     outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_val')
# val_evaluator = dict(ann_file=test_ann_file)

test_evaluator = val_evaluator

test_evaluator = dict(ann_file=data_root + '/' + test_ann_file,
                     outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_test')


# test_evaluator = dict(ann_file=test_ann_file,
                    #  outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_test')

max_epoch = 200

default_hooks = dict(
    checkpoint=dict(interval=20, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=20)

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
    optimizer=dict(lr=0.00002),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))

work_dir = 'work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}'
load_from = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/grouding_dino_swin-t_vis_caption/epoch_10.pth'  # noqa
