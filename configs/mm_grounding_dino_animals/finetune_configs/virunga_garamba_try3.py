_base_ = '../../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'


data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba'
train_ann_file = 'groundtruth/json/sub_frames/train_subframes_A_B_E_K_WH_WB.json'
val_ann_file = 'patch_annotations.json'
test_ann_file = 'patch_annotations.json'


backend_args = None
patch_size = (1024, 1024)
patch_overlap_ratio = 0.5
merge_iou_thr = 0.5
class_name = ("Alcelaphinae",
              "Buffalo",
              "Kob",
              "Warthog",
              "Waterbuck",
              "Elephant", )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (220, 20, 60),   # Alcelaphinae
    (34, 139, 34),   # Buffalo
    (30, 144, 255),  # Kob
    (205, 133, 63),  # Warthog
    (0, 191, 255),   # Waterbuck
    (128, 128, 128)  # Elephant
])

patch_size = (1024, 1024)

lang_model_name = 'checkpoints/bert/bert-base-uncased'


model = dict(bbox_head=dict(num_classes=num_classes))


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
        # batch_size=4,
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file=train_ann_file,
        data_prefix=dict(img='train_subframes')))

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        metainfo=metainfo,
        data_root='/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping/',
        ann_file=test_ann_file,
        data_prefix=dict(img='')))

test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        metainfo=metainfo,
        data_root='/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping/',
        ann_file=test_ann_file,
        data_prefix=dict(img='')))

val_evaluator = dict(ann_file='/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping/' + '/' + test_ann_file,
                      outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_test')
test_evaluator = dict(ann_file='/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping/' + '/' + test_ann_file,
                      outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_test')

max_epoch = 100

# save_best='auto'
default_hooks = dict(
    checkpoint=dict(interval=10),
    logger=dict(type='LoggerHook', interval=50))
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
    optimizer=dict(lr=0.00002),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))


pickle_file = f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_val'

work_dir = 'work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}'
load_from = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/grouding_dino_swin-t_no_caption/epoch_20.pth'  # noqa 
