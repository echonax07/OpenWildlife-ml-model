_base_ = '../../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

data_root = ''
train_ann_file= '/home/m32patel/scratch/animal_patches/SAVMAP/train/train.json'
# val_ann_file = 'val_2017.json'
test_ann_file = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/coco_split_val.json'


backend_args = None
patch_size = (1024, 1024)
patch_overlap_ratio = 0
merge_iou_thr = 0.5
class_name = ("animal",)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (220, 20, 60),   # beluga whale
])

patch_size = (1024, 1024)

lang_model_name = 'checkpoints/bert/bert-base-uncased'
model = dict(
    sliding_window_inference=dict(enable=True, patch_size=patch_size[0], batch_size=-1,
                                           patch_overlap_ratio=patch_overlap_ratio, merge_nms_type='nms', merge_iou_thr=merge_iou_thr),
    language_model=dict(
    type='BertModel',
    name=lang_model_name,
    max_tokens=256,
    pad_to_max=False,
    use_sub_sentence_represent=True,
    special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
    add_pooling_layer=False,
), bbox_head=dict(num_classes=num_classes))

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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        # batch_size=4,
        type='CocoDataset',
        data_root='',
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file=train_ann_file,
        data_prefix=dict(img='')))


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
    dataset=dict(
        metainfo=metainfo,
        data_root='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images',
        pipeline=test_pipeline,
        ann_file='coco_split_val.json',
        data_prefix=dict(img='')))

# test_dataloader = dict(
#     dataset=dict(
#         metainfo=metainfo,
#         data_root='',
#         pipeline=test_pipeline,
#         ann_file='test_2017.json',
#         data_prefix=dict(img='test/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=test_ann_file)
test_evaluator = dict(ann_file=test_ann_file)

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=5)

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


val_evaluator = dict(ann_file=test_ann_file,
                     outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_val')
test_evaluator = dict(ann_file=test_ann_file,
                     outfile_prefix=f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_test')

pickle_file = f'./work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_finetune_val'

work_dir='work_dir_grounding_dino/finetune/{{fileBasenameNoExtension}}'
load_from = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/t/grouding_dino_swin-t_no_caption/epoch_20.pth'  # noqa