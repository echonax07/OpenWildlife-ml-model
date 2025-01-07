_base_ = [
    '../_base_/datasets/DFO_whale.py',
    '../_base_/schedules/schedule_40000_iter.py',
    '../_base_/models/faster-rcnn_r50_fpn.py'
]
batch_size = 8

train_dataloader = dict(
    batch_size=batch_size,)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]))
vis_backends = [
    dict(
        type='LocalVisBackend'
    ),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='mmwhale',
            project='animal_slicing',
            name='{{fileBasenameNoExtension}}',
        ),
        define_metric_cfg=None,
        commit=True,
        log_code_name=None,
        watch_kwargs=None
    ),
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

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ))