
_base_ = '../grouding_dino_swin-t_no_caption_new_split_clip.py'

data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/AED/test'
ann_file = 'test.json'
class_name = ('elephant', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])


backend_args = None
patch_size = (1024, 1024)
patch_overlap_ratio = 0.5
merge_iou_thr = 0.5
model = dict(sliding_window_inference=dict(enable=True, patch_size=patch_size[0], batch_size=-1, slice_batch_size=36,
                                           patch_overlap_ratio=patch_overlap_ratio, merge_nms_type='nms', merge_iou_thr=merge_iou_thr),
             bbox_head=dict(num_classes=num_classes))

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
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + ann_file,
                     outfile_prefix=f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_nocaption_new_split_clip')
test_evaluator = val_evaluator
pickle_file = f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_nocaption_new_split_clip'
