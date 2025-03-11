
_base_ = '../grouding_dino_swin-t_finetune_all.py'
lang_model_name = 'checkpoints/bert/bert-base-uncased'
data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_seabirdwatch'
ann_file = 'test_viz_grounded.json'
class_name = ('bird', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomSamplingNegPos_with_caption',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        mode='test',
        max_tokens=256),
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
        type = 'CocoDatasetWithCaption',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + ann_file,
                    outfile_prefix=f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_viz_caption_new_split')
test_evaluator = val_evaluator
pickle_file = f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_viz_caption_new_split'
