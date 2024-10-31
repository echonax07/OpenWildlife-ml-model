
_base_ = '../grouding_dino_swin-t_finetune_all.py'

data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test'
ann_file = 'test.json'
class_name = ('sheep', 'cattle', 'seal', 'camelus', 'kiang', 'zebra')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (255, 0, 0),      # sheep - Red
    (0, 255, 0),      # cattle - Green
    (0, 0, 255),      # seal - Blue
    (255, 165, 0),    # camelus - Orange
    (128, 0, 128),    # kiang - Purple
    (0, 255, 255)     # zebra - Cyan
])

model = dict(bbox_head=dict(num_classes=num_classes))


val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + ann_file,
                    outfile_prefix=f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino')
test_evaluator = val_evaluator
pickle_file = f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino'
