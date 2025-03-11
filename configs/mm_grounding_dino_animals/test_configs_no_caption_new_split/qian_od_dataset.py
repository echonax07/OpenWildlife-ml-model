
_base_ = '../grouding_dino_swin-t_finetune_all.py'

data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin/coco'
ann_file = 'test.json'
class_name = ('penguin', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))


val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + ann_file,
                    outfile_prefix=f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_nocaption_new_split')
test_evaluator = val_evaluator
pickle_file = f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_nocaption_new_split'
