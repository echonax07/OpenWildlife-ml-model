_base_ = '../../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

lang_model_name = 'checkpoints/bert/bert-base-uncased'

patch_size = (1024, 1024)
patch_overlap_ratio = 0.25

merge_iou_thr = 0.5
model = dict(sliding_window_inference=dict(enable=True, patch_size=patch_size[0], batch_size=-1,
                                           slice_batch_size=24,
                                           patch_overlap_ratio=patch_overlap_ratio, merge_nms_type='nms', merge_iou_thr=merge_iou_thr),
            )
