from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def calculate_map(gt_json_path, pred_json_path):
    """
    Calculates mAP metrics including mAP@30 for COCO ground truth and prediction files.

    Parameters:
        gt_json_path (str): Path to the ground truth COCO JSON file.
        pred_json_path (str): Path to the predictions COCO JSON file.

    Returns:
        dict: Dictionary containing mAP@30 along with other standard COCO evaluation metrics.
    """
    # Load ground truth and predictions
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Standard COCO evaluation (IoU=0.50:0.95)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract standard COCO mAP metrics
    metrics = {
        "AP (IoU=0.50:0.95)": coco_eval.stats[0],  # mAP over IoU=0.50:0.95
        "AP (IoU=0.50)": coco_eval.stats[1],       # AP at IoU=0.50
        "AP (IoU=0.75)": coco_eval.stats[2],       # AP at IoU=0.75
        "AP (Small)": coco_eval.stats[3],          # AP for small objects
        "AP (Medium)": coco_eval.stats[4],         # AP for medium objects
        "AP (Large)": coco_eval.stats[5],          # AP for large objects
    }

    # Compute mAP@30 separately
    coco_eval.params.iouThrs = np.array([0.30])  # Only IoU=0.30
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Store mAP@30
    metrics["AP (IoU=0.30)"] = coco_eval.stats[0]  # AP at IoU=0.30
    
    return metrics

# Paths to your JSON files
test_gt_json = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/test.json'
test_pred_json = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/Eikelboom_dataset/prediction_mm_grounding_dino_finetune_test.bbox.json'
# test_pred_json = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json"

print("\nTest mAP Metrics:")
test_metrics = calculate_map(test_gt_json, test_pred_json)
print(test_metrics)
