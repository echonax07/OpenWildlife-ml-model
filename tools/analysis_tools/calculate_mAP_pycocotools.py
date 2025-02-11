from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_map(gt_json_path, pred_json_path):
    """
    Calculates the mAP for COCO ground truth and prediction files.

    Parameters:
        gt_json_path (str): Path to the ground truth COCO JSON file.
        pred_json_path (str): Path to the predictions COCO JSON file.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Load ground truth and predictions
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.maxDets = [100, 1000, 10000]
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract mAP results
    metrics = {
        "AP (IoU=0.50:0.95)": coco_eval.stats[0],  # mAP for IoU=0.50:0.95
        "AP (IoU=0.50)": coco_eval.stats[1],       # mAP for IoU=0.50
        "AP (IoU=0.75)": coco_eval.stats[2],       # mAP for IoU=0.75
        "AP (Small)": coco_eval.stats[3],          # mAP for small objects
        "AP (Medium)": coco_eval.stats[4],         # mAP for medium objects
        "AP (Large)": coco_eval.stats[5],          # mAP for large objects
    }
    
    return metrics

# Paths to your JSON files
test_gt_json = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json'
test_pred_json = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba_try3/prediction_full_patches.json"


print("\nTest mAP:")
test_metrics = calculate_map(test_gt_json, test_pred_json)
print(test_metrics)
