import json
import numpy as np

def calculate_count_rmse_with_threshold(ground_truth_file, prediction_file, score_threshold=0.3):
    # Load the COCO annotation and prediction files
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    with open(prediction_file, 'r') as f:
        pred_data = json.load(f)

    # Create a mapping of image IDs to image file names from ground truth
    image_id_to_name = {img['id']: img['file_name'] for img in gt_data['images']}

    # Group ground truth annotations by image
    gt_counts = {}
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        gt_counts[image_id] = gt_counts.get(image_id, 0) + 1

    # Group predictions by image, applying score threshold
    pred_counts = {}
    for pred in pred_data:
        if pred['score'] >= score_threshold:
            image_id = pred['image_id']
            pred_counts[image_id] = pred_counts.get(image_id, 0) + 1

    # Calculate RMSE based on counts and store details
    errors = []
    image_details = []
    all_image_ids = set(gt_counts.keys()).union(set(pred_counts.keys()))
    for image_id in all_image_ids:
        gt_count = gt_counts.get(image_id, 0)
        pred_count = pred_counts.get(image_id, 0)
        errors.append((pred_count - gt_count) ** 2)

        # Get image name
        image_name = image_id_to_name.get(image_id, f"Unknown_ID_{image_id}")
        image_details.append({
            'image_name': image_name,
            'gt_count': gt_count,
            'pred_count': pred_count
        })

    rmse = np.sqrt(np.mean(errors))
    return rmse, image_details

# Usage
ground_truth_file = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin/coco/test_viz_grounded.json'  # Replace with your ground truth COCO file path
prediction_file = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/qian_finetune/prediction_mm_grounding_dino_finetune_test.bbox.json'  # Replace with your prediction COCO file path

rmse, image_details = calculate_count_rmse_with_threshold(ground_truth_file, prediction_file, score_threshold=0.4)

print(f"Count-based RMSE: {rmse}")
print("Details for each image:")
for detail in image_details:
    print(f"Image: {detail['image_name']}, GT Count: {detail['gt_count']}, Pred Count: {detail['pred_count']}")