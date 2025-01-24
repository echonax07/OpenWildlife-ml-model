import csv
import json
from collections import defaultdict

def create_csv_from_coco_and_predictions(gt_json_path, pred_json_path, score_threshold, csv_output_path):
    """
    Create a CSV file summarizing ground truth and predictions for specific categories.
    
    Columns: Filename, GT_male_ducks, Pred_Male, GT_female_ducks, Pred_Female_duck, Total_GT, Total_pred
    """
    # Load COCO annotations and predictions
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    # Map image IDs to filenames
    image_id_to_filename = {img['id']: img['file_name'] for img in gt_data['images']}

    # Prepare to count GT annotations by category and image
    gt_counts_by_image = defaultdict(lambda: {"male duck": 0, "female duck": 0, "total": 0})
    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        category_id = ann['category_id']

        if category_id == 1:  # Male duck
            gt_counts_by_image[img_id]["male duck"] += 1
        elif category_id == 4:  # Female duck
            gt_counts_by_image[img_id]["female duck"] += 1

        gt_counts_by_image[img_id]["total"] += 1

    # Prepare to count predictions by category and image, applying score threshold
    pred_counts_by_image = defaultdict(lambda: {"male duck": 0, "female duck": 0, "total": 0})
    for pred in pred_data:
        img_id = pred['image_id']
        score = pred.get('score', 0)
        category_id = pred['category_id']

        if score >= score_threshold:
            if category_id == 1:  # Male duck
                pred_counts_by_image[img_id]["male duck"] += 1
            elif category_id == 0:  # Female duck
                pred_counts_by_image[img_id]["female duck"] += 1

            pred_counts_by_image[img_id]["total"] += 1

    # Create CSV
    with open(csv_output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "GT_male_ducks", "Pred_Male", "GT_female_ducks", "Pred_Female_duck", "Total_GT", "Total_pred"])

        for img_id, filename in image_id_to_filename.items():
            gt_counts = gt_counts_by_image[img_id]
            pred_counts = pred_counts_by_image[img_id]

            writer.writerow([
                filename,
                gt_counts["male duck"],
                pred_counts["male duck"],
                gt_counts["female duck"],
                pred_counts["female duck"],
                gt_counts["total"],
                pred_counts["total"]
            ])

    print(f"CSV file created at {csv_output_path}")

# Example usage
csv_output_path = "output_summary_non_overlapping_patches.csv"


# # Example usage
gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/non_overlapping_slices/patch_annotations.json'

pred_json_path = 'work_dirs/mm_grounding_dino_real_filtered_epoch10/prediction_test.bbox.json'
img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/non_overlapping_slices/'
save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/mm_grounding_dino_real_filtered_epoch10/real_nonover"
score_threshold = 0.4

create_csv_from_coco_and_predictions(gt_json_path, pred_json_path, score_threshold, csv_output_path)
