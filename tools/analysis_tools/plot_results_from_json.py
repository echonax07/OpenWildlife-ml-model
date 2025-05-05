import json
import os
import cv2
import numpy as np
from collections import defaultdict
from icecream import ic
from tqdm import tqdm
import csv

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_iou(box1, box2):
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    # Compute intersection
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)

    # Compute union
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

def draw_bbox(img, bbox, color, label, show_text, thickness=2):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    if show_text:
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def plot_coco_image(gt_path, pred_path, img_folder, save_folder, mode='file', image_name=None, score_threshold=0.5, iou_threshold=0.5, show_text=True):
    # Load ground truth JSON file
    gt_data = load_json(gt_path)

    # Load prediction JSON file if provided
    pred_data = None
    if pred_path:
        pred_data = load_json(pred_path)

    # Create a category ID to name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in gt_data['categories']}

    # Process images
    images_to_process = [image_name] if mode == 'file' and image_name else [img['file_name'] for img in gt_data['images']]

    for img_name in tqdm(images_to_process):
        # Find image ID
        image_id = None
        for img_info in gt_data['images']:
            if img_info['file_name'] == img_name:
                image_id = img_info['id']
                break

        if image_id is None:
            print(f"Image {img_name} not found in ground truth data.")
            continue

        # Load image
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Group GT annotations by category
        gt_anns = defaultdict(list)
        for ann in gt_data['annotations']:
            if ann['image_id'] == image_id:
                gt_anns[ann['category_id']].append(ann)

        # Sort predictions by score if prediction data is provided
        predictions = []
        if pred_data:
            predictions = [pred for pred in pred_data if pred['image_id'] == image_id and pred['score'] >= score_threshold]
            predictions.sort(key=lambda x: x['score'], reverse=True)

        # Draw predictions and ground truth
        for pred in predictions:
            category_id = pred['category_id']
            pred_bbox = pred['bbox']
            pred_category_name = category_id_to_name.get(category_id, "Unknown")

            if category_id in gt_anns:
                max_iou = 0
                max_gt = None
                for gt in gt_anns[category_id]:
                    iou = compute_iou(pred_bbox, gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt = gt
                if max_iou >= iou_threshold:
                    # True Positive
                    color = (0, 255, 0)  # Green
                    label = f"TP: {pred_category_name} ({pred['score']:.2f})"
                    gt_anns[category_id].remove(max_gt)
                else:
                    # False Positive
                    color = (0, 0, 255)  # Red
                    label = f"FP: {pred_category_name} ({pred['score']:.2f})"
            else:
                # False Positive
                color = (0, 0, 255)  # Red
                label = f"FP: {pred_category_name} ({pred['score']:.2f})"
                
            draw_bbox(img, pred_bbox, color, label, show_text)

        # Draw remaining GT boxes
        for category_id, anns in gt_anns.items():
            gt_category_name = category_id_to_name.get(category_id, "Unknown")
            for ann in anns:
                label = f"FN: {gt_category_name}"
                draw_bbox(img, ann['bbox'], (255, 0, 0), label, show_text)

        # Calculate total ground truths and predictions
        original_gt_count = sum(1 for ann in gt_data['annotations'] if ann['image_id'] == image_id)
        pred_count = len(predictions)

        # Add the total number of predictions and ground truths
        summary_text = f"GT: {original_gt_count}, Pred: {pred_count}"
        text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = img.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10

        # Draw background rectangle for the text
        rect_start = (text_x - 5, text_y - text_size[1] - 5)  # Slight padding
        rect_end = (text_x + text_size[0] + 5, text_y + 5)
        cv2.rectangle(img, rect_start, rect_end, (0, 255, 255), -1)  # Yellow background

        # Add the text on top of the rectangle
        cv2.putText(img, summary_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Black text

        # Save the image
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, os.path.basename(img_name))
        cv2.imwrite(save_path, img)

def calculate_rmse(gt_counts, pred_counts):
    """
    Calculate RMSE between the counts of GT and predictions.
    """
    errors = np.array(gt_counts) - np.array(pred_counts)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    return rmse

def analyze_coco_and_predictions(coco_file, pred_file, score_threshold=0.5):
    """
    Analyze COCO annotations and predictions, applying a score threshold for predictions.
    """
    # Load COCO annotations and predictions
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # Map image IDs to filenames
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # Count GT annotations by image
    gt_counts_by_image = {img_id: 0 for img_id in image_id_to_filename.keys()}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        gt_counts_by_image[img_id] += 1

    # Count predictions by image, applying score threshold
    pred_counts_by_image = {img_id: 0 for img_id in image_id_to_filename.keys()}
    for pred in pred_data:
        img_id = pred['image_id']
        score = pred.get('score', 0)  # Default score is 0 if not present
        if img_id in pred_counts_by_image and score >= score_threshold:
            pred_counts_by_image[img_id] += 1

    # Prepare data for RMSE calculation
    gt_counts = []
    pred_counts = []
    results = []

    for img_id, gt_count in gt_counts_by_image.items():
        filename = image_id_to_filename[img_id]
        pred_count = pred_counts_by_image.get(img_id, 0)

        gt_counts.append(gt_count)
        pred_counts.append(pred_count)

        results.append({
            'filename': filename,
            'total_gt': gt_count,
            'total_pred': pred_count,
            'error': gt_count - pred_count
        })

    # Calculate RMSE
    rmse = calculate_rmse(gt_counts, pred_counts)

    # Print results
    for result in results:
        print(f"Filename: {result['filename']}, Total GT: {result['total_gt']}, Total Pred: {result['total_pred']}, Error: {result['error']}")
    print(f"\nAverage RMSE: {rmse:.2f}")
    print(f"\nTotal GT: {sum(gt_counts):.2f}")
    print(f"\nTotal pred: {sum(pred_counts):.2f}")
    print(f"\nPercentage error: {100 *(sum(gt_counts)-sum(pred_counts))/sum(gt_counts):.2f}")


def calculate_tp_fp_fn_ignore_category(coco_file, pred_file, score_threshold=0.5, iou_threshold=0.5, output_csv='results.csv'):
    """
    Calculate TP, FP, FN for each image and save the results in a CSV file.
    """
    # Load COCO annotations and predictions
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # Map image IDs to filenames
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # Prepare results storage
    results = []

    for img_id, filename in image_id_to_filename.items():
        # Get GT annotations for the current image
        gt_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        gt_bboxes = [ann['bbox'] for ann in gt_anns]
        gt_matched = [False] * len(gt_bboxes)  # Track matched GT boxes

        # Get predictions for the current image
        preds = [pred for pred in pred_data if pred['image_id'] == img_id and pred.get('score', 0) >= score_threshold]
        preds.sort(key=lambda x: x['score'], reverse=True)  # Sort predictions by score

        # Calculate TP, FP, FN
        tp = 0
        fp = 0

        for pred in preds:
            pred_bbox = pred['bbox']
            max_iou = 0
            max_idx = -1

            # Compare prediction with GT boxes
            for idx, gt_bbox in enumerate(gt_bboxes):
                if not gt_matched[idx]:  # Only compare with unmatched GT boxes
                    iou = compute_iou(pred_bbox, gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx

            if max_iou >= iou_threshold:
                tp += 1
                gt_matched[max_idx] = True  # Mark GT as matched
            else:
                fp += 1

        # Count remaining unmatched GT boxes as FN
        fn = gt_matched.count(False)

        # Append results for this image
        results.append({
            'filename': filename,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })

    # Save results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'tp', 'fp', 'fn']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")



# # Example usage
gt_json_path = '/home/pc2041/VIP_lab/labelstudio/mmwhale2/work_dirs/Eider Demo V2/debug_temp/train.json'

pred_json_path = '/home/pc2041/VIP_lab/labelstudio/mmwhale2/work_dirs/Eider Demo V2/debug_temp/prediction_val.bbox.json'
img_folder = '/home/pc2041/VIP_lab/labelstudio/mmwhale2/'
save_folder = "/home/pc2041/VIP_lab/labelstudio/mmwhale2/work_dirs/Eider Demo V2/vis"


# # # Example usage
# gt_json_path = '/home/m32patel/scratch/animal_patches/eider_duck_patches/train/train_slice_filtered.json_coco.json'

# pred_json_path = 'work_dirs/eider_faster_rcnn_1024_convnext_T/prediction.bbox.json'
# img_folder = '/home/m32patel/scratch/animal_patches/eider_duck_patches/train/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck_real_sparse_dense/fasterRCNN_ConvNext"
# score_threshold = 0.1

# # Example usage
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/sparse/annotations.json'

# pred_json_path = 'work_dirs/vanilla_faster_rcnn_sparse/prediction_val.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/sparse/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/vanilla_faster_rcnn_sparse"
# score_threshold = 0.4

# # Example usage
# gt_json_path = '/home/m32patel/scratch/animal_patches/eider_duck_patches/train/train_slice_filtered.json_coco.json'

# pred_json_path = 'work_dirs/mm_grounding_dino_real_filtered_epoch10/prediction_val.bbox.json'
# img_folder = '/home/m32patel/scratch/animal_patches/eider_duck_patches/train/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/mm_grounding_dino_real_filtered_epoch10"
# score_threshold = 0.4

# # # Example usage
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/non_overlapping_slices/patch_annotations.json'

# pred_json_path = 'work_dirs/mm_grounding_dino_real_filtered_epoch10/prediction_test.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/non_overlapping_slices/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/mm_grounding_dino_real_filtered_epoch10/real_all"
# score_threshold = 0.4



# # # Example usage
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/coco_test_all_categories.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/eiderduck_predict_on_real/prediction_mm_grounding_dino_finetune_test.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck_real_from_real/"
# score_threshold = 0.3

# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/dense/annotations.json'

# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/eiderduck_finetune_syn_plus_real/prediction_mm_grounding_dino_finetune_test.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/dense/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck_dense_from_mixed/"
# score_threshold = 0.3


# # AED
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/AED/test/test.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/AED/test/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/AED/no_caption_new_split/"
# score_threshold = 0.3

# # aerial seabird westafrica
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Aerial_seabird_westafrica_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Aerial_Seabirds_West_Africa/nocaption/"
# score_threshold = 0.3



# # Eikelboom
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/test.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/Eikelboom_dataset/prediction_mm_grounding_dino_finetune_val.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Eikelboom/zeroshot_new_split_bert/"
# score_threshold = 0.3

# # WAID
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/test.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/WAID_dataset/prediction_mm_grounding_dino_finetune_val.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/WAID_livestock_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/WAID/zeroshot_new_split_bert/"
# score_threshold = 0.3



# # polar bear
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/test_15.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/polar_bear/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/polar_bear_finetune/prediction_mm_grounding_dino_finetune_val.bbox.json
# # '
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/polar_bear/zeroshot_new_split_bert/"
# score_threshold = 0.25


# # sea lion
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_sea_lion_blackout/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_sealion_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_sea_lion_blackout/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/NOAA_sea_lion_blackout/viz/"
# score_threshold = 0.3


# # penguin
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_penguins/viz/"
# score_threshold = 0.3

# # turtle
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/turtle/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/turtle_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/turtle/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/turtle/viz/"
# score_threshold = 0.3

# # DFOW15
# gt_json_path = '/home/m32patel/projects/def-dclausi/whale/merged/test/test_2015.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2015_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/def-dclausi/whale/merged/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Beluga_2015_dataset/viz/"
# score_threshold = 0.3


# # Narval
# gt_json_path = '/home/m32patel/projects/def-dclausi/whale/merged/test/test_ES_2016.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Narwhal_2016_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/def-dclausi/whale/merged/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Narwhal_2016_dataset/zeroshot_new_split_bert/"
# score_threshold = 0.3


# # DFOW23
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/coco_iter12345_val_viz_grounded.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/DFO_Whale23/prediction_mm_grounding_dino_viz_caption_new_split.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/DFO_Whale23/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'

# img_folder = ''
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/DFOW23/nocaption/zeroshot_new_split_bert/"
# score_threshold = 0.3



# # NOAA Seal
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_artic_seal_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/NOAA_arctic_seals/zeroshot_new_split_bert/"
# score_threshold = 0.3



# # tern bioarxiv
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/tern_bioarxiv/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/terns_bioarxiv/zeroshot_new_split_bert/"
# score_threshold = 0.3


# # Michigan gull
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/test.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/michigan_od_dataset/prediction_mm_grounding_dino_finetune_test.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/michigan_od_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/'
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_michigan/finetune_new_split_bert/"
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_michigan/zeroshot_new_split_bert/"
# score_threshold = 0.1


# # birds_monash
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_monash/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/monash_od_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_monash/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_monash/viz/"
# score_threshold = 0.25

# # aerial livestock
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/test.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_viz_caption_new_split.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/'
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/aerial_livestock/zeroshot_new_split_bert/"
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/aerial_livestock/zeroshot_new_split_bert_viz/"
# score_threshold = 0.1

# # savmap
# gt_json_path =  '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/coco_split_val.json'
# pred_json_path = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/SAVMAP/prediction_mm_grounding_dino_finetune_val.bbox.json"

# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/SAVMAP_test/finetune_nocaption/"
# score_threshold = 0.1



# # # palmyra
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_palmyra/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/palmyra_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_palmyra/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_palmyra/nocaption/"
# score_threshold = 0.3


# # # birds_qian
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin/coco/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/qian_od_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin/coco'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_qian_penguin/viz/"
# score_threshold = 0.3



# # virunga garamba
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba/prediction_mm_grounding_dino_finetune_test.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Virunga_Garamba/nocaption_finetune/"
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Virunga_Garamba/nocaption/"

# score_threshold = 0.1


# /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/120.jpg
# For single image
# plot_coco_image(gt_json_path, pred_json_path, img_folder, save_folder, mode='file', image_name='120.jpg', score_threshold=score_threshold, iou_threshold=0.1, show_text=True)

# # # For all images
# analyze_coco_and_predictions(gt_json_path, pred_json_path, score_threshold)

# # AED
# gt_json_path = '/home/pc2041/VIP_lab/mmwhale2/data.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# pred_json_path = ''
# img_folder = ''
# save_folder = "resutl_viz/"

score_threshold = 0.3
iou_threshold = 0.2
plot_coco_image(gt_json_path, pred_json_path, img_folder, save_folder, mode='all', score_threshold=score_threshold, iou_threshold=iou_threshold, show_text=False)
# plot_coco_image(gt_json_path, None, img_folder, save_folder, mode='all', score_threshold=score_threshold, iou_threshold=iou_threshold, show_text=False)

# calculate_tp_fp_fn_ignore_category(
#     gt_json_path,
#     pred_json_path, 
#    score_threshold,
#     iou_threshold=0.5,
#     output_csv='tp_fp_fn_results.csv'
# )