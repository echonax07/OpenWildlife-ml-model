import json
import os
import cv2
import numpy as np
from collections import defaultdict
from icecream import ic
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

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

def plot_coco_image(gt_path, pred_path, img_folder, save_folder, mode='file', image_name=None, score_threshold=0.5, iou_threshold=0.5, show_text=True, plot_images=True, suffix=None):
    # Load JSON files
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    # Create a category ID to name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in gt_data['categories']}

    # Process images
    images_to_process = [image_name] if mode == 'file' and image_name else [img['file_name'] for img in gt_data['images']]

    # Initialize a list to store results
    results = []
    score_of_tp_without_class_confusion = []
    score_of_fp_without_class_confusion = []
    

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

        # Get all ground truth annotations for this image
        gt_anns = [ann for ann in gt_data['annotations'] if ann['image_id'] == image_id]

        # Get all predictions for this image (filtered by score threshold)
        predictions = [pred for pred in pred_data if pred['image_id'] == image_id and pred['score'] >= score_threshold]
        predictions.sort(key=lambda x: x['score'], reverse=True)

        # Initialize counters for TP, FP, FN (with and without class confusion)
        tp_with_class = 0
        fp_with_class = 0
        fn_with_class = 0

        tp_without_class = 0
        fp_without_class = 0
        fn_without_class = 0

        # Track which ground truth boxes have been matched (with and without class confusion)
        matched_gt_indices_with_class = set()
        matched_gt_indices_without_class = set()

        # Calculate TP, FP, FN WITH class confusion
        for pred in predictions:
            pred_bbox = pred['bbox']
            pred_category_id = pred['category_id']
            max_iou = 0
            best_gt_index = -1

            # Find the ground truth box with the highest IoU (same class only)
            for i, gt in enumerate(gt_anns):
                if i in matched_gt_indices_with_class:
                    continue  # Skip already matched ground truth boxes
                if gt['category_id'] != pred_category_id:
                    continue  # Skip if class labels don't match
                iou = compute_iou(pred_bbox, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_gt_index = i

            # Check if the best IoU meets the threshold
            if max_iou >= iou_threshold:
                tp_with_class += 1
                matched_gt_indices_with_class.add(best_gt_index)  # Mark this ground truth box as matched
            else:
                fp_with_class += 1

        # Calculate FN WITH class confusion
        fn_with_class = len(gt_anns) - len(matched_gt_indices_with_class)

        # Calculate TP, FP, FN WITHOUT class confusion
        for pred in predictions:
            pred_bbox = pred['bbox']
            score = pred['score']
            max_iou = 0
            best_gt_index = -1

            # Find the ground truth box with the highest IoU (regardless of class)
            for i, gt in enumerate(gt_anns):
                if i in matched_gt_indices_without_class:
                    continue  # Skip already matched ground truth boxes
                iou = compute_iou(pred_bbox, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_gt_index = i

            # Check if the best IoU meets the threshold
            if max_iou >= iou_threshold:
                tp_without_class += 1
                score_of_tp_without_class_confusion.append(score)
                matched_gt_indices_without_class.add(best_gt_index)  # Mark this ground truth box as matched
            else:
                fp_without_class += 1
                score_of_fp_without_class_confusion.append(score)

        # Calculate FN WITHOUT class confusion
        fn_without_class = len(gt_anns) - len(matched_gt_indices_without_class)

        # Draw predictions and ground truth (if plot_images is True)
        if plot_images:
            # Draw predictions
            for pred in predictions:
                pred_bbox = pred['bbox']
                pred_category_name = category_id_to_name.get(pred['category_id'], "Unknown")
                color = (0, 255, 0) if pred in matched_gt_indices_with_class else (0, 0, 255)  # Green for TP, Red for FP
                label = f"TP: {pred_category_name}" if pred in matched_gt_indices_with_class else f"FP: {pred_category_name}"
                draw_bbox(img, pred_bbox, color, label, show_text)

            # Draw unmatched ground truth boxes (FN)
            for i, gt in enumerate(gt_anns):
                if i not in matched_gt_indices_with_class:
                    gt_category_name = category_id_to_name.get(gt['category_id'], "Unknown")
                    label = f"FN: {gt_category_name}"
                    draw_bbox(img, gt['bbox'], (255, 0, 0), label, show_text)

            # Add summary text
            summary_text = f"GT: {len(gt_anns)}, Pred: {len(predictions)}"
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

        # Append the results for this image
        results.append({
            'filename': img_name,
            'tp_with_class': tp_with_class,
            'fp_with_class': fp_with_class,
            'fn_with_class': fn_with_class,
            'tp_without_class': tp_without_class,
            'fp_without_class': fp_without_class,
            'fn_without_class': fn_without_class
        })

    # Write results to CSV
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, f'results_{suffix}.csv')
    whisker_path = os.path.join(save_folder, f'whisker_{suffix}.png')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'filename',
            'tp_with_class', 'fp_with_class', 'fn_with_class',
            'tp_without_class', 'fp_without_class', 'fn_without_class'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Calculate total TP, FP, FN (with and without class confusion)
    total_tp_with_class = sum(result['tp_with_class'] for result in results)
    total_fp_with_class = sum(result['fp_with_class'] for result in results)
    total_fn_with_class = sum(result['fn_with_class'] for result in results)

    total_tp_without_class = sum(result['tp_without_class'] for result in results)
    total_fp_without_class = sum(result['fp_without_class'] for result in results)
    total_fn_without_class = sum(result['fn_without_class'] for result in results)

    # Append totals to the CSV file
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])  # Add an empty row for separation
        writer.writerow(['Total (with class confusion)', total_tp_with_class, total_fp_with_class, total_fn_with_class])
        writer.writerow(['Total (without class confusion)', total_tp_without_class, total_fp_without_class, total_fn_without_class])

    print(f"Results saved to {csv_path}")
    
    # Create the box plot
     # Create a single figure with two box plots
    plt.figure(figsize=(2, 3))  # Adjust width to fit two plots
    plt.boxplot([score_of_tp_without_class_confusion, score_of_fp_without_class_confusion], vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                positions=[1, 2])  # Two positions for TP and FP
    
    # Customize the plot with larger fonts
    plt.title("")  # Remove the title to save space
    plt.xticks([1, 2], ["TP", "FP"], fontsize=14)  # Increased font size for x-axis labels
    plt.yticks(np.linspace(0, 1, 6), fontsize=14)  # Increased font size for y-axis ticks
    plt.ylim(0, 1)  # Ensure y-axis range is fixed from 0 to 1
    plt.ylabel("Score", fontsize=12)  # Larger font size for y-axis label
    plt.tight_layout()

    # plt.boxplot(score_of_tp_without_class_confusion, showfliers=False, vert='False')
    # # Add title and labels
    # plt.title('Box and Whisker Plot')
    # plt.ylabel('Values')

    # Show the plot
    plt.savefig(whisker_path)
    
    # Save TP and FP scores to CSV
    scores_csv_path = os.path.join(save_folder, f'tp_fp_confidence_scores_{suffix}.csv')

    with open(scores_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["score_of_tp_without_class_confusion", "score_of_fp_without_class_confusion"])

        # Write row-wise, ensuring both lists are correctly aligned
        max_length = max(len(score_of_tp_without_class_confusion), len(score_of_fp_without_class_confusion))
        for i in range(max_length):
            tp_score = score_of_tp_without_class_confusion[i] if i < len(score_of_tp_without_class_confusion) else ""
            fp_score = score_of_fp_without_class_confusion[i] if i < len(score_of_fp_without_class_confusion) else ""
            writer.writerow([tp_score, fp_score])
        print(f"Scores saved to {scores_csv_path}")

def calculate_rmse(gt_counts, pred_counts):
    """
    Calculate RMSE between the counts of GT and predictions.
    """
    errors = np.array(gt_counts) - np.array(pred_counts)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    return rmse

def analyze_coco_and_predictions(coco_file, pred_file, score_threshold=0.5, suffix=None):
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

# # # Example usage
# gt_json_path = '/home/m32patel/projects/def-dclausi/whale/merged/test/test.json'

# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dirs/DFO_whale_faster_rcnn_4096_convnext_T/prediction.bbox.json'
# img_folder = '/home/m32patel/projects/def-dclausi/whale/merged/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/whale_ConvNext_T/"


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
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/AED/test/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/AED/viz/"
# score_threshold = 0.3

# # aerial seabird westafrica
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Aerial_seabird_westafrica_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Aerial_Seabirds_West_Africa/nocaption/"
# score_threshold = 0.3



# # Eikelboom
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Eikelboom/viz/"
# score_threshold = 0.3



# # polar bear
# gt_json_path =    '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/test_15.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/polar_bear_finetune/prediction_mm_grounding_dino_finetune_val.bbox.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/polar_bear/prediction_mm_grounding_dino_nocaption.bbox.json'

# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/polar_bear/finetune_nocaption/"
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/polar_bear/nocaption/"
# score_threshold = 0.1


# # sea lion
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_sea_lion_blackout/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_sealion_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_sea_lion_blackout/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/NOAA_sea_lion_blackout/viz/"
# score_threshold = 0.3


# # penguin
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/test_viz_grounded.json'
# pred_json_path = '//home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/penguins_od_finetune/prediction_mm_grounding_dino_finetune_test.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_penguins/finetune_nocaption/"
# score_threshold = 0.1

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


# DFOW17
# gt_json_path = '/home/m32patel/projects/def-dclausi/whale/merged/test/test_2017.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/DFO_whale_17/prediction_mm_grounding_dino_finetune_test.bbox.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2017_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'
# img_folder = '/home/m32patel/projects/def-dclausi/whale/merged/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Beluga_2017_dataset/finetune_nocaption/"
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Beluga_2017_dataset/nocaption/"
# score_threshold = 0.1


# # # DFOW23
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/coco_iter12345_val.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/DFO_whale_23/prediction_mm_grounding_dino_finetune_val.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/wo rk_dir_grounding_dino/DFO_Whale23/prediction_mm_grounding_dino_nocaption.bbox.json'
# img_folder = ''
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/DFO_Whale23/nocaption_finetune/"

# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/DFO_Whale23/nocaption/"
# score_threshold = 0.1


# # birds_monash
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_monash/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/monash_od_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_monash/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_monash/viz/"
# score_threshold = 0.25

# # aerial livestock
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/test_viz_grounded.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/aerial_livestock/viz/"
# score_threshold = 0.3



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



# # savmap
# gt_json_path =  '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/coco_split_val.json'
# pred_json_path = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/SAVMAP/prediction_mm_grounding_dino_nocaption.bbox.json"
# # pred_json_path = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/SAVMAP/prediction_mm_grounding_dino_finetune_val.bbox.json"


# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/SAVMAP_test/nocaption/"
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/SAVMAP_test/finetune_nocaption/"

# score_threshold = 0.1


# # # virunga garamba
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB_grounded.json'
# # pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba/prediction_mm_grounding_dino_finetune_test.bbox.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Virunga_garamba_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'

# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/test'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Virunga_Garamba/nocaption/"
# # save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Virunga_Garamba/nocaption_finetune/"
# score_threshold = 0.1


# /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/120.jpg
# For single image
# plot_coco_image(gt_json_path, pred_json_path, img_folder, save_folder, mode='file', image_name='120.jpg', score_threshold=score_threshold, iou_threshold=0.1, show_text=True)


# zero shot
gt_json_paths = [
    '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/coco_iter12345_val.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/test.json',
    # '/home/m32patel/projects/def-dclausi/whale/merged/test/test_ES_2016.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/test_15.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/test.json'
    ]
pred_json_paths =[
        '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/DFO_Whale23/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/michigan_od_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Narwhal_2016_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_artic_seal_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/polar_bear/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/tern_bioarxiv/prediction_mm_grounding_dino_nocaption_new_split.bbox.json',
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/WAID_livestock_dataset/prediction_mm_grounding_dino_nocaption_new_split.bbox.json'
                ]

img_folders = [
    # '',
    '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/',
    # '/home/m32patel/projects/def-dclausi/whale/merged/test/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/'
    ]
save_folders = [
    # DFOW 23
    "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/Eikelboom",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/Aerial-livestock-dataset",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/birds_michigan",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/whale",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/NOAA_arctic_seals",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/birds_penguins",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/polar_bear_annotated",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/birds_terns_(Already_processed_Hayes)",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/WAID"
]
score_threshold = 0.1
suffix='zero_shot'

# # fine tune
# gt_json_paths = [
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/coco_iter12345_val.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/test.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/test.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/test.json',
#     '/home/m32patel/projects/def-dclausi/whale/merged/test/test_ES_2016.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals/test.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/test.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/test_15.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/test.json',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/test.json'
#     ]

# pred_json_paths =[
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/DFO_whale_23/prediction_mm_grounding_dino_finetune_val.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/Eikelboom_dataset/prediction_mm_grounding_dino_finetune_test.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/Han_aerial_livestock_dataset/prediction_mm_grounding_dino_finetune_val.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/michigan_od_dataset/prediction_mm_grounding_dino_finetune_test.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/Narwhal_2016_dataset/prediction_mm_grounding_dino_finetune_val.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/NOAA_arctic_seal_dataset/prediction_mm_grounding_dino_finetune_val.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/penguins_od_finetune/prediction_mm_grounding_dino_finetune_test.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/polar_bear_finetune/prediction_mm_grounding_dino_finetune_val.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/tern_bioarxiv/prediction_mm_grounding_dino_finetune_test.bbox.json',
#         '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/WAID_livestock_dataset/prediction_mm_grounding_dino_finetune_test.bbox.json'
#                 ]

# img_folders = [
#     '',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/',
#     '/home/m32patel/projects/def-dclausi/whale/merged/test/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/',
#     '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/'
#     ]
# save_folders = [
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/DFO_whale_23",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/Eikelboom",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/Aerial-livestock-dataset",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/birds_michigan",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/whale",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/NOAA_arctic_seals",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/birds_penguins",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/polar_bear_annotated",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/birds_terns_(Already_processed_Hayes)",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision/WAID"
# ]
score_threshold = 0.1
suffix='finetune'

for gt_json_path,pred_json_path,img_folder,save_folder in zip(gt_json_paths,pred_json_paths,img_folders,save_folders):
    # # # For all images
    # analyze_coco_and_predictions(gt_json_path, pred_json_path, score_threshold,suffix)
    plot_coco_image(gt_json_path, pred_json_path, img_folder, save_folder, mode='all', score_threshold=score_threshold, iou_threshold=0.5, show_text=True, plot_images=False,suffix=suffix)
