import json
import os
import cv2
import numpy as np
from collections import defaultdict
from icecream import ic
from tqdm import tqdm

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

def draw_bbox(img, bbox, color, label, show_text, thickness=1):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    if show_text:
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

def plot_coco_image(gt_path, pred_path, img_folder, save_folder, mode='file', image_name=None, score_threshold=0.5, iou_threshold=0.5, show_text=True):
    # Load JSON files
    gt_data = load_json(gt_path)
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

        # Sort predictions by score
        predictions = [pred for pred in pred_data if pred['image_id'] == image_id and pred['score'] >= score_threshold]
        predictions.sort(key=lambda x: x['score'], reverse=True)

        # Evaluate predictions
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
                    # label = f"TP: {pred_category_name} ({pred['score']:.2f})"
                    label = f"TP: {pred_category_name}"
                    
                    gt_anns[category_id].remove(max_gt)
                else:
                    # False Positive
                    color = (0, 0, 255)  # Red
                    # label = f"FP: {pred_category_name} ({pred['score']:.2f})"
                    label = f"FP: {pred_category_name}"
                    
            else:
                # False Positive (no GT for this category)
                color = (0, 0, 255)  # Red
                # label = f"FP: {pred_category_name} ({pred['score']:.2f})"
                label = f"FP: {pred_category_name}"
                
                
            draw_bbox(img, pred_bbox, color, label, show_text)

        # Draw remaining GT boxes (False Negatives)
        for category_id, anns in gt_anns.items():
            gt_category_name = category_id_to_name.get(category_id, "Unknown")
            for ann in anns:
                label = f"FN: {gt_category_name}"
                draw_bbox(img, ann['bbox'], (255, 0, 0), label, show_text)  # Blue for False Negatives

        # Save the image
        os.makedirs(save_folder, exist_ok=True)
        img_name = os.path.basename(img_name)
        save_path = os.path.join(save_folder, f"{img_name}")
        cv2.imwrite(save_path, img)

# # Example usage
# gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/eider_duck_synthesized/annotations.json'

# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/eider_syn_finetune/prediction_mm_grounding_dino_finetune_test.bbox.json'
# img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/eider_duck_synthesized/'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck_synthesized/viz_caption"


# Example usage
gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/coco_test.json'

pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eider_duck/prediction_mm_grounding_dino_nocaption_finetune.bbox.json'
img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/'
save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck_real_from_synth/viz_caption"
score_threshold = 0.2
# # For single image
plot_coco_image(gt_json_path, pred_json_path, img_folder, save_folder, mode='file', image_name='2008/grants_photos_1/100EOS5D/IMG_4430.JPG', score_threshold=score_threshold, iou_threshold=0.5, show_text=False)

# # # For all images
# plot_coco_image(gt_json_path, pred_json_path, img_folder, save_folder, mode='all', score_threshold=score_threshold, iou_threshold=0.3, show_text=False)
