import json
import os
import cv2
import numpy as np
from collections import defaultdict
from icecream import ic

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

def draw_bbox(img, bbox, color, label, thickness=2):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def plot_coco_image(gt_path, pred_path, img_folder, image_name, save_folder, score_threshold=0.5, iou_threshold=0.5):
    # Load JSON files
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    # Find image ID
    image_id = None
    for img_info in gt_data['images']:
        if img_info['file_name'] == image_name:
            image_id = img_info['id']
            break

    if image_id is None:
        print(f"Image {image_name} not found in ground truth data.")
        return

    # Load image
    img_path = os.path.join(img_folder, image_name)
    ic(img_path)
    img = cv2.imread(img_path)
    

    if img is None:
        print(f"Failed to load image: {img_path}")
        return

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
                label = f"TP: {pred['score']:.2f}"
                gt_anns[category_id].remove(max_gt)
            else:
                # False Positive
                color = (0, 0, 255)  # Red
                label = f"FP: {pred['score']:.2f}"
        else:
            # False Positive (no GT for this category)
            color = (0, 0, 255)  # Red
            # label = f"FP: {pred['score']:.2f}"
            label = f"{pred['score']:.2f}"
            
            

        draw_bbox(img, pred_bbox, color, label)

    # Draw remaining GT boxes (False Negatives)
    for category_id, anns in gt_anns.items():
        for ann in anns:
            draw_bbox(img, ann['bbox'], (255, 0, 0), "FN")  # Blue for False Negatives

    # Save the image
    os.makedirs(save_folder, exist_ok=True)
    image_name = os.path.basename(image_name)
    save_path = os.path.join(save_folder, f"plotted_{image_name}")
    cv2.imwrite(save_path, img)
    print(f"Saved plotted image to: {save_path}")

# gt_json_path = '/home/m32patel/projects/def-dclausi/whale/merged/test/test.json'
# pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dirs/DFO_whale_faster_rcnn_4096_convnext_T/prediction.bbox.json
gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/savmap_dataset_v2/test.json'
pred_json_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/SAVMAP/prediction.bbox.json'

img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/savmap_dataset_v2/'
# img_folder = ''

# image_name = "20150812_25mm_002545.jpg"
image_name = '7e9eb9142dac4e1bb592cbaedec7af72.JPG'
# save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dirs/prediction_images/DFO_whale_faster_rcnn_4096_convnext_T_AL"
save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/SAVMAP/prediction/"
score_threshold = 0.2  # You can adjust this value
plot_coco_image(gt_json_path, pred_json_path, img_folder, image_name, save_folder, score_threshold=score_threshold, iou_threshold=0.1)

