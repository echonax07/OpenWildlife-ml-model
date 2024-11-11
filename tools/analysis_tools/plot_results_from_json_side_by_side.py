import json
import os
import cv2
import numpy as np
from collections import defaultdict
from icecream import ic
from tqdm import tqdm
import textwrap

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

def plot_coco_image_side_by_side(gt_path, pred_path1, pred_path2, img_folder, save_folder, mode='file', image_name=None, score_threshold=0.5, iou_threshold=0.5, show_text=True):
    # Load JSON files
    gt_data = load_json(gt_path)
    pred_data1 = load_json(pred_path1)
    pred_data2 = load_json(pred_path2)

    # Create a category ID to name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in gt_data['categories']}

    # Determine images to process based on mode
    images_to_process = [image_name] if mode == 'file' and image_name else [img['file_name'] for img in gt_data['images']]

    for img_name in tqdm(images_to_process):
        # Find image ID
        image_id = None
        for img_info in gt_data['images']:
            if img_info['file_name'] == img_name:
                image_id = img_info['id']
                if 'caption' in img_info:
                    caption=img_info['caption']
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

        # Plot results for both models
        def annotate_image(img, pred_data, caption):
            annotated_img = img.copy()
            gt_anns = defaultdict(list)
            for ann in gt_data['annotations']:
                if ann['image_id'] == image_id:
                    gt_anns[ann['category_id']].append(ann)

            predictions = [pred for pred in pred_data if pred['image_id'] == image_id and pred['score'] >= score_threshold]
            predictions.sort(key=lambda x: x['score'], reverse=True)

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
                        color = (0, 255, 0)  # True Positive: Green
                        label = f"TP: {pred_category_name} ({pred['score']:.2f})"
                        gt_anns[category_id].remove(max_gt)
                    else:
                        color = (0, 0, 255)  # False Positive: Red
                        label = f"FP: {pred_category_name} ({pred['score']:.2f})"
                else:
                    color = (0, 0, 255)  # False Positive: Red
                    label = f"FP: {pred_category_name} ({pred['score']:.2f})"
                
                draw_bbox(annotated_img, pred_bbox, color, label, show_text)

            for category_id, anns in gt_anns.items():
                gt_category_name = category_id_to_name.get(category_id, "Unknown")
                for ann in anns:
                    label = f"FN: {gt_category_name}"
                    draw_bbox(annotated_img, ann['bbox'], (255, 0, 0), label, show_text)  # False Negative: Blue
                # Add caption at the bottom of the image

    # Add caption at the bottom of the image, wrapped to fit within the image width
            if caption:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                max_width = annotated_img.shape[1] - 20  # Add 10-pixel padding on each side

                # Wrap text based on estimated character width
                approx_char_width = cv2.getTextSize("a", font, font_scale, font_thickness)[0][0]
                max_chars_per_line = max_width // approx_char_width
                wrapped_text = textwrap.fill(caption, width=max_chars_per_line)

                # Start slightly above the bottom of the image and draw each line downward
                start_y = annotated_img.shape[0] - 40  # Start 40 pixels from the bottom
                for i, line in enumerate(wrapped_text.splitlines()):
                    text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                    text_x = (annotated_img.shape[1] - text_size[0]) // 2  # Center align
                    text_y = start_y + i * (text_size[1] + 5)  # Increment y position for each line
                    cv2.putText(annotated_img, line, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


            return annotated_img

        img1_annotated = annotate_image(img, pred_data1, caption)
        img2_annotated = annotate_image(img, pred_data2, '')

        # Combine images side by side
        combined_img = np.hstack((img1_annotated, img2_annotated))

        # Save the combined image
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{img_name}")
        cv2.imwrite(save_path, combined_img)

# Example usage
gt_json_path = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/test_grounded.json'
pred_json_path1 = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_caption.bbox.json'
pred_json_path2 = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'
img_folder = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/'
save_folder = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/birds_penguins/combined"
score_threshold = 0.3

# For single image
# plot_coco_image_side_by_side(gt_json_path, pred_json_path1, pred_json_path2, img_folder, save_folder, mode='file', image_name='14155f30121958a811385dd40c96f8e9294da086.JPG', score_threshold=score_threshold, iou_threshold=0.1, show_text=True)

# For all images
plot_coco_image_side_by_side(gt_json_path, pred_json_path1, pred_json_path2, img_folder, save_folder, mode='all', score_threshold=score_threshold, iou_threshold=0.1, show_text=True)
