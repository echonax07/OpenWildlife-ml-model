import json
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from collections import Counter

def visualize_keypoints(image_path, json_path, output_path, 
                       keypoint_radius=8, score_threshold=0.5):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load predictions
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract predictions and apply threshold
    labels = data['labels']
    scores = data['scores']
    bboxes = data['bboxes']
    
    # Filter predictions based on score threshold
    filtered_predictions = [
        (label, score, bbox) 
        for label, score, bbox in zip(labels, scores, bboxes)
        if score >= score_threshold
    ]
    
    # Exit if no predictions remain after filtering
    if not filtered_predictions:
        print("No predictions above score threshold")
        cv2.imwrite(output_path, image)
        return
    
    # Separate filtered components
    filtered_labels = [p[0] for p in filtered_predictions]
    filtered_scores = [p[1] for p in filtered_predictions]
    filtered_bboxes = [p[2] for p in filtered_predictions]

    # Calculate class counts from filtered labels
    class_counts = Counter(filtered_labels)
    unique_labels = sorted(list(class_counts.keys()))
    
    # Create color mapping
    hsv_colors = [(x/len(unique_labels), 0.8, 0.8) 
                 for x in range(len(unique_labels))]
    # rgb_colors = [tuple((255 * np.array(hsv_to_rgb(h))).astype(int)) for h in hsv_colors] 
    rgb_colors = [tuple((255 * np.array(hsv_to_rgb(h))).astype(int)) for h in hsv_colors]
    label_colors = {label: tuple(map(int, (color[2], color[1], color[0])))  # BGR
                    for label, color in zip(unique_labels, rgb_colors)}
    
    # Draw keypoints and labels
    for label, score, bbox in filtered_predictions:
        x1, y1, x2, y2 = map(int, bbox)
        color = label_colors[label]
        
        # Calculate center point
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        
        # Draw keypoint
        cv2.circle(image, (x_center, y_center), keypoint_radius, color, -1)
        
        # Create text label
        label_text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Text position
        text_x = x_center + keypoint_radius + 5
        text_y = y_center - keypoint_radius
        
        # # Text background
        # cv2.rectangle(image, 
        #               (text_x - 2, text_y - text_height - 5),
        #               (text_x + text_width + 2, text_y + 5),
        #               color, -1)
        
        # # Text label
        # cv2.putText(image, label_text,
        #            (text_x, text_y),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Create color bar with counts
    bar_width = 300
    bar_height = image.shape[0]
    color_bar = np.ones((bar_height, bar_width, 3), dtype=np.uint8) * 255
    
    # Calculate block dimensions
    num_labels = len(unique_labels)
    block_height = max(40, bar_height // num_labels)
    
    actual_label_name = {0: "Female duck", 1: "Male duck", 2: "Ice", 3: "Juvenile duck", 4: "Duck"}
    # Draw class information
    for idx, label in enumerate(unique_labels):
        color = label_colors[label]
        count = class_counts[label]
        y_start = idx * block_height
        y_end = (idx + 1) * block_height
        
        # Color block
        cv2.rectangle(color_bar, (0, y_start), (bar_width, y_end), color, -1)
        
        # Class text
        text = f"Class {label}: {count}"
        text = f"{actual_label_name[label]}: {count}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        text_y = y_start + (block_height + th) // 2
        cv2.putText(color_bar, text, (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Add total count and threshold info
    total_text = f"Total: {len(filtered_labels)}"
    threshold_text = f"Score Threshold: {score_threshold}"
    (tw, th), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    
    # Draw threshold text
    cv2.putText(color_bar, threshold_text, (10, bar_height - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    # Draw total text
    cv2.putText(color_bar, total_text, (10, bar_height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    # Combine and save
    combined = np.hstack([image, color_bar])
    cv2.imwrite(output_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Output saved to {output_path}")

# Usage example with threshold
visualize_keypoints(
    '/home/pc2041/d8d096d5-overhead_2.png',
    '/home/pc2041/VIP_lab/labelstudio/mmwhale2/output2/preds/d8d096d5-overhead_2.json',
    '/home/pc2041/VIP_lab/labelstudio/mmwhale2/output2/preds/d8d096d5-overhead_2_plotted.png',
    score_threshold=0.3,  # Adjust this value as needed
    keypoint_radius = 3
)


