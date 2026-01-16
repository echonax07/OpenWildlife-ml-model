import os
import requests
import json
from urllib.parse import urlparse, unquote
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from collections import Counter
from label_studio_sdk import Client
from PIL import Image
import io

# Configuration
LS_URL = 'http://129.97.250.147:8080'
API_TOKEN = 'ebdc6fa5f2c3abcd502b55d5ccc1dc0e4ae9f68d'
PROJECT_ID = 110

# Output directory
OUTPUT_DIR = 'visualizations/Scott-2008-all_photos_combined_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color mapping based on your configuration (BGR format for OpenCV)
KEYPOINT_COLORS = {
    'female duck': (68, 68, 239),    # #EF4444 -> BGR
    'male duck': (255, 0, 0),       # blue -> BGR
    'Ice': (0, 255, 0),             # green -> BGR
    'Juvenile duck': (0, 255, 255), # yellow -> BGR
    'duck': (0, 165, 255)           # orange -> BGR
}

POLYGON_COLORS = {
    'Train region': (255, 0, 200)   # #c800ff -> BGR
}

RECTANGLE_COLORS = {
    'rectangle': (0, 0, 0)          # black -> BGR
}

CIRCLE_RADIUS = 1

def download_image(image_url, headers=None):
    """Download image from URL and return OpenCV image"""
    try:
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        
        # Convert to OpenCV format
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None

def extract_predictions(task):
    """Extract predictions from a Label Studio task"""
    predictions = {
        'keypoints': [],
        'polygons': [],
        'rectangles': [],
        'width': None,
        'height': None,
        'train_region_choice': None,
        'scores': []
    }
    
    if not task.get('predictions'):
        return predictions
    
    # Get all predictions (or just the latest one)
    all_predictions = task['predictions']
    
    for prediction in all_predictions:
        results = prediction.get('result', [])
        
        for result in results:
            score = result.get('score', None)
            
            if result.get('type') == 'keypointlabels':
                keypoints = result.get('value', {})
                label = keypoints.get('keypointlabels', [])
                if label:
                    prediction_data = {
                        'label': label[0],
                        'x': keypoints.get('x', 0),
                        'y': keypoints.get('y', 0),
                        'width': keypoints.get('width', 0),
                        'height': keypoints.get('height', 0),
                        'score': score
                    }
                    predictions['keypoints'].append(prediction_data)
                    if score is not None:
                        predictions['scores'].append(score)
            
            elif result.get('type') == 'polygonlabels':
                polygon = result.get('value', {})
                label = polygon.get('polygonlabels', [])
                if label:
                    prediction_data = {
                        'label': label[0],
                        'points': polygon.get('points', []),
                        'score': score
                    }
                    predictions['polygons'].append(prediction_data)
                    if score is not None:
                        predictions['scores'].append(score)
            
            elif result.get('type') == 'rectanglelabels':
                rectangle = result.get('value', {})
                label = rectangle.get('rectanglelabels', [])
                if label:
                    prediction_data = {
                        'label': label[0],
                        'x': rectangle.get('x', 0),
                        'y': rectangle.get('y', 0),
                        'width': rectangle.get('width', 0),
                        'height': rectangle.get('height', 0),
                        'score': score
                    }
                    predictions['rectangles'].append(prediction_data)
                    if score is not None:
                        predictions['scores'].append(score)
            
            elif result.get('from_name') == 'width':
                predictions['width'] = result.get('value', {}).get('text', [None])[0]
            
            elif result.get('from_name') == 'height':
                predictions['height'] = result.get('value', {}).get('text', [None])[0]
            
            elif result.get('from_name') == 'train_region_choice':
                predictions['train_region_choice'] = result.get('value', {}).get('choices', [None])[0]
    
    return predictions

def draw_statistics_overlay(image, predictions):
    """Draw statistics overlay on the image"""
    img_height, img_width = image.shape[:2]
    
    # Count all labels
    label_counts = Counter()
    
    # Count keypoints
    for kp in predictions['keypoints']:
        label_counts[kp['label']] += 1
    
    # Count polygons
    for poly in predictions['polygons']:
        label_counts[poly['label']] += 1
    
    # Count rectangles
    for rect in predictions['rectangles']:
        label_counts[rect['label']] += 1
    
    if not label_counts:
        return image
    
    # Calculate total count and average score
    total_count = sum(label_counts.values())
    avg_score = np.mean(predictions['scores']) if predictions['scores'] else None
    
    # Create overlay box (larger to accommodate bigger text and additional info)
    overlay_width = 500
    overlay_height = 280
    overlay_x = 20
    overlay_y = 60
    
    # Create semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (overlay_x, overlay_y), 
                  (overlay_x + overlay_width, overlay_y + overlay_height), 
                  (255, 255, 255), -1)
    
    # Draw border
    cv2.rectangle(overlay, (overlay_x, overlay_y), 
                  (overlay_x + overlay_width, overlay_y + overlay_height), 
                  (0, 0, 0), 2)
    
    # Blend with original image for transparency
    alpha = 0.9
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw title
    title_text = "Prediction Statistics:"
    cv2.putText(image, title_text, (overlay_x + 10, overlay_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw statistics with colored bullets
    y_offset = overlay_y + 60
    line_height = 30
    
    for label, count in label_counts.items():
        # Get color for this label
        color = KEYPOINT_COLORS.get(label, POLYGON_COLORS.get(label, RECTANGLE_COLORS.get(label, (128, 128, 128))))
        
        # Draw bullet point (filled circle)
        bullet_x = overlay_x + 20
        bullet_y = y_offset - 8
        cv2.circle(image, (bullet_x, bullet_y), 8, color, -1)
        cv2.circle(image, (bullet_x, bullet_y), 8, (0, 0, 0), 1)
        
        # Draw text
        text = f"{label}: {count}"
        cv2.putText(image, text, (bullet_x + 25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        
        y_offset += line_height
    
    # Add total count
    y_offset += 10
    total_text = f"Total: {total_count}"
    cv2.putText(image, total_text, (overlay_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Add average score
    if avg_score is not None:
        y_offset += 25
        score_text = f"Avg Score: {avg_score:.3f}"
        cv2.putText(image, score_text, (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Add dimensions and training info
    y_offset += 30
    
    # Draw dimensions if available
    if predictions.get('width') or predictions.get('height'):
        dim_text = f"Dimensions: {predictions.get('width', 'N/A')} x {predictions.get('height', 'N/A')}"
        cv2.putText(image, dim_text, (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += line_height
    
    # Draw training choice if available
    if predictions.get('train_region_choice'):
        choice_text = f"Training: {predictions['train_region_choice']}"
        cv2.putText(image, choice_text, (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    return image

def visualize_predictions(image, predictions, task_id, image_filename):
    """Visualize predictions on image using OpenCV"""
    # Make a copy of the image
    vis_image = image.copy()
    img_height, img_width = vis_image.shape[:2]
    
    # Draw keypoints
    for kp in predictions['keypoints']:
        label = kp['label']
        score = kp.get('score')
        
        # Convert percentage coordinates to pixel coordinates
        x = int((kp['x'] / 100) * img_width)
        y = int((kp['y'] / 100) * img_height)
        
        color = KEYPOINT_COLORS.get(label, (0, 0, 255))  # Default to red
        
        # Draw keypoint as filled circle
        cv2.circle(vis_image, (x, y), CIRCLE_RADIUS, color, -1)
        cv2.circle(vis_image, (x, y), CIRCLE_RADIUS + 2, (0, 0, 0), 1)  # Black border
        
        # Add label text with score
        if score is not None:
            label_text = f"{label} ({score:.2f})"
        else:
            label_text = label
            
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Text position
        text_x = x + 12
        text_y = y - 12
        
        # Ensure text is within image bounds
        if text_x + text_width > img_width:
            text_x = x - text_width - 12
        if text_y < text_height:
            text_y = y + text_height + 12
        
        # # Text background
        # cv2.rectangle(vis_image, 
        #               (text_x - 2, text_y - text_height - 2),
        #               (text_x + text_width + 2, text_y + 2),
        #               color, -1)
        
        # # Text label
        # cv2.putText(vis_image, label_text,
        #            (text_x, text_y),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw polygons (train regions)
    for poly in predictions['polygons']:
        label = poly['label']
        score = poly.get('score')
        points = poly['points']
        
        if points:
            # Convert percentage coordinates to pixel coordinates
            pixel_points = []
            for point in points:
                x = int((point[0] / 100) * img_width)
                y = int((point[1] / 100) * img_height)
                pixel_points.append([x, y])
            
            color = POLYGON_COLORS.get(label, (255, 0, 255))  # Default to magenta
            
            # Draw polygon
            pts = np.array(pixel_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], True, color, 3)
            
            # Add label text at the center
            if pixel_points:
                center_x = int(np.mean([p[0] for p in pixel_points]))
                center_y = int(np.mean([p[1] for p in pixel_points]))
                
                if score is not None:
                    label_text = f"{label} ({score:.2f})"
                else:
                    label_text = label
                    
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Text background
                cv2.rectangle(vis_image,
                              (center_x - text_width//2 - 5, center_y - text_height//2 - 5),
                              (center_x + text_width//2 + 5, center_y + text_height//2 + 5),
                              color, -1)
                
                # Text
                cv2.putText(vis_image, label_text,
                           (center_x - text_width//2, center_y + text_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw rectangles
    for rect in predictions['rectangles']:
        label = rect['label']
        score = rect.get('score')
        
        # Convert percentage coordinates to pixel coordinates
        x = int((rect['x'] / 100) * img_width)
        y = int((rect['y'] / 100) * img_height)
        width = int((rect['width'] / 100) * img_width)
        height = int((rect['height'] / 100) * img_height)
        
        color = RECTANGLE_COLORS.get(label, (0, 0, 0))  # Default to black
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + width, y + height), color, 1)
        
        # Add label text with score
        if score is not None:
            label_text = f"{label} ({score:.2f})"
        else:
            label_text = label
            
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Text background
        cv2.rectangle(vis_image,
                      (x, y - text_height - 10),
                      (x + text_width + 10, y),
                      color, -1)
        
        # Text
        cv2.putText(vis_image, label_text,
                   (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add statistics overlay
    vis_image = draw_statistics_overlay(vis_image, predictions)
    
    # Add title to the image with a more distinguishable font style
    title_text = f"Task {task_id} - {image_filename} [PREDICTIONS]"
    (text_width, text_height), baseline = cv2.getTextSize(
        title_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    # Coordinates for the rectangle
    rect_x1, rect_y1 = 5, 10
    rect_x2, rect_y2 = 5 + text_width + 20, 10 + text_height + 20

    # Draw semi-transparent rectangle
    overlay = vis_image.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    alpha = 0.5
    vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)

    # Draw a thick black outline for better contrast
    cv2.putText(vis_image, title_text, (15, rect_y1 + text_height + 5),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    # Draw the white text on top
    cv2.putText(vis_image, title_text, (15, rect_y1 + text_height + 5),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    return vis_image

def main():
    """Main function to process all tasks and create prediction visualizations"""
    print("Connecting to Label Studio...")
    
    # Initialize Label Studio client
    ls = Client(url=LS_URL, api_key=API_TOKEN)
    
    try:
        # Get project
        project = ls.get_project(PROJECT_ID)
        print(f"Successfully connected to project: {project.title}")
        
        # Get all tasks
        tasks = project.get_tasks()
        print(f"Found {len(tasks)} tasks")
        
        # Process each task
        processed_count = 0
        skipped_count = 0
        
        for i, task in enumerate(tasks):
            print(f"\nProcessing task {i+1}/{len(tasks)} (ID: {task['id']})")
            
            # Check if task has predictions
            if not task.get('predictions'):
                print(f"No predictions found for task {task['id']} - skipping")
                skipped_count += 1
                continue
            
            # Get image URL
            image_url = task.get('data', {}).get('image')
            if not image_url:
                print(f"No image URL found for task {task['id']}")
                skipped_count += 1
                continue
            
            # Handle relative URLs
            if image_url.startswith('/'):
                image_url2 = LS_URL.rstrip('/') + image_url
            else:
                image_url2 = image_url
            
            # Download image
            headers = {'Authorization': f'Token {API_TOKEN}'}
            image = download_image(image_url2, headers)
            
            if image is None:
                print(f"Failed to download image for task {task['id']}")
                skipped_count += 1
                continue
            
            # Extract predictions
            predictions = extract_predictions(task)
            
            # Determine image filename
            is_uploaded_file = image_url.startswith("/data/upload")
            is_local_storage_file = image_url.startswith("/data/local-files") and "?d=" in image_url
            is_cloud_storage_file = (
                image_url.startswith("s3:") or image_url.startswith("gs:") or image_url.startswith("azure-blob:")
            )
            
            if is_uploaded_file:
                # Create filename
                image_filename = os.path.basename(urlparse(image_url).path) or f"task_{task['id']}.jpg"
            elif is_local_storage_file:
                # Extract filename from local storage URL
                image_filename = os.path.basename(unquote(image_url.split('?d=')[1]))
            elif is_cloud_storage_file:
                image_filename = f"task_{task['id']}.jpg"
            else:
                image_filename = f"task_{task['id']}.jpg"
            
            # Create visualization
            vis_image = visualize_predictions(image, predictions, task['id'], image_filename)
            
            # Save the visualization
            output_filename = f"{image_filename.split('.')[0].split('-')[-1]}.png"
            
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, vis_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            print(f"Saved prediction visualization: {output_path}")
            print(f"Found {len(predictions['keypoints']) + len(predictions['polygons']) + len(predictions['rectangles'])} predictions")
            processed_count += 1
        
        print(f"\nSummary:")
        print(f"- Processed: {processed_count} tasks with predictions")
        print(f"- Skipped: {skipped_count} tasks (no predictions or errors)")
        print(f"- All prediction visualizations saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Label Studio server is running and accessible")

if __name__ == "__main__":
    main()