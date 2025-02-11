import json
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from datetime import datetime
import os

def calculate_classwise_mae(gt_file, pred_file, score_threshold=0.0):
    """
    Calculate class-wise Mean Absolute Error (MAE) for object detection counts
    
    Args:
        gt_file (str): Path to COCO ground truth JSON file
        pred_file (str): Path to COCO prediction JSON file
        score_threshold (float): Confidence score threshold for predictions (default: 0.0)
    
    Returns:
        dict: Class-wise MAE values
        float: Overall MAE across all classes
        dict: Per-image errors for each class
    """
    # Load the files
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # Create mappings for category IDs to names
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    # Create mapping for image IDs to file names (if available)
    image_mapping = {img['id']: img.get('file_name', str(img['id'])) 
                    for img in gt_data['images']}
    
    # Initialize counters for ground truth
    gt_counts = defaultdict(lambda: defaultdict(int))
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        gt_counts[image_id][category_id] += 1
    
    # Initialize counters for predictions
    pred_counts = defaultdict(lambda: defaultdict(int))
    for pred in pred_data:
        if pred.get('score', 1.0) >= score_threshold:
            image_id = pred['image_id']
            category_id = pred['category_id']
            pred_counts[image_id][category_id] += 1
    
    # Calculate MAE for each class
    class_errors = defaultdict(list)
    per_image_errors = defaultdict(lambda: defaultdict(dict))
    
    # Get all unique image IDs
    all_image_ids = set(gt_counts.keys()) | set(pred_counts.keys())
    
    # Calculate absolute errors for each class in each image
    for image_id in all_image_ids:
        for category_id in categories.keys():
            gt_count = gt_counts[image_id][category_id]
            pred_count = pred_counts[image_id][category_id]
            abs_error = abs(pred_count - gt_count)
            class_errors[category_id].append(abs_error)
            
            # Store detailed information for each image and class
            image_name = image_mapping.get(image_id, str(image_id))
            per_image_errors[image_name][categories[category_id]] = {
                'gt_count': gt_count,
                'pred_count': pred_count,
                'abs_error': abs_error
            }
    
    # Calculate MAE for each class
    class_mae = {}
    for category_id, errors in class_errors.items():
        class_mae[categories[category_id]] = np.mean(errors)
    
    # Calculate overall MAE
    overall_mae = np.mean([mae for mae in class_mae.values()])
    
    return class_mae, overall_mae, per_image_errors

def save_results(class_mae, overall_mae, per_image_errors, score_threshold, output_dir="results"):
    """
    Save results to a text file
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"mae_results_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"MAE Analysis Results - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write threshold information
        f.write(f"Confidence Score Threshold: {score_threshold}\n")
        f.write("-" * 50 + "\n\n")
        
        # Write overall MAE
        f.write("1. Overall Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall MAE: {overall_mae:.3f}\n\n")
        
        # Write class-wise MAE
        f.write("2. Class-wise MAE:\n")
        f.write("-" * 50 + "\n")
        for class_name, mae in class_mae.items():
            f.write(f"{class_name}: {mae:.3f}\n")
        f.write("\n")
        
        # Write detailed per-image results
        f.write("3. Detailed Per-Image Results:\n")
        f.write("-" * 50 + "\n\n")
        
        for image_name, class_errors in per_image_errors.items():
            f.write(f"Image: {image_name}\n")
            table_data = []
            headers = ["Class", "Ground Truth", "Predicted", "Absolute Error"]
            
            for class_name, error_info in class_errors.items():
                if error_info['gt_count'] > 0 or error_info['pred_count'] > 0:
                    table_data.append([
                        class_name,
                        error_info['gt_count'],
                        error_info['pred_count'],
                        error_info['abs_error']
                    ])
            
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
    
    return output_file

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    gt_file = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB_grounded.json"
    pred_file = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba/prediction_mm_grounding_dino_finetune_test.bbox.json"
    save_folder= '/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/Virunga_Garamba/nocaption_finetune/'
    score_threshold = 0.3  # Adjust this threshold as needed
    
    # Calculate MAE
    class_mae, overall_mae, per_image_errors = calculate_classwise_mae(
        gt_file, pred_file, score_threshold
    )
    
    # Save results to file
    output_file = save_results(class_mae, overall_mae, per_image_errors, score_threshold, output_dir=save_folder)
    
    # Print results to console as well
    print(f"Results have been saved to: {output_file}")
    
    # # Optionally, display results in console
    # with open(output_file, 'r') as f:
    #     print(f.read())