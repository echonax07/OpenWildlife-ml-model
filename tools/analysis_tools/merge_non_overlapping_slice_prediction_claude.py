import json
import os
from typing import Dict, List

def extract_patch_coordinates(filename: str) -> tuple:
    """Extract x, y coordinates from patch filename (e.g., 'image_x512_y256.png')"""
    parts = filename.split('_')
    x = int(parts[-2].replace('x', ''))
    y = int(parts[-1].replace('y', '').split('.')[0])
    return x, y

def merge_patch_predictions(full_images_path: str, 
                          image_patches_path: str, 
                          prediction_patches_path: str,
                          output_path: str):
    """
    Merge predictions from image patches back to full images.
    
    Args:
        full_images_path: Path to original COCO format JSON with full images
        image_patches_path: Path to JSON containing patch information
        prediction_patches_path: Path to JSON containing predictions on patches
        output_path: Path to save the merged predictions JSON
    """
    # Load all JSON files
    with open(full_images_path, 'r') as f:
        full_images_data = json.load(f)
    with open(image_patches_path, 'r') as f:
        patches_data = json.load(f)
    with open(prediction_patches_path, 'r') as f:
        predictions_data = json.load(f)  # This is now expected to be a list of predictions

    # Create mapping from patch filename to original image id
    patch_to_original: Dict[int, dict] = {}
    for patch_img in patches_data['images']:
        # Extract original image name and patch coordinates
        filename_parts = os.path.basename(patch_img['file_name']).split('_x')
        original_name = filename_parts[0]
        x, y = extract_patch_coordinates(patch_img['file_name'])
        
        # Find corresponding full image
        for full_img in full_images_data['images']:
            if os.path.splitext(os.path.basename(full_img['file_name']))[0] == original_name:
                patch_to_original[patch_img['id']] = {
                    'full_image_id': full_img['id'],
                    'offset_x': x,
                    'offset_y': y
                }
                break

    # Transform patch predictions to full image coordinates
    merged_predictions = []
    next_annotation_id = 1

    # Now iterate directly over predictions_data which is a list
    for pred in predictions_data:
        patch_id = pred['image_id']
        if patch_id in patch_to_original:
            original_info = patch_to_original[patch_id]
            
            # Transform bbox coordinates
            bbox = pred['bbox']
            transformed_bbox = [
                bbox[0] + original_info['offset_x'],  # x
                bbox[1] + original_info['offset_y'],  # y
                bbox[2],  # width (unchanged)
                bbox[3]   # height (unchanged)
            ]
            
            # Create new prediction with transformed coordinates
            merged_pred = {
                'id': next_annotation_id,
                'image_id': original_info['full_image_id'],
                'category_id': pred['category_id'],
                'bbox': transformed_bbox,
                'score': pred['score'] if 'score' in pred else 1.0,
                'area': pred['area'] if 'area' in pred else (bbox[2] * bbox[3])
            }
            
            merged_predictions.append(merged_pred)
            next_annotation_id += 1

    # Save merged predictions directly as a list
    with open(output_path, 'w') as f:
        json.dump(merged_predictions, f)

    print(f"Merged predictions saved to {output_path}")
    print(f"Total predictions after merging: {len(merged_predictions)}")

# Example usage
full_images_path = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json"
image_patches_path = "/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping/patch_annotations.json"
prediction_patches_path = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba_try3/prediction_mm_grounding_dino_finetune_test.bbox.json"
output_path = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba_try3/prediction_full_patches.json"

merge_patch_predictions(
    full_images_path,
    image_patches_path,
    prediction_patches_path,
    output_path
)