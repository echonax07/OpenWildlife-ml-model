import os
import json
import re

def merge_patch_predictions(full_images_path, image_patches_path, prediction_patches_path, output_path):
    # Load full image metadata
    with open(full_images_path, 'r') as f:
        full_images = json.load(f)["images"]
    
    # Load patch-to-full image mapping
    with open(image_patches_path, 'r') as f:
        image_patches = json.load(f)["images"]
    
    # Load patch predictions
    with open(prediction_patches_path, 'r') as f:
        patch_predictions = json.load(f)
    
    # Create a mapping from patch image_id to full image
    patch_to_full = {}
    full_image_dims = {}
    
    for full_img in full_images:
        full_image_dims[full_img["id"]] = (full_img["width"], full_img["height"])
    
    for patch in image_patches:
        match = re.search(r"_x(\d+)_y(\d+)", patch["file_name"])
        if match:
            x_offset, y_offset = map(int, match.groups())
            full_image_id = patch["file_name"].split("/")[0]  # Assuming patches are stored in subfolders per full image
            patch_to_full[patch["id"]] = (full_image_id, x_offset, y_offset)
    
    # Transform patch predictions back to full image coordinates
    full_predictions = []
    
    for pred in patch_predictions:
        patch_id = pred["image_id"]
        if patch_id not in patch_to_full:
            continue  # Skip if no mapping found
        
        full_image_id, x_offset, y_offset = patch_to_full[patch_id]
        bbox_x, bbox_y, bbox_w, bbox_h = pred["bbox"]
        
        # Transform bbox coordinates to full image space
        new_x = bbox_x + x_offset
        new_y = bbox_y + y_offset
        
        # Ensure bbox is within full image bounds
        full_width, full_height = full_image_dims[full_image_id]
        if new_x + bbox_w > full_width or new_y + bbox_h > full_height:
            continue  # Skip out-of-bound predictions
        
        full_predictions.append({
            "image_id": full_image_id,
            "bbox": [new_x, new_y, bbox_w, bbox_h],
            "score": pred["score"],
            "category_id": pred["category_id"]
        })
    
    # Save merged predictions
    with open(output_path, 'w') as f:
        json.dump(full_predictions, f)
    
    print(f"Merged predictions saved to {output_path}")

# Example usage
merge_patch_predictions(
    "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json",
    "/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping/patch_annotations.json",
    "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba_try3/prediction_mm_grounding_dino_finetune_test.bbox.json",
    "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba_try3/prediction_full_patches.json"
)
