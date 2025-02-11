import os
import json
from PIL import Image
from tqdm import tqdm

def split_coco_into_patches(coco_annotation_path, img_root_dir, output_dir, patch_size=(1024, 1024)):
    """
    Splits images from a COCO dataset into non-overlapping patches and updates annotations accordingly.

    Args:
        coco_annotation_path (str): Path to the COCO annotation JSON file.
        img_root_dir (str): Directory containing the images referenced in the COCO file.
        output_dir (str): Directory to save the patches and updated annotations.
        patch_size (tuple): Size of each patch (height, width).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load COCO annotations
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    patch_annotations = []
    patch_images = []
    annotation_id = 1
    patch_id = 1

    for img_info in tqdm(images):
        img_path = os.path.join(img_root_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"Warning: Image {img_info['file_name']} not found in {img_root_dir}.")
            continue

        # Open the image
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            patch_width, patch_height = patch_size

            # Process the image into patches
            for y in range(0, img_height, patch_height):
                for x in range(0, img_width, patch_width):
                    box = (x, y, x + patch_width, y + patch_height)

                    # Crop the patch
                    patch = img.crop(box)

                    # Create the patch file path
                    subfolder_path = os.path.join(output_dir, os.path.dirname(img_info['file_name']))
                    if not os.path.exists(subfolder_path):
                        os.makedirs(subfolder_path)

                    patch_name = f"{os.path.splitext(os.path.basename(img_info['file_name']))[0]}_x{x}_y{y}.png"
                    patch_path = os.path.join(subfolder_path, patch_name)

                    # Save the patch
                    patch.save(patch_path)

                    # Add patch info to COCO images
                    patch_images.append({
                        "id": patch_id,
                        "file_name": os.path.relpath(patch_path, output_dir),
                        "width": patch_width,
                        "height": patch_height
                    })

                    # Adjust annotations for the patch
                    for ann in annotations:
                        if ann['image_id'] == img_info['id']:
                            bbox = ann['bbox']
                            bbox_x, bbox_y, bbox_w, bbox_h = bbox

                            # Check if the annotation falls within this patch
                            if (bbox_x + bbox_w > x and bbox_x < x + patch_width and
                                bbox_y + bbox_h > y and bbox_y < y + patch_height):
                                
                                # Adjust the bounding box to patch coordinates
                                new_bbox_x = max(0, bbox_x - x)
                                new_bbox_y = max(0, bbox_y - y)
                                new_bbox_w = min(bbox_w, x + patch_width - bbox_x)
                                new_bbox_h = min(bbox_h, y + patch_height - bbox_y)

                                if new_bbox_w > 0 and new_bbox_h > 0:  # Ensure valid bbox
                                    patch_annotations.append({
                                        "id": annotation_id,
                                        "image_id": patch_id,
                                        "category_id": ann['category_id'],
                                        "bbox": [new_bbox_x, new_bbox_y, new_bbox_w, new_bbox_h],
                                        "area": new_bbox_w * new_bbox_h,
                                        "iscrowd": ann['iscrowd']
                                    })
                                    annotation_id += 1

                    patch_id += 1

    # Save the new COCO annotation file
    output_coco_path = os.path.join(output_dir, "patch_annotations.json")
    new_coco_data = {
        "images": patch_images,
        "annotations": patch_annotations,
        "categories": categories
    }

    with open(output_coco_path, 'w') as f:
        json.dump(new_coco_data, f)

    print(f"Patches and annotations have been saved to {output_dir}")
    
# Example usage
coco_annotation_file = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json"
image_root_directory = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/test"
output_directory = "/home/m32patel/scratch/animal_patches/Virunga_Garamba/test_patches_non_overlapping"
split_coco_into_patches(coco_annotation_file, image_root_directory, output_directory)
