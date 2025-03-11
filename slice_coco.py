from typing import List, Union, Dict, Optional
from sahi.slicing import slice_coco

def main():
    # # Define the arguments as variables
    # coco_annotation_file_path = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json"  # Path to the COCO annotation file
    # image_dir = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/train"  # Directory containing the images
    # output_coco_annotation_file_name = "train_small_size_A_B_E_K_WH_WB.json"  # Name of the output COCO annotation file
    # output_dir = "/home/m32patel/scratch/animal_patches/Virunga_Garamba/train_wildlifemapper_reproduction"  # Directory to save the sliced images and annotations
    
    # Define the arguments as variables
    coco_annotation_file_path = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json"  # Path to the COCO annotation file
    image_dir = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/train"  # Directory containing the images
    output_coco_annotation_file_name = "train_only_positive_A_B_E_K_WH_WB.json"  # Name of the output COCO annotation file
    output_dir = "/home/m32patel/scratch/animal_patches/Virunga_Garamba/train_only_positive"  # Directory to save the sliced images and annotations
    
    ignore_negative_samples = True  # Ignore negative samples during slicing
    slice_height = 1024  # Height of the slices
    slice_width = 1024  # Width of the slices
    overlap_height_ratio = 0  # Overlap ratio for height
    overlap_width_ratio = 0 # Overlap ratio for width
    min_area_ratio = 0.1  # Minimum area ratio for keeping an annotation
    out_ext = ".jpg"  # Output image extension (e.g., '.jpg', '.png')
    verbose = False  # Enable verbose output
    save_only_positive_slices = True

    # Call the slice_coco function with the defined variables
    result = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=image_dir,
        output_coco_annotation_file_name=output_coco_annotation_file_name,
        output_dir=output_dir,
        ignore_negative_samples=ignore_negative_samples,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        min_area_ratio=min_area_ratio,
        out_ext=out_ext,
        verbose=verbose,
        save_only_positive_slices = save_only_positive_slices
    )

    if verbose:
        print("Slicing completed. Results:", result)

if __name__ == "__main__":
    main()