import os
from PIL import Image
import re
from collections import defaultdict
from tqdm import tqdm

def merge_patches_by_prefix(input_dir, output_dir):
    """
    Groups and merges patches based on their image prefixes.

    Args:
        input_dir (str): Path to the directory containing patches.
        output_dir (str): Path to save the merged images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group patches by their image prefix
    patches_by_prefix = defaultdict(list)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            match = re.match(r"^(.*)_x\d+_y\d+", filename)
            if match:
                prefix = match.group(1)
                patches_by_prefix[prefix].append(filename)

    for prefix, patch_files in tqdm(patches_by_prefix.items()):
        # Sort patches by coordinates
        def extract_coords(filename):
            match = re.search(r"_x(\d+)_y(\d+)", filename)
            if match:
                return int(match.group(1)), int(match.group(2))
            raise ValueError(f"Invalid filename format: {filename}")

        patch_files.sort(key=lambda f: extract_coords(f))

        # Determine the size of the merged image
        with Image.open(os.path.join(input_dir, patch_files[0])) as patch:
            patch_width, patch_height = patch.size

        # Calculate the dimensions of the merged image
        max_x = max(extract_coords(f)[0] for f in patch_files) // patch_width + 1
        max_y = max(extract_coords(f)[1] for f in patch_files) // patch_height + 1
        merged_width = max_x * patch_width
        merged_height = max_y * patch_height

        # Create a blank image for the merged result
        merged_image = Image.new("RGB", (merged_width, merged_height))

        # Paste patches into the blank image
        for patch_file in patch_files:
            patch_path = os.path.join(input_dir, patch_file)
            x, y = extract_coords(patch_file)
            with Image.open(patch_path) as patch:
                merged_image.paste(patch, (x, y))

        # Save the merged image
        output_path = os.path.join(output_dir, f"{prefix}_merged.png")
        merged_image.save(output_path)
        # print(f"Merged image saved at {output_path}")

# Example usage
input_directory = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/mm_grounding_dino_real_filtered_epoch10/real_nonoverlapping_patches/"
output_directory = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/mm_grounding_dino_real_filtered_epoch10/real_nonoverlapping_merged"
merge_patches_by_prefix(input_directory, output_directory)
