import os
from PIL import Image
from collections import defaultdict

# Path to folder containing slices
folder_path = "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider/mm_grounding_dino_real_filtered_epoch10/real_all"  # Update this with the path to your sliced images
# Create output folder
output_folder = os.path.join(os.path.dirname(folder_path), "merged")
os.makedirs(output_folder, exist_ok=True)

# Group slices by image prefix
file_groups = defaultdict(list)

for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg"):
        # Extract the prefix (everything before the bounding box coordinates)
        prefix = "_".join(file_name.split("_")[:-4])
        file_groups[prefix].append(file_name)

for prefix, files in file_groups.items():
    # Initialize variables to determine the canvas size
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0

    # Store slices and their coordinates
    slices = []
    for file in files:
        # Remove the file extension and parse coordinates
        parts = file.replace(".jpg", "").split("_")[-4:]
        x_min, y_min, x_max, y_max = map(int, parts)
        slices.append((x_min, y_min, x_max, y_max, file))
        # Update canvas bounds
        min_x = min(min_x, x_min)
        min_y = min(min_y, y_min)
        max_x = max(max_x, x_max)
        max_y = max(max_y, y_max)

    # Calculate canvas size
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    # Create a blank canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height))

    # Paste each slice onto the canvas
    for x_min, y_min, x_max, y_max, file in slices:
        slice_img = Image.open(os.path.join(folder_path, file))
        canvas.paste(slice_img, (x_min - min_x, y_min - min_y))

    # Save the merged image
    output_file = os.path.join(output_folder, f"{prefix}_merged.jpg")
    canvas.save(output_file)
    print(f"Merged image saved to {output_file}")

print("All images merged!")
