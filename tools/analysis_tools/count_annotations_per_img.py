import json
from collections import defaultdict

# Path to your COCO test file
coco_test_file = "/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/train.json"

# Load the COCO file
with open(coco_test_file, "r") as file:
    coco_data = json.load(file)

# Create a mapping of image_id to file_name
image_id_to_name = {image["id"]: image["file_name"] for image in coco_data.get("images", [])}

# Initialize annotation counts for all images
annotations_per_image = defaultdict(int)

# Count annotations for images with annotations
for annotation in coco_data.get("annotations", []):
    image_id = annotation["image_id"]
    annotations_per_image[image_id] += 1

# Ensure all images are included, even those without annotations
print("Number of annotations per image (by file name):")
for image_id, file_name in image_id_to_name.items():
    count = annotations_per_image[image_id]  # Default to 0 for images without annotations
    print(f"Image Name {file_name}: {count}")

