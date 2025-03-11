import json

# Load COCO annotation file
with open("/home/m32patel/scratch/animal_patches/Eikelboom/train/train.json", "r") as f:
    data = json.load(f)

# Define category ID mapping (swap Zebra and Elephant)
id_mapping = {1: 2, 2: 1}  # Zebra (1) -> Elephant (2), Elephant (2) -> Zebra (1)

# Update categories
for category in data["categories"]:
    if category["id"] in id_mapping:
        category["id"] = id_mapping[category["id"]]

# Update annotations
for annotation in data["annotations"]:
    if annotation["category_id"] in id_mapping:
        annotation["category_id"] = id_mapping[annotation["category_id"]]

# Save updated annotations
with open("/home/m32patel/scratch/animal_patches/Eikelboom/train/train_same_category_as_test.json", "w") as f:
    json.dump(data, f, indent=4)
