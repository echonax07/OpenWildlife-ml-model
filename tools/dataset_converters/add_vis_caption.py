import json
import random

caption_file = "/home/m32patel/scratch/animal_patches/2015_Beluga/train/train_od_vis_desc_grounded_modified.json"
data_file = "/home/m32patel/scratch/animal_patches/2017_Beluga/train/train_full_patch_od.json"
output_file = "/home/m32patel/scratch/animal_patches/2017_Beluga/train/train_od_vis_desc_grounded_modified.json"

# Step 1: Store all captions
captions = []
with open(caption_file, "r") as infile:
    for line in infile:
        data = json.loads(line)
        if "caption" in data:
            captions.append(data["caption"])

# Step 2: Read jsonl and add random captions
with open(data_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        if captions:
            data["caption"] = random.choice(captions)  # Assign random caption
        outfile.write(json.dumps(data) + "\n")
