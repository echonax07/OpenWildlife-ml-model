import json

input_file = "/home/m32patel/scratch/animal_patches/2016_Narwhal/train/train_od_vis_desc_grounded.json"
output_file = "/home/m32patel/scratch/animal_patches/2016_Narwhal/train/train_od_vis_desc_grounded_modified.json"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)  # Parse JSONL line
        if "caption" in data:
            data["caption"] = data["caption"].replace("narwhal", "narwhal whales")
        outfile.write(json.dumps(data) + "\n")  # Write modified line back
