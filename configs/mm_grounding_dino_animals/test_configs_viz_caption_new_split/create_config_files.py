import os

# Directory to save the Python files
directory = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs'
os.makedirs(directory, exist_ok=True)

# List of filenames to create
file_names = [
    "Aerial_seabird_westafrica_od_dataset",
    "birds_izembek_lagoon_od_dataset",
    "michigan_od_dataset",
    "monash_od_dataset",
    "new_mexico_od_dataset",
    "palmyra_od_dataset",
    "penguins_od_dataset",
    "pfeifer_od_dataset",
    "seabirdwatch_od_dataset",
    "birds_poland_od_dataset",
    "qian_od_dataset",
    "aerial_livestock_dataset",
    "WAID_livestock_dataset",
    "AED_dataset",
    "Eikelboom_dataset",
    "NOAA_sealion_dataset",
    "turtle_dataset",
    "NOAA_artic_seal_dataset",
    "Beluga_2014_dataset",
    "Beluga_2015_dataset",
    "Narwhal_2016_dataset"
]

# Content for each file
content = """
_base_ = '../grouding_dino_swin-t_finetune_all.py'
lang_model_name = 'checkpoints/bert/bert-base-uncased'
data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins'
ann_file = 'test_viz_grounded.json'
class_name = ('penguin', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=ann_file,
        type = 'CocoDatasetWithCaption',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + ann_file)
test_evaluator = val_evaluator
"""

# Create each file with the specified content
for name in file_names:
    file_path = os.path.join(directory, f"{name}.py")
    with open(file_path, 'w') as file:
        file.write(content)

print(f"{len(file_names)} files created in '{directory}' directory.")
