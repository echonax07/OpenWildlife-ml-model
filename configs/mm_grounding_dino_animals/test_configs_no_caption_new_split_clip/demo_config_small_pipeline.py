_base_ = '../grouding_dino_swin-t_no_caption_new_split_clip.py'

data_root = '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_newmexico'
ann_file = 'test.json'
class_name = (
    "Elephant", "Seal", "Beluga whale", "Narwhal", "Yak", "Sheep", "Bird",
    "Brant", "Canada goose", "Gull", "Emperor goose", "Zebra", "Giraffe",
    "Penguin", "Sea lion", "Turtle", "Royal tern", "Caspian tern",
    "Slender-billed gull", "Gray-headed gull", "Great cormorant",
    "Great white pelican", "Cattle", "Camelus", "Kiang"
)

num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[
        (255, 0, 0),   # Elephant
        (30, 144, 255),   # Seal
        (50, 205, 50),   # Beluga whale
        (255, 215, 0),   # Narwhal
        (255, 140, 0),   # Yak
        (138, 43, 226),   # Sheep
        (0, 206, 209),   # Bird
        (220, 20, 60),   # Brant
        (127, 255, 0),   # Canada goose
        (139, 69, 19),   # Gull
        (255, 20, 147),   # Emperor goose
        (165, 42, 42),   # Zebra
        (184, 134, 11),   # Giraffe
        (0, 250, 154),   # Penguin
        (255, 69, 0),   # Sea lion
        (70, 130, 180),   # Turtle
        (154, 205, 50),   # Royal tern
        (106, 90, 205),   # Caspian tern
        (32, 178, 170),   # Slender-billed gull
        (205, 92, 92),   # Gray-headed gull
        (210, 105, 30),   # Great cormorant
        (244, 164, 96),   # Great white pelican
        (85, 107, 47),   # Cattle
        (112, 128, 144),   # Camelus
        (255, 99, 71)    # Kiang
    ]
)

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        metainfo=metainfo))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/' + ann_file,
                    outfile_prefix=f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_nocaption_new_split_clip')
test_evaluator = val_evaluator
pickle_file = f'./work_dir_grounding_dino/{{fileBasenameNoExtension}}/prediction_mm_grounding_dino_nocaption_new_split_clip'
