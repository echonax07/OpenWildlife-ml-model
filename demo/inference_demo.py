from mmdet.apis import DetInferencer
import json

### model without finetuning
# Choose to use a config
config_path = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/finetune_configs/eiderduck_predict_on_real.py'

# Setup a checkpoint file to load
checkpoint = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/eiderduck_finetune_syn_plus_real/epoch_120.pth'

# Initialize the DetInferencer
inferencer = DetInferencer(model=config_path, weights=checkpoint, device= 'cuda:0')
result = inferencer('/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_survey_project/2008/grants_photos_1/100EOS5D/IMG_4427.JPG', out_dir='./output', pred_score_thr=0.3)


# from icecream import ic 
# ic(result)

# def convert_to_coco_format(predictions, image_id, score_threshold=0.5):
#     coco_predictions = []
#     for i in range(len(predictions['bboxes'])):
#         score = predictions['scores'][i]
#         if score >= score_threshold:
#             x_min, y_min, x_max, y_max = predictions['bboxes'][i]
#             width = x_max - x_min
#             height = y_max - y_min
#             coco_prediction = {
#                 "image_id": image_id,
#                 "bbox": [x_min, y_min, width, height],
#                 "score": score,
#                 "category_id": predictions['labels'][i]
#             }
#             coco_predictions.append(coco_prediction)
#     return coco_predictions


# image_id = 301
# score_threshold = 0
# coco_predictions = convert_to_coco_format(result['predictions'][0], image_id, score_threshold)
# def save_to_json(data, filename):
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=4)
        
# # Save to JSON file
# output_filename = '/home/m32patel/scratch/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/plotting.json'
# save_to_json(coco_predictions, output_filename)
