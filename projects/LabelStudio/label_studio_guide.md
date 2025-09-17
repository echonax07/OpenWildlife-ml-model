# cd to directory 
cd /home/pc2041/VIP_lab/labelstudio/mmwhale2 && source /home/pc2041/env_labelstudio2/bin/activate


# Start Label studio

`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041 label-studio --data-dir "/home/pc2041/.local/share/label-studio"`

ML_TIMEOUT_TRAIN=300000 ML_TIMEOUT_PREDICT=300000 LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 label-studio --data-dir "/home/pc2041/.local/share/label-studio"


# Start ML backend with gunicorn (use for production setup)

LOG_LEVEL=INFO device=cuda LABEL_STUDIO_HOST=http://localhost:8080 LABEL_STUDIO_API_KEY=e5fccfb4114847bb3dd53b014c87901f5a5417af checkpoint_file="work_dirs/mm_grounding_dino_real_filtered_epoch10/epoch_50.pth" model_params_file=projects/LabelStudio/backend_template/model_params_eider.json config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10.py LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 gunicorn  --bind :8003 --workers 1 --threads 1 --timeout 0 _wsgi:app

# Run ML backend on local

`cd /home/pc2041/VIP_lab/labelstudio/mmwhale2 && source /home/pc2041/env_labelstudio2/bin/activate`

`device=cuda LABEL_STUDIO_HOST=http://localhost:8080 LABEL_STUDIO_API_KEY=e5fccfb4114847bb3dd53b014c87901f5a5417af checkpoint_file=work_dirs/mm_grounding_dino_real_10_imgs_fully_manual_1024/epoch_50.pth model_params_file=projects/LabelStudio/backend_template/model_params.json config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10.py LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041 label-studio-ml start projects/LabelStudio/backend_template --port 8003`

# Run the redis server and rq workers
In a separate terminal, run the resis server:
`redis-server`

In another terminal, run the rq workers for training and prediction:

`LOG_LEVEL=INFO device=cuda LABEL_STUDIO_HOST=http://localhost:8080 LABEL_STUDIO_API_KEY=e5fccfb4114847bb3dd53b014c87901f5a5417af checkpoint_file="work_dirs/mm_grounding_dino_real_filtered_epoch10/epoch_50.pth" model_params_file=projects/LabelStudio/backend_template/model_params_eider.json config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10.py LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 rq worker train`

`rq worker predict`

# Run ML backend and load checkpoint from AES Labelling Part 1
device=cuda LABEL_STUDIO_HOST=http://localhost:8080 LABEL_STUDIO_API_KEY=e5fccfb4114847bb3dd53b014c87901f5a5417af checkpoint_file="/home/pc2041/VIP_lab/labelstudio/mmwhale2/work_dirs/AES Labelling Part 1/checkpoint_20250702_161927.pth" model_params_file=projects/LabelStudio/backend_template/model_params.json config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10.py LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/pc2041 label-studio-ml start projects/LabelStudio/backend_template --port 8003

`device=cuda  checkpoint_file=/home/pc2041/VIP_lab/mmwhale2/work_dirs/grouding_dino_swin-t_vis_caption/epoch_10.pth config_file=/home/pc2041/VIP_lab/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/demo_config_small_pipeline.py LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041 label-studio-ml start projects/LabelStudio/backend_template --with config_file=/home/pc2041/VIP_lab/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/demo_config_small_pipeline.py checkpoint_file=/home/pc2041/VIP_lab/mmwhale2/work_dirs/grouding_dino_swin-t_vis_caption/epoch_10.pth device=cuda --port 8003`

checkpoint_file=work_dirs/faster_rcnn_clean_v2/best_coco_bbox_mAP_epoch_10.pth config_file=configs/whale/faster_rcnn_clean_v2.py checkpoint_file=work_dirs/faster_rcnn_clean_v2/best_coco_bbox_mAP_epoch_10.pth LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041 label-studio-ml start projects/LabelStudio/backend_template --with config_file=configs/whale/faster_rcnn_clean_v2.py checkpoint_file=work_dirs/faster_rcnn_clean_v2/best_coco_bbox_mAP_epoch_10.pth device=cuda --port 8003


# Run ML backend on narval

`LABEL_STUDIO_HOST=http://localhost:8080 LABEL_STUDIO_API_KEY=6c4e47ba68bd5b4a23ab09ef75ccddb7e7c74d43 checkpoint_file=work_dirs/mm_grounding_dino_real_filtered_epoch10/epoch_50.pth config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10.py LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/m32patel label-studio-ml start projects/LabelStudio/backend_template --with config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10.py checkpoint_file=work_dirs/mm_grounding_dino_real_filtered_epoch10/epoch_50.pth device=cuda --port 8003`

# Command to convert coco GT to label studio

`label-studio-converter import coco -i /media/pc2041/data/vip_lab/whale/whale_data/whale_data_simple_format_all/group_4/test/annotation_coco.json -o /home/pc2041/narval_16_test2.json --image-root-url /data/local-files/?d=data/vip_lab/whale/whale_data/whale_data_simple_format_all/group_4/test/`

Replace `data/vip_lab/whale/whale_data/whale_data_simple_format_all/group_4/test/` in the  `image-root-url` with your actual image root path

All of this assumes that the home folder is  `/home/pc2041`

** Important ** Dont forget to add the folder as local storage first


# Command to convert mmdet pred to label studio

For test data, the coco format does not have ground truth (GT), and the json file only contains image info and Categories. This file will be called annotation-info.json

The prediction file contains predictions as a list and it has predictions with image_id which are borrowed from annoatation-info.json. The prediction file will be called pred.json

So now these two files (annotation-info.json, pred.json) are required to produce label-studio task format.

There's a python script written called [convert_mmdet_pred_to_labelstudio_tasks.py](/tools/dataset_converters/whale/convert_mmdet_pred_to_labelstudio_tasks.py)

To run this script 

`python convert_mmdet_pred_to_labelstudio_tasks.py <Dataset_folder_path> <path_to_GT_JSON> <path_to_Pred_json> <output_file_path> --LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<labelstudio_local_file_root> ` 

e.g

`python convert_mmdet_pred_to_labelstudio_tasks.py /media/pc2041/data/vip_lab/whale/whale_data/merged/test/ /media/pc2041/data/vip_lab/whale/whale_data/merged/test/test_whale_non_whale.json /media/pc2041/data/vip_lab/whale/mmwhale2/work_dirs/faster_rcnn_clean_v2_pipeline4/test_results_whale_non_whale.bbox.json /media/pc2041/data/vip_lab/whale/mmwhale2/work_dirs/faster_rcnn_clean_v2_pipeline4/test_results_whale_non_whale_labelstudio.json --LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041`

# Convert mmdet GT and pred to labelstudio 

There's a python script written called [convert_mmdet_GT_pred_to_labelstudio_tasks.py](/tools/dataset_converters/whale/convert_mmdet_GT_pred_to_labelstudio_tasks.py)

`python convert_mmdet_GT_pred_to_labelstudio_tasks.py <Dataset_folder_path> <path_to_GT_JSON> <path_to_Pred_json> <output_file_path> --LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<labelstudio_local_file_root> ` 

e.g

`python convert_mmdet_GT_pred_to_labelstudio_tasks.py /media/pc2041/data/vip_lab/whale/whale_data/merged/test/ /media/pc2041/data/vip_lab/whale/whale_data/merged/test/test_whale_non_whale.json /media/pc2041/data/vip_lab/whale/mmwhale2/work_dirs/faster_rcnn_clean_v2_pipeline4/test_results_whale_non_whale.bbox.json /media/pc2041/data/vip_lab/whale/mmwhale2/work_dirs/faster_rcnn_clean_v2_pipeline4/test_results_whale_non_whale_labelstudio.json --LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041`



# Working URL to dataset

http://localhost:8080/data/local-files/?d=data/vip_lab/dataset/whale_data/whale_only_images/ES_AI_Narwhal_2016/2016_08_07%20Narwhal/ES_20160807_25mm_0000195.jpg


# Label map file

Key means label in label studio and value means label in mmdetection
```JSON
{
    "female duck": "female duck",
    "male duck": "male duck",
    "Ice": "Ice",
    "Juvenile duck": "Juvenile duck",
    "duck": "duck"
  }
```