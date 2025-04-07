# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from label_studio_ml.model import LabelStudioMLBase, ModelResponse
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_data_dir, get_local_path

from botocore.exceptions import ClientError

from mmdet.apis import inference_detector, init_detector
from tools.slice_train_imgs import slice_train_images
from icecream import ic
from typing import List, Dict, Optional
from PIL import Image, ImageDraw
import numpy as np
import tempfile

## fit() related import
import tempfile
import torch
import gc
from mmengine.config import Config
from mmdet.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.apis import init_detector
import label_studio_sdk


from PIL import Image, ImageDraw
import numpy as np
import os
import tempfile

logger = logging.getLogger(__name__)

LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

ic(LABEL_STUDIO_HOST)
# TODO: Test memory bank code 
# TODO: Test cuda memory allocation and clearence

class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 score_threshold=0.3,
                 device='cpu',
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)
        self.labels_file = labels_file or os.environ.get('labels_file')
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            # mapping is mmdection label -> label studio label
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}
        # ic(self.label_map)
        # ic(self.parsed_label_config)
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'KeyPointLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        # ic(self.labels_attrs)
        # if labeling config has Label tags
        if self.labels_attrs:
            # try to find something like <Label value="Vehicle" predicted_values="airplane,car">
            for ls_label, label_attrs in self.labels_attrs.items():
                predicted_values = label_attrs.get(
                    "predicted_values", "").split(",")
                for predicted_value in predicted_values:
                    # remove spaces at the beginning and at the end
                    predicted_value = predicted_value.strip()
                    if predicted_value:  # it shouldn't be empty (like '')
                        # if predicted_value not in mmdet_labels:
                        #     print(
                        #         f'Predicted value "{predicted_value}" is not in mmdet labels'
                        #     )
                        self.label_map[predicted_value] = ls_label
        self.score_threshold = float(os.environ.get("SCORE_THRESHOLD", 0.3))
                        # Initialize memory bank with proper error handling
        # try:
        #     memory_bank = self.get('memory_bank')
        #     self.memory_bank = json.loads(memory_bank)
        #     # Convert list back to set for trained_task_ids
        #     self.memory_bank['trained_task_ids'] = set(self.memory_bank.get('trained_task_ids', []))
        # except (KeyError,TypeError):
        #     self.memory_bank = {'trained_task_ids': set()}
        # except Exception as e:
        #     logger.error(f"Error loading memory bank: {str(e)}")
        #     self.memory_bank = {'trained_task_ids': set()}
            
        self.max_memory_size = 1000
        # self.model_version = self.get("model_version")
        
    
    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def _get_prompt(self, annotation: Optional[Dict] = None) -> Dict:
        from_name_prompt, _, _ = self.get_first_tag_occurence('TextArea', 'Image')
        

        if annotation and 'result' in annotation:
            prompt = next(r['value']['text'][0] for r in annotation['result'] if r['from_name'] == from_name_prompt)
            logger.debug(f"Prompt: {prompt}")
            return {
                'prompt': prompt,
                'from_name': from_name_prompt
            }

        prompt = self.get('prompt')
        logger.debug(f"Prompt saved in cache: {prompt}")
        return {
            'prompt': prompt,
            'from_name': from_name_prompt
    }
    
    def _get_image_url(self, task: Dict) -> str:
        image_url = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        # retrieve image from s3 bucket,
        # set env vars AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        # and AWS_SESSION_TOKEN to allow boto3 to access the bucket
        if image_url.startswith("s3://") and os.getenv('AWS_ACCESS_KEY_ID'):
            # pre-sign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip("/")
            client = boto3.client("s3")
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": bucket_name, "Key": key},
                )
            except ClientError as exc:
                logger.warning(
                    f"Can't generate pre-signed URL for {image_url}. Reason: {exc}"
                )

        return image_url
        
    def predict(self, tasks, context: Optional[Dict] = None, **kwargs):
        
        # if 'model' not in globals():
        #     latest_ckpt = os.environ.get('checkpoint_file')
        #     device = os.environ.get("device", "cpu")
        #     config_file = os.environ.get('config_file')
        #     model = init_detector(config_file, latest_ckpt, device=device)
        config_file = os.environ['config_file']
        checkpoint_file = os.environ['checkpoint_file']
        device = os.environ.get("device", "cpu")
        model = init_detector(config_file, checkpoint_file, device=device)
        logger.info(f"Load new model from: {config_file}, {checkpoint_file}")
        # self.model_version = self.get("model_version")
        
        label_studio_results = []
        # ic(context)
        for task in tasks:
            # prompt_control = self._get_prompt(context)
            # prompt = prompt_control['prompt']
            # ic(prompt)
            # if not prompt:
                # logger.warning("Prompt not found")
                # ModelResponse(predictions=[])
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            model_results = inference_detector(model,
                                               image_path, text_prompt="female duck. male duck. Ice. Juvenile duck. duck", custom_entities=True).pred_instances
            results = []
            all_scores = []
            img_width, img_height = get_image_size(image_path)
            # print(f'>>> model_results: {model_results}')
            # print(f'>>> label_map {self.label_map}')
            # print(f'>>> model.dataset_meta: {model.dataset_meta}')
            classes = model.dataset_meta.get('classes')
            # classes = cfg.val_dataloader
            
            # ic(classes)
            # print(f'Classes >>> {classes}')
            for item in model_results:
                # print(f'item >>>>> {item}')
                bboxes, label, scores = item['bboxes'], item['labels'], item[
                    'scores']
                score = float(scores[-1])
                if score < self.score_threshold:
                    continue
                # print(f'bboxes >>>>> {bboxes}')
                # print(f'label >>>>> {label}')
                output_label = classes[list(
                    self.label_map.get(label, label))[0]]
                # print(f'>>> output_label: {output_label}')
                if output_label not in self.labels_in_config:
                    print(output_label + ' label not found in project config.')
                    continue

                for bbox in bboxes:
                    bbox = list(bbox)
                    if not bbox:
                        continue

                    # Convert bbox to center point
                    x_min, y_min, x_max, y_max = bbox[:4]
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2

                    results.append({
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'keypointlabels',  # Changed type
                        'value': {
                            'keypointlabels': [output_label],  # Changed key
                            'x': float(x_center) / img_width * 100,
                            'y': float(y_center) / img_height * 100,
                            "width": 0.4054054054054053,
                        },
                        'score': score,
                    })
                    all_scores.append(score)
            avg_score = sum(all_scores) / max(len(all_scores), 1)
            # print(f'>>> RESULTS: {results}')
            label_studio_results.append(
                {'result': results, 'score': round(avg_score, 3), 'model_version': self.get("model_version")})
            ic("Prediction returned successfully")
            # free gpu memory
            del model
            # Run garbage collection
            gc.collect()
            torch.cuda.empty_cache()
        return label_studio_results

    def fit(self, event, data, **kwargs):
        """Train MMDetection model with Label Studio annotations"""
        gc.collect()
        torch.cuda.empty_cache()
        # Environment configuration
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        config_file = os.environ['config_file']
        checkpoint_file = os.environ['checkpoint_file']
        device = os.environ.get("device", "cpu")
        # model = init_detector(config_file, checkpoint_file, device=device)
        logger.info(f"Load new model from: {config_file}, {checkpoint_file}")
        
        result = {
            'model_path': checkpoint_file,
            'checkpoints': [],
            'labels': self.labels_in_config,
            'error': None
        }
        try:
            # Validate event type
            if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
                logger.info(f"Skipping training for unsupported event: {event}")
                return result

            # Initialize Label Studio client
            ls = label_studio_sdk.Client(
                LABEL_STUDIO_HOST,
                LABEL_STUDIO_API_KEY
            )
            project = ls.get_project(data['project']['id'])
            self.project_id = str(data['project']['id'])
            self.set("model_version", self.get("model_version")) 
            
            
            # Get image key from labeling config
            # from_name, to_name, image_key = self.label_interface.get_first_tag_occurence('Image', 'Image')
            # project.get_tasks(selected_ids=[70665, 70664])
            # Fetch and validate tasks
            tasks = project.get_labeled_tasks()
            # Get existing trained tasks from memory bank
            self.memory_bank = self.get("memory_bank")
            self.memory_bank = json.loads(self.memory_bank)
            trained_tasks = self.memory_bank.get('trained_task_ids', set())
            ic(trained_tasks)
            # Filter out already trained tasks
            new_tasks = [task for task in tasks if str(task['id']) not in trained_tasks]
            for task in new_tasks:            
                ic(task['id'])
            if not new_tasks:
                # TODO: Add warning msg according to the api button implemented
                logger.warning("No new labeled tasks since last training\nPlease click on force re-train")
                return {
                    'model_path': checkpoint_file,
                    'checkpoints': [],
                    'labels': self.labels_in_config,
                    'error': "No new tasks to train on"
                }
            # Store current task IDs before training
            current_task_ids = {str(task['id']) for task in new_tasks}
            os.makedirs(os.path.join('work_dirs',project.title), exist_ok=True)
            
            # Training setup
            with tempfile.TemporaryDirectory(dir=os.path.join('work_dirs',project.title)) as temp_dir:
                # Convert tasks to COCO format
                coco_data = self._tasks_to_coco(new_tasks, temp_dir)
                # Save COCO annotations
                ann_file = os.path.join(temp_dir, 'train.json')
                with open(ann_file, 'w') as f:
                    json.dump(coco_data, f)

                # Configure MMDetection
                cfg = Config.fromfile(config_file)
                # Update model configuration
                cfg.model.bbox_head.num_classes = len(self.labels_in_config)
                cfg.num_queries = 2000
                cfg.test_cfg.max_per_img = 2000
                # TODO : Each project should have its own workdir
                # Config 1 : Work dir location
                cfg.work_dir = os.path.join('work_dirs',project.title)
                os.makedirs(cfg.work_dir, exist_ok=True)

                # Dataset configuration
                cfg.data_root = ''
                cfg.train_dataloader.batch_size=1
                cfg.train_dataloader.dataset.data_root = ''
                cfg.train_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.train_dataloader.num_workers = 0  # Disable multiprocessing
                cfg.train_dataloader.persistent_workers = False
                
                # Modify val paths
                cfg.val_dataloader.dataset.data_root = ''
                cfg.val_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.val_dataloader.num_workers = 0
                cfg.val_dataloader.persistent_workers = False
                cfg.val_dataloader.prefetch_factor = None
                cfg.val_evaluator.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.val_evaluator.outfile_prefix = os.path.join(
                    temp_dir, 'prediction_val')

                # Modify test paths
                cfg.test_dataloader.dataset.data_root = ''
                cfg.test_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.test_dataloader.num_workers = 0
                cfg.test_dataloader.persistent_workers = False
                cfg.test_dataloader.prefetch_factor = None
                cfg.test_evaluator.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.test_evaluator.outfile_prefix = os.path.join(
                    temp_dir, 'prediction_test')
                
                # Modify train slices path
                os.makedirs(os.path.join(temp_dir, 'slices'), exist_ok=True)
                cfg.data_root_slice = os.path.join(temp_dir, 'slices')
                # whole images data_root
                cfg.data_root_whole = ''

                # Training parameters
                cfg.train_cfg = dict(
                    type='EpochBasedTrainLoop',
                    max_epochs=10,
                    val_interval=1000000
                )
                cfg.default_hooks.checkpoint.interval = 10
                cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                cfg.load_from = os.environ['checkpoint_file']
                # Initialize and run training
                
                slice_configuration = cfg.get('slice_configuration')
                cfg = slice_train_images(cfg, **slice_configuration)
                cfg.log_level = 'ERROR'  # Add this line to suppress INFO/WARNING logs
                runner = Runner.from_cfg(cfg)
                sucess = False
                try: 
                    runner.train()
                    sucess = True
                    # gc.collect()
                    # torch.cuda.empty_cache()         
                    # Only update memory bank if training was successful
                    # model = init_detector(cfg, os.environ['checkpoint_file'], device=cfg.device)
                except Exception as e:
                    logger.error(f"Training failed: {str(e)}", exc_info=True)    
                    gc.collect()
                    torch.cuda.empty_cache()
                    result['error'] = str(e)
                    
                                # Handle checkpoints
                
                if(sucess):
                    with open(os.path.join(cfg.work_dir, 'last_checkpoint'), 'r') as f:
                        latest_ckpt = f.read().strip()
                        result['model_path'] = latest_ckpt
                        result['checkpoints'].append(latest_ckpt)
                        checkpoint_file = latest_ckpt
                        self._update_memory_bank(current_task_ids)
                        gc.collect()
                        torch.cuda.empty_cache()
                        # model = init_detector(cfg, latest_ckpt, device=device)
                        
                        # set the environment variable to use the new model
                        # self.bump_model_version()
                        # self.model_version = self.get("model_version")
                        
                        os.environ['checkpoint_file'] = latest_ckpt
                        logger.info("Training completed successfully")
                        ic("Training completed successfully")
                else:
                    gc.collect()
                    torch.cuda.empty_cache()
                    # raise RuntimeError("Training failed - no checkpoint created")


            return result

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            result['error'] = str(e)
            gc.collect()
            torch.cuda.empty_cache()
            # model = init_detector(cfg, os.environ['checkpoint_file'], device)
        
        return result

    def force_fit(self, event, data, **kwargs):
        """Train MMDetection model with Label Studio annotations"""
        # Environment configuration
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        config_file = os.environ['config_file']
        checkpoint_file = os.environ['checkpoint_file']
        device = os.environ.get("device", "cpu")
        # model = init_detector(config_file, checkpoint_file, device=device)
        ic("Called force fit function")
        logger.info(f"Load new model from: {config_file}, {checkpoint_file}")
        
        result = {
            'model_path': checkpoint_file,
            'checkpoints': [],
            'labels': self.labels_in_config,
            'error': None
        }
        try:
            # Initialize Label Studio client
            ls = label_studio_sdk.Client(
                LABEL_STUDIO_HOST,
                LABEL_STUDIO_API_KEY
            )
            project = ls.get_project(data['project']['id'])
            self.project_id = str(data['project']['id'])
            
            # Get image key from labeling config
            # from_name, to_name, image_key = self.label_interface.get_first_tag_occurence('Image', 'Image')
            # project.get_tasks(selected_ids=[70665, 70664])
            # Fetch and validate tasks
            tasks = data['tasks']
            current_task_ids = {str(task['id']) for task in tasks}
            # Store current task IDs before training
            os.makedirs(os.path.join('work_dirs',project.title), exist_ok=True)
            
            # Training setup
            with tempfile.TemporaryDirectory(dir=os.path.join('work_dirs',project.title)) as temp_dir:
                # Convert tasks to COCO format
                coco_data = self._tasks_to_coco(tasks, temp_dir)
                # Save COCO annotations
                ann_file = os.path.join(temp_dir, 'train.json')
                with open(ann_file, 'w') as f:
                    json.dump(coco_data, f)

                # Configure MMDetection
                cfg = Config.fromfile(config_file)
                # Update model configuration
                cfg.model.bbox_head.num_classes = len(self.labels_in_config)
                cfg.num_queries = 2000
                cfg.test_cfg.max_per_img = 2000
                # TODO : Each project should have its own workdir
                # Config 1 : Work dir location
                cfg.work_dir = os.path.join('work_dirs',project.title)
                os.makedirs(cfg.work_dir, exist_ok=True)

                # Dataset configuration
                cfg.data_root = ''
                cfg.train_dataloader.batch_size=1
                cfg.train_dataloader.dataset.data_root = ''
                cfg.train_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.train_dataloader.num_workers = 0  # Disable multiprocessing
                cfg.train_dataloader.persistent_workers = False
                
                # Modify val paths
                cfg.val_dataloader.dataset.data_root = ''
                cfg.val_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.val_dataloader.num_workers = 0
                cfg.val_dataloader.persistent_workers = False
                cfg.val_dataloader.prefetch_factor = None
                cfg.val_evaluator.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.val_evaluator.outfile_prefix = os.path.join(
                    temp_dir, 'prediction_val')

                # Modify test paths
                cfg.test_dataloader.dataset.data_root = ''
                cfg.test_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.test_dataloader.num_workers = 0
                cfg.test_dataloader.persistent_workers = False
                cfg.test_dataloader.prefetch_factor = None
                cfg.test_evaluator.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.test_evaluator.outfile_prefix = os.path.join(
                    temp_dir, 'prediction_test')
                
                # Modify train slices path
                os.makedirs(os.path.join(temp_dir, 'slices'), exist_ok=True)
                cfg.data_root_slice = os.path.join(temp_dir, 'slices')
                # whole images data_root
                cfg.data_root_whole = ''

                # Training parameters
                cfg.train_cfg = dict(
                    type='EpochBasedTrainLoop',
                    max_epochs=10,
                    val_interval=1000000
                )
                cfg.default_hooks.checkpoint.interval = 10
                cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                cfg.load_from = os.environ['checkpoint_file']
                # Initialize and run training
                
                slice_configuration = cfg.get('slice_configuration')
                cfg = slice_train_images(cfg, **slice_configuration)
                cfg.log_level = 'ERROR'  # Add this line to suppress INFO/WARNING logs
                runner = Runner.from_cfg(cfg)
                sucess = False
                try: 
                    runner.train()
                    sucess = True
                    # gc.collect()
                    # torch.cuda.empty_cache()         
                    # Only update memory bank if training was successful
                    # model = init_detector(cfg, os.environ['checkpoint_file'], device=cfg.device)
                except Exception as e:
                    logger.error(f"Training failed: {str(e)}", exc_info=True)    
                    gc.collect()
                    torch.cuda.empty_cache()
                    result['error'] = str(e)
                    
                                # Handle checkpoints
                
                if(sucess):
                    with open(os.path.join(cfg.work_dir, 'last_checkpoint'), 'r') as f:
                        latest_ckpt = f.read().strip()
                        result['model_path'] = latest_ckpt
                        result['checkpoints'].append(latest_ckpt)
                        checkpoint_file = latest_ckpt
                        self._update_memory_bank(current_task_ids)
                        gc.collect()
                        torch.cuda.empty_cache()
                        # model = init_detector(cfg, latest_ckpt, device=device)
                        
                        # set the environment variable to use the new model
                        self.bump_model_version()
                        os.environ['checkpoint_file'] = latest_ckpt
                        logger.info("Training completed successfully")
                        ic("Training completed successfully")
                else:
                    gc.collect()
                    torch.cuda.empty_cache()
                    # raise RuntimeError("Training failed - no checkpoint created")

            return result

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            result['error'] = str(e)
            gc.collect()
            torch.cuda.empty_cache()
            # model = init_detector(cfg, os.environ['checkpoint_file'], device)
        
        return result


    def _update_memory_bank(self, new_task_ids):
        """Update memory bank with new task IDs, maintaining max size"""
        current_trained = self.memory_bank.get('trained_task_ids', set())
        current_trained = set(current_trained)
        new_task_ids = set(new_task_ids)
        # Add new task IDs
        updated_trained = current_trained.union(new_task_ids)
        
        # Maintain max size by keeping most recent
        if len(updated_trained) > self.max_memory_size:
            updated_trained = set(list(updated_trained)[-self.max_memory_size:])
            
        self.memory_bank['trained_task_ids'] = updated_trained
        
        # Store as JSON string with list conversion for sets
        memory_bank_to_store = {
            'trained_task_ids': list(updated_trained)
        }
        self.set('memory_bank', json.dumps(memory_bank_to_store))
        logger.info(f"Memory bank updated. Now contains {len(updated_trained)} trained tasks")
    
    
    def _tasks_to_coco(self, tasks, temp_dir):
        """Convert Label Studio tasks to COCO format"""
        # Invert label map
        self.label_map_inverted = {v: k for k, v in self.label_map.items()}
        self.label_index = {l: i for i, l in enumerate(self.label_map_inverted.values())}
        
        coco = {
            'images': [],
            'annotations': [],
            'categories': [{'id': i, 'name': l} for i, l in enumerate(self.label_map_inverted.values())]
        }
        
        ann_id = 1
        for task_id, task in enumerate(tasks):
            try:
                # Handle image data
                img_path = self.get_local_path(task['data']['image'])
                if not os.path.exists(img_path):
                    continue
                    
                height, width = get_image_size(img_path)
                
                # Load the image
                img = Image.open(img_path).convert("RGB")
                
                # Check the user's choice for training region
                train_on_whole_image = True  # Default to whole image
                for ann in task.get('annotations', []):
                    for result in ann.get('result', []):
                        if result['type'] == 'choices' and result['from_name'] == 'train_region_choice':
                            if result['value']['choices'][0] == 'Train only on train region':
                                train_on_whole_image = False
                            break
                    if not train_on_whole_image:
                        break
                
                if train_on_whole_image:
                    # Use the whole image
                    cropped_img = img
                    cropped_img_path = img_path
                else:
                    # Extract polygon coordinates for "Train region"
                    train_region_polygon = None
                    for ann in task.get('annotations', []):
                        for result in ann.get('result', []):
                            if result['type'] == 'polygonlabels' and 'Train region' in result['value'].get('polygonlabels', []):
                                train_region_polygon = result['value']['points']
                                break
                        if train_region_polygon:
                            break
                    
                    if not train_region_polygon:
                        continue  # Skip if no train region is defined
                    
                    # Convert polygon coordinates to a mask
                    mask = Image.new('L', (width, height), 0)
                    ImageDraw.Draw(mask).polygon([(int(x * width / 100), int(y * height / 100)) for x, y in train_region_polygon], outline=1, fill=1)
                    mask = np.array(mask)
                    
                    # Apply the mask to the image to extract the polygon region
                    img_array = np.array(img)
                    masked_img_array = np.zeros_like(img_array)
                    masked_img_array[mask > 0] = img_array[mask > 0]
                    
                    # Find the bounding box of the polygon to crop the image
                    coords = np.column_stack(np.where(mask > 0))
                    if coords.size == 0:
                        continue  # Skip if the polygon is invalid or empty
                    min_y, min_x = coords.min(axis=0)
                    max_y, max_x = coords.max(axis=0)
                    
                    # Crop the masked image to the polygon region
                    cropped_img_array = masked_img_array[min_y:max_y, min_x:max_x]
                    cropped_img = Image.fromarray(cropped_img_array)
                    
                    # Save the cropped image to a temporary directory
                    cropped_img_path = os.path.join(temp_dir, f'cropped_{task_id}.jpg')
                    cropped_img.save(cropped_img_path)
                
                # Update COCO image entry
                coco['images'].append({
                    'id': task_id,
                    'file_name': cropped_img_path,
                    'width': cropped_img.width,
                    'height': cropped_img.height
                })
                
                # Process annotations (same as before)
                for ann in task.get('annotations', []):
                    height_bbox = float(ann.get('result', [])[-2].get('value')['text'][0])
                    width_bbox = float(ann.get('result', [])[-1].get('value')['text'][0])
                    
                    for result in ann.get('result', []):
                        if result['type'] != 'keypointlabels':
                            continue

                        label = result['value']['keypointlabels'][0]
                        if label not in self.label_map_inverted.keys():
                            continue
                        
                        # Convert to COCO bbox format
                        x_pct = result['value']['x'] / 100
                        y_pct = result['value']['y'] / 100
                        w_pct = height_bbox / 100
                        h_pct = width_bbox / 100
                        
                        bbox = [
                            x_pct * width - w_pct * width / 2,
                            y_pct * height - h_pct * height / 2,
                            w_pct * width,
                            h_pct * height
                        ]
                        
                        # Check if the bbox is within the polygon (if applicable)
                        if not train_on_whole_image:
                            bbox_mask = Image.new('L', (width, height), 0)
                            ImageDraw.Draw(bbox_mask).rectangle([
                                int(bbox[0]), int(bbox[1]),
                                int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                            ], outline=1, fill=1)
                            bbox_mask = np.array(bbox_mask)
                            
                            if not np.any(mask & bbox_mask):
                                continue  # Skip if the bbox is outside the train region
                        
                        # Adjust bbox coordinates to the cropped image
                        adjusted_bbox = [
                            bbox[0] - (min_x if not train_on_whole_image else 0),
                            bbox[1] - (min_y if not train_on_whole_image else 0),
                            bbox[2],
                            bbox[3]
                        ]
                        
                        # Check if the adjusted bbox is within the cropped image
                        if (adjusted_bbox[0] >= 0 and adjusted_bbox[1] >= 0 and
                            adjusted_bbox[0] + adjusted_bbox[2] <= cropped_img.width and
                            adjusted_bbox[1] + adjusted_bbox[3] <= cropped_img.height):
                            coco['annotations'].append({
                                'id': ann_id,
                                'image_id': task_id,
                                'category_id': self.label_index.get(label),
                                'bbox': adjusted_bbox,
                                'area': (w_pct * width) * (h_pct * height),
                                'iscrowd': 0
                            })
                            ann_id += 1

            except Exception as e:
                logger.error(f"Error processing task {task_id}: {e}")
                
        return coco

    def clear_memory_bank(self,):
        """Reset the memory bank of trained task IDs"""
        try:
            # Clear both the in-memory and stored memory bank
            # self.memory_bank['trained_task_ids'] = set()
            # Update the persistent storage with empty list
            self.set('memory_bank', json.dumps({'trained_task_ids': []}))
            logger.info("Successfully cleared memory bank")
            return True
        except Exception as e:
            logger.error(f"Failed to clear memory bank: {str(e)}", exc_info=True)
            return False

def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data

