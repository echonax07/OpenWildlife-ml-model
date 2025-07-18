# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
import time
import shutil
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
from PIL import Image, ImageDraw, ImageOps  # Added ImageOps
import numpy as np
import tempfile
import math 
import matplotlib.colors as mcolors

# fit() related import
import tempfile
import torch
import gc
from mmengine.config import Config
from mmdet.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.apis import init_detector
import label_studio_sdk


# Added for polygon operations


logger = logging.getLogger(__name__)

try:
    
    from shapely.geometry import Polygon, Point, MultiPolygon

    from shapely.validation import make_valid

    from shapely.errors import GEOSException

    SHAPELY_AVAILABLE = True

except ImportError:

    SHAPELY_AVAILABLE = False

    logger.warning(
        "Shapely library not found. Polygon shrinking will not be available. pip install shapely")


DEFAULT_NUM_CKPTS_TO_KEEP = 10

LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

ic(LABEL_STUDIO_HOST)
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
        self.model_params_file = os.environ.get('model_params_file')
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        self.label_map = {}
        if self.labels_file and os.path.exists(self.labels_file):
            # mapping is mmdection label -> label studio label
            # self.label_map = json_load(self.labels_file)
            pass
        else:
            # self.label_map = {}
            pass
        # ic(self.label_map)
        # ic(self.parsed_label_config)
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'KeyPointLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        # create a dictionary from list values as key and value from schema
        self.label_map = {label: label for label in schema['labels']}
        self.labels_in_config = set(self.labels_in_config) # These are the labels defined in labelstudio
        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        
        # ic(self.labels_attrs)
        # if labeling config has Label tags
        # TODO: Remove label map completely and use something like <Label value="Vehicle" predicted_values="airplane,car"> to define the mapping
        # refer here: https://github.com/HumanSignal/label-studio/blob/bcec8fc4bf9b56165ac44e658fce91a1d3a495bc/docs/source/tutorials/object-detector.md?plain=1#L32m        # TODO: Change the thr back to 0.1
        self.score_threshold = float(os.environ.get("SCORE_THRESHOLD", 0.3))
        self.max_memory_size = 1000

    def setup(self):
        self.load_model_checkpoint_and_version()

    def get_versions(self):
        model_version_history = json.loads(self.get("model_version_history")) if self.has('model_version_history') else []
        major_model_versions = json.loads(self.get("major_model_versions")) if self.has('major_model_versions') else []
        all_model_versions = model_version_history + major_model_versions
        if all_model_versions:
            versions = [entry[0] for entry in all_model_versions]
            # Flip the list to get the latest version first
            versions = versions[::-1]
            return versions
        elif self.has('model_version'):
            model_version = self.get('model_version')
            return [model_version]
        else:
            return []

    def get_model_extra_params_config(self):
        with open(os.path.join(self.model_params_file), 'r') as f:
            extra_params = json.load(f)
        return extra_params

    def _get_prompt(self, annotation: Optional[Dict] = None) -> Dict:
        from_name_prompt, _, _ = self.get_first_tag_occurence(
            'TextArea', 'Image')

        if annotation and 'result' in annotation:
            prompt = next(r['value']['text'][0]
                          for r in annotation['result'] if r['from_name'] == from_name_prompt)
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
        config_file = os.environ['config_file']
        checkpoint_file = os.environ['checkpoint_file']
        
        # TODO: delete below line later
        # checkpoint_file = "work_dirs/AES Labelling Part 1/checkpoint_20250710_040905.pth"
        # TODO: Replace this later
        ic(self.get("extra_params"))
        extra_params = json.loads(self.get("extra_params"))
        # extra_params = self.get_model_extra_params_config()
        
        cfg = Config.fromfile(config_file)
        test_patch_size = tuple(map(int, extra_params.get("test_patch_size","(1024,1024)").strip("()").split(",")))
        
        cfg.model.bbox_head.num_classes = len(self.labels_in_config)
        cfg.model.sliding_window_inference.patch_size = test_patch_size
        cfg.model.sliding_window_inference.slice_batch_size = int(extra_params.get("test_slice_batch_size", 4))
        cfg.model.test_cfg.max_per_img = int(extra_params.get("max_per_img", 900))
        cfg.model.num_queries = int(extra_params.get("num_queries",900))
        ic(f"PREDICTING WITH MODEL VERSION {self.model_version}")
        ic(checkpoint_file)
        ic(os.getcwd())
        device = os.environ.get("device", "cpu")
        ic(device)
        metainfo = dict(
        classes=tuple(self.labels_attrs.keys()),
        palette=[
        tuple(int(255 * c) for c in mcolors.to_rgb(attr["background"]))
        for attr in self.labels_attrs.values()])
        cfg.train_dataloader.dataset.metainfo=metainfo
        cfg.val_dataloader.dataset.metainfo=metainfo
        cfg.test_dataloader.dataset.metainfo=metainfo
        # TODO: Load sliding window_patch size from the model settings JSON.
        model = init_detector(cfg, checkpoint_file, device=device)
        
        # Get polygon label keys from config
        poly_from_name, poly_to_name, _, _ = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image'
        )
        label_studio_results = []
        
        for task in tasks:
            image_url = self._get_image_url(task)
            ic(image_url)
            image_path = self.get_local_path(image_url)
            ic(image_path)
            img_width, img_height = get_image_size(image_path)
            
            # Get existing annotations and train regions
            existing_annotations = []
            train_regions = []
            with open(os.path.join('work_dirs', 'task_predict.json'), 'w') as f:
                    json.dump(tasks, f)

            
            # Parse annotations to find train regions and existing points
            num_train_regions = 0
            for ann in task.get('annotations', []):
                for result in ann.get('result', []):
                    # Collect train regions
                    if result.get('type') == 'polygonlabels' and 'Train region' in result.get('value', {}).get('polygonlabels', []):
                        train_regions.append({
                            'points': result['value']['points'],
                            'original_width': result.get('original_width', img_width),
                            'original_height': result.get('original_height', img_height)
                        })
                        num_train_regions += 1
            
            for ann in task.get('annotations', []):
                for result in ann.get('result', []):
                    # Collect existing keypoint annotations
                    if result.get('type') == 'keypointlabels':
                        # Check if the keypoint is within the train regions
                        if result['value'].get('x') is None or result['value'].get('y') is None:
                            continue
                        x_percent = result['value']['x']
                        y_percent = result['value']['y']
                        x_abs = x_percent / 100 * img_width
                        y_abs = y_percent / 100 * img_height
                        in_train_region = False
                        if SHAPELY_AVAILABLE:
                            point = Point(x_abs, y_abs)
                            for region in train_regions:
                                # Convert percentage points to absolute coordinates
                                abs_points = [
                                    (p[0]/100 * region['original_width'], 
                                    p[1]/100 * region['original_height']) 
                                    for p in region['points']
                                ]
                                poly = Polygon(abs_points)
                                if poly.contains(point):
                                    in_train_region = True
                                    break
                        
                        if in_train_region:
                            label = result['value'].get('keypointlabels', [None])[0]
                            if label in self.label_map.keys():
                                # key means Labelstudio label, value means mmdetection label
                                existing_annotations.append({
                                    'x': result['value']['x'],
                                    'y': result['value']['y'],
                                    'label': label,
                                    'score': 1.0  # Existing annotations get max confidence
                                })
            text_prompt = '. '.join(self.label_map.values())
            
            ic(text_prompt)
            
            # Get model predictions
            model_results = inference_detector(model, image_path, text_prompt=text_prompt, custom_entities=True).pred_instances
            results = []
            all_scores = []
            
            # classes = model.dataset_meta.get('classes')
            classes=tuple(self.labels_attrs.keys())
            # Add train regions to results first
            for region in train_regions:
                results.append({
                    'from_name': poly_from_name,
                    'to_name': poly_to_name,
                    'type': 'polygonlabels',
                    'value': {
                        'points': region['points'],
                        'polygonlabels': ['Train region'],
                        'original_width': region['original_width'],
                        'original_height': region['original_height']
                    },
                    'score': 1.0  # Full confidence for existing regions
                })
                # all_scores.append(1.0)

            # Process model predictions
            for item in model_results:
                bboxes, label, scores = item['bboxes'], item['labels'], item['scores']
                score = float(scores[-1])
                if score < self.score_threshold:
                    continue
                output_label = classes[label.item()]
                if output_label not in self.labels_in_config:
                    continue

                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox[:4]
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    x_percent = x_center / img_width * 100
                    y_percent = y_center / img_height * 100

                    # Check if prediction is inside any train region
                    in_train_region = False
                    if SHAPELY_AVAILABLE:
                        point = Point(x_center, y_center)
                        for region in train_regions:
                            # Convert percentage points to absolute coordinates
                            abs_points = [
                                (p[0]/100 * region['original_width'], 
                                p[1]/100 * region['original_height']) 
                                for p in region['points']
                            ]
                            poly = Polygon(abs_points)
                            if poly.contains(point):
                                in_train_region = True
                                break

                    # Only keep predictions outside train regions
                    if not in_train_region:
                        results.append({
                            'from_name': self.from_name,
                            'to_name': self.to_name,
                            'type': 'keypointlabels',
                            'value': {
                                'keypointlabels': [output_label],
                                'x': float(x_percent),
                                'y': float(y_percent),
                                "width": 0.1,
                            },
                            'score': score,
                        })
                        all_scores.append(score)

            # Add existing annotations from train regions
            for ann in existing_annotations:
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'keypointlabels',
                    'value': {
                        'keypointlabels': [ann['label']],
                        'x': ann['x'],
                        'y': ann['y'],
                        "width": 0.4054054054054053,
                    },
                    'score': ann['score'],
                })
                all_scores.append(ann['score'])

            avg_score = sum(all_scores) / max(len(all_scores), 1)
            label_studio_results.append({
                'result': results,
                'score': round(avg_score, 3),
                'model_version': self.get("model_version")
            })
        ic("Prediction returned successfully")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return label_studio_results
    
    
    # this function does not do anything, so commented out to remove confusion
    # def fit(self, event, data, **kwargs):
    #     """Train MMDetection model with Label Studio annotations"""
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     # Environment configuration
    #     os.environ['OMP_NUM_THREADS'] = '2'
    #     os.environ['MKL_NUM_THREADS'] = '2'

    #     config_file = os.environ['config_file']
    #     checkpoint_file = os.environ['checkpoint_file']
    #     device = os.environ.get("device", "cpu")
    #     # model = init_detector(config_file, checkpoint_file, device=device)
    #     logger.info(f"Load new model from: {config_file}, {checkpoint_file}")

    #     result = {
    #         'model_path': checkpoint_file,
    #         'checkpoints': [],
    #         'labels': self.labels_in_config,
    #         'error': None
    #     }
    #     try:
    #         # Validate event type
    #         if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
    #             logger.info(
    #                 f"Skipping training for unsupported event: {event}")
    #             return result

    #         # Initialize Label Studio client
    #         ls = label_studio_sdk.Client(
    #             LABEL_STUDIO_HOST,
    #             LABEL_STUDIO_API_KEY
    #         )
    #         project = ls.get_project(data['project']['id'])
    #         self.project_id = str(data['project']['id'])
    #         self.set("model_version", self.get("model_version"))

    #         # Get image key from labeling config
    #         # from_name, to_name, image_key = self.label_interface.get_first_tag_occurence('Image', 'Image')
    #         # project.get_tasks(selected_ids=[70665, 70664])
    #         # Fetch and validate tasks
    #         tasks = project.get_labeled_tasks()
    #         # Get existing trained tasks from memory bank
    #         # self.memory_bank = self.get("memory_bank")
    #         # self.memory_bank = json.loads(self.memory_bank)
    #         # trained_tasks = self.memory_bank.get('trained_task_ids', set())
    #         # ic(trained_tasks)
    #         # Filter out already trained tasks
    #         # new_tasks = [task for task in tasks if str(
    #         #     task['id']) not in trained_tasks]
    #         for task in tasks:
    #             ic(task['id'])
    #         if not tasks:
    #             # TODO: Add warning msg according to the api button implemented
    #             logger.warning(
    #                 "No new labeled tasks since last training\nPlease click on force re-train")
    #             return {
    #                 'model_path': checkpoint_file,
    #                 'checkpoints': [],
    #                 'labels': self.labels_in_config,
    #                 'error': "No new tasks to train on"
    #             }
    #         # Store current task IDs before training
    #         current_task_ids = {str(task['id']) for task in tasks}
    #         os.makedirs(os.path.join(
    #             'work_dirs', project.title), exist_ok=True)

    #         # Training setup
    #         with tempfile.TemporaryDirectory(dir=os.path.join('work_dirs', project.title)) as temp_dir2:
    #             # Define a path manually for debugging
    #             temp_dir = os.path.join('work_dirs', project.title, 'debug_temp')
    #             os.makedirs(temp_dir, exist_ok=True)
    #             # Convert tasks to COCO format
    #             coco_data = self._tasks_to_coco(tasks, temp_dir)
    #             # Save COCO annotations
    #             ann_file = os.path.join(temp_dir, 'train.json')
    #             with open(ann_file, 'w') as f:
    #                 json.dump(coco_data, f)

    #             # Configure MMDetection
    #             cfg = Config.fromfile(config_file)
                
    #             # Load extra parameters from the DB
    #             # Update model configuration
    #             cfg.model.bbox_head.num_classes = len(self.labels_in_config)
    #             cfg.num_queries = extra_params.get("num_queries",2000)
    #             cfg.test_cfg.max_per_img = extra_params.get("max_per_img",2000)
                
    #             # TODO : Each project should have its own workdir
    #             # Config 1 : Work dir location
    #             cfg.work_dir = os.path.join('work_dirs', project.title)
    #             os.makedirs(cfg.work_dir, exist_ok=True)

    #             # Dataset configuration
    #             cfg.data_root = ''
    #             # cfg.train_dataloader.batch_size = 1
    #             cfg.train_dataloader.batch_size = extra_params.get("batch_size",1)
    #             cfg.train_dataloader.dataset.data_root = ''
    #             cfg.train_dataloader.dataset.ann_file = os.path.join(
    #                 temp_dir, 'train.json')
    #             cfg.train_dataloader.num_workers = 4  # Disable multiprocessing
    #             cfg.train_dataloader.persistent_workers = False
    #             cfg.train_dataloader.dataset.pipeline= cfg.train_pipeline

    #             # Modify val paths
    #             cfg.val_dataloader.dataset.data_root = ''
    #             cfg.val_dataloader.dataset.ann_file = os.path.join(
    #                 temp_dir, 'train.json')
    #             # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
    #             cfg.val_dataloader.num_workers = 4
    #             cfg.val_dataloader.persistent_workers = False
    #             cfg.val_dataloader.prefetch_factor = None
    #             cfg.val_evaluator.ann_file = os.path.join(
    #                 temp_dir, 'train.json')
    #             cfg.val_evaluator.outfile_prefix = os.path.join(
    #                 temp_dir, 'prediction_val')

    #             # Modify test paths
    #             cfg.test_dataloader.dataset.data_root = ''
    #             cfg.test_dataloader.dataset.ann_file = os.path.join(
    #                 temp_dir, 'train.json')
    #             # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
    #             cfg.test_dataloader.num_workers = 0
    #             cfg.test_dataloader.persistent_workers = False
    #             cfg.test_dataloader.prefetch_factor = None
    #             cfg.test_evaluator.ann_file = os.path.join(
    #                 temp_dir, 'train.json')
    #             cfg.test_evaluator.outfile_prefix = os.path.join(
    #                 temp_dir, 'prediction_test')

    #             # Modify train slices path
    #             os.makedirs(os.path.join(temp_dir, 'slices'), exist_ok=True)
    #             cfg.data_root_slice = os.path.join(temp_dir, 'slices')
    #             # whole images data_root
    #             cfg.data_root_whole = ''

    #             # Training parameters
    #             cfg.train_cfg = dict(
    #                 type='EpochBasedTrainLoop',
    #                 max_epochs=1,
    #                 val_interval=1000000
    #             )
    #             cfg.default_hooks.logger.interval=1
    #             cfg.default_hooks.checkpoint.interval = 10
    #             cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #             ic(os.environ['checkpoint_file'])
    #             cfg.load_from = os.environ['checkpoint_file']
    #             # Initialize and run training

    #             slice_configuration = cfg.get('slice_configuration')
    #             cfg = slice_train_images(cfg, **slice_configuration)
    #             cfg.log_level = 'ERROR'  # Add this line to suppress INFO/WARNING logs
    #             runner = Runner.from_cfg(cfg)
    #             sucess = False
    #             try:
    #                 runner.train()
    #                 sucess = True
    #                 # gc.collect()
    #                 # torch.cuda.empty_cache()
    #                 # Only update memory bank if training was successful
    #                 # model = init_detector(cfg, os.environ['checkpoint_file'], device=cfg.device)
    #             except Exception as e:
    #                 logger.error(f"Training failed: {str(e)}", exc_info=True)
    #                 gc.collect()
    #                 torch.cuda.empty_cache()
    #                 result['error'] = str(e)

    #                 # Handle checkpoints

    #             if (sucess):
    #                 with open(os.path.join(cfg.work_dir, 'last_checkpoint'), 'r') as f:
    #                     latest_ckpt = f.read().strip()
    #                     result['model_path'] = latest_ckpt
    #                     result['checkpoints'].append(latest_ckpt)
    #                     checkpoint_file = latest_ckpt
    #                     # self._update_memory_bank(current_task_ids)
    #                     gc.collect()
    #                     torch.cuda.empty_cache()
    #                     # model = init_detector(cfg, latest_ckpt, device=device)

    #                     # set the environment variable to use the new model
    #                     self.bump_model_version(latest_ckpt, max_ckpts=cfg.default_hooks.checkpoint.max_keep_ckpts)
    #                     # self.model_version = self.get("model_version")
    #                     logger.info("Training completed successfully")
    #                     ic("Training completed successfully")
    #             else:
    #                 gc.collect()
    #                 torch.cuda.empty_cache()
    #                 # raise RuntimeError("Training failed - no checkpoint created")

    #         return result

    #     except Exception as e:
    #         logger.error(f"Training failed: {str(e)}", exc_info=True)
    #         result['error'] = str(e)
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         # model = init_detector(cfg, os.environ['checkpoint_file'], device)

    #     return result

    def set_training_status(self, status: str):
        TRAINING_STATUSES = ["COMPLETE/IDLE", "SETUP", "IN PROGRESS", "ERROR"]
        if status not in TRAINING_STATUSES:
            return
        
        self.set("training_status", status)

    def force_fit(self, event, data, **kwargs):
        """Train MMDetection model with Label Studio annotations"""

        self.set_training_status("SETUP")

        # Environment configuration
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        config_file = os.environ['config_file']
        checkpoint_file = os.environ['checkpoint_file']
        # TODO: delete below line later
        # checkpoint_file = "work_dirs/AES Labelling Part 1/checkpoint_20250710_040905.pth"
        # os.environ['checkpoint_file'] = "work_dirs/mm_grounding_dino_real_filtered_epoch10/epoch_50.pth"
        device = os.environ.get("device", "cpu")
        # model = init_detector(config_file, checkpoint_file, device=device)
        ic("Called force fit function")
        # ic(os.environ['checkpoint_file'])
        logger.info(f"Load new model from: {config_file}, {checkpoint_file}")

        result = {
            'model_path': checkpoint_file,
            'checkpoints': [],
            'labels': list(self.labels_in_config),
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
            os.makedirs(os.path.join(
                'work_dirs', project.title), exist_ok=True)

            # with open(os.path.join('work_dirs', project.title,'task.json'), "w") as f:
            #         json.dump(tasks,f)
            # Training setup
            with tempfile.TemporaryDirectory(dir=os.path.join('work_dirs', project.title)) as temp_dir2:
                temp_dir = os.path.join('work_dirs', project.title, 'debug_temp')
                os.makedirs(temp_dir, exist_ok=True)
                # Convert tasks to COCO format
                coco_data = self._tasks_to_coco(tasks, temp_dir)
                # Save COCO annotations
                ann_file = os.path.join(temp_dir, 'train.json')
                with open(ann_file, 'w') as f:
                    json.dump(coco_data, f)

                # Configure MMDetection
                cfg = Config.fromfile(config_file)
                
                metainfo = dict(
                classes=tuple(self.labels_attrs.keys()),
                palette=[
                tuple(int(255 * c) for c in mcolors.to_rgb(attr["background"]))
                for attr in self.labels_attrs.values()])
                cfg.train_dataloader.dataset.metainfo=metainfo
                cfg.val_dataloader.dataset.metainfo=metainfo
                cfg.test_dataloader.dataset.metainfo=metainfo            
                # Load extra parameters from the DB
                extra_params = json.loads(self.get("extra_params"))
                # Update model configuration
                cfg.model.bbox_head.num_classes = len(self.labels_in_config)
                
                ic(extra_params.get("learning_rate"))
                cfg.optim_wrapper.optimizer.lr = float(extra_params.get("learning_rate",0.0002))
                
                ic(cfg.optim_wrapper.optimizer.lr)
                cfg.model.num_queries = int(extra_params.get("num_queries",900))
                cfg.model.test_cfg.max_per_img = int(extra_params.get("max_per_img",900))
                # TODO : Each project should have its own workdir
                # Config 1 : Work dir location
                cfg.work_dir = os.path.join('work_dirs', project.title)
                os.makedirs(cfg.work_dir, exist_ok=True)

                # Dataset configuration
                cfg.data_root = ''
                cfg.train_dataloader.batch_size = int(extra_params.get("train_batch_size",1))
                cfg.train_dataloader.dataset.data_root = ''
                cfg.train_dataloader.dataset.ann_file = os.path.join(
                    temp_dir, 'train.json')
                cfg.train_dataloader.num_workers = int(extra_params.get("train_num_workers",0))  # Disable multiprocessing
                cfg.train_dataloader.persistent_workers = extra_params.get("train_persistent_workers",False)
                cfg.train_dataloader.dataset.pipeline= cfg.train_pipeline2

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
                    max_epochs=int(extra_params.get("numEpochs",1)),
                    val_interval=1000000
                )
                cfg.default_hooks.logger.interval=1
                cfg.default_hooks.checkpoint.interval = 10
                cfg.device = device
                ic(os.environ['checkpoint_file'])
                cfg.load_from = os.environ['checkpoint_file']
                # Initialize and run training
                
                
                # Slicing based parameters
                train_patch_size = tuple(map(int, extra_params.get("train_patch_size","(1024,1024)").strip("()").split(",")))
                cfg.slice_configuration.slice_height= train_patch_size[0]
                cfg.slice_configuration.slice_width= train_patch_size[1]
                test_patch_size = tuple(map(int, extra_params.get("test_patch_size","(1024,1024)").strip("()").split(",")))
                cfg.model.sliding_window_inference.patch_size = test_patch_size
                cfg.model.sliding_window_inference.slice_batch_size = int(extra_params.get("test_slice_batch_size", 4))
                slice_configuration = cfg.get('slice_configuration')
                cfg = slice_train_images(cfg, **slice_configuration)
                cfg.log_level = 'ERROR'  # Add this line to suppress INFO/WARNING logs
                runner = Runner.from_cfg(cfg)
                sucess = False
                try:
                    self.set_training_status("IN PROGRESS")
                    print("Starting training...")
                    runner.train()
                    print("Training completed.")
                    sucess = True
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # Only update memory bank if training was successful
                    # model = init_detector(cfg, os.environ['checkpoint_file'], device=cfg.device)
                except Exception as e:
                    self.set_training_status("ERROR")
                    logger.error(f"Training failed: {str(e)}", exc_info=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    result['error'] = str(e)

                    # Handle checkpoints

                if (sucess):
                    self.set_training_status("COMPLETE/IDLE")
                    with open(os.path.join(cfg.work_dir, 'last_checkpoint'), 'r') as f:
                        latest_ckpt = f.read().strip()
                        # Rename the file to include the timestamp
                        latest_ckpt_dir = os.path.dirname(latest_ckpt)
                        new_checkpoint_path = os.path.join(latest_ckpt_dir, f"checkpoint_{time.strftime('%Y%m%d_%H%M%S')}.pth")
                        shutil.copyfile(latest_ckpt, new_checkpoint_path)
                        latest_ckpt = new_checkpoint_path
                        
                        result['model_path'] = latest_ckpt
                        result['checkpoints'].append(latest_ckpt)
                        checkpoint_file = latest_ckpt
                        # self._update_memory_bank(current_task_ids)
                        gc.collect()
                        torch.cuda.empty_cache()
                        # model = init_detector(cfg, latest_ckpt, device=device)

                        # set the environment variable to use the new model
                        self.bump_model_version(latest_ckpt, max_ckpts=cfg.default_hooks.checkpoint.max_keep_ckpts)
                        logger.info("Training completed successfully")
                        ic("Training completed successfully")
                else:
                    self.set_training_status("ERROR")
                    gc.collect()
                    torch.cuda.empty_cache()
                    # raise RuntimeError("Training failed - no checkpoint created")
            
            return result

        except Exception as e:
            self.set_training_status("ERROR")
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            result['error'] = str(e)
            gc.collect()
            torch.cuda.empty_cache()
            # model = init_detector(cfg, os.environ['checkpoint_file'], device)

        return result

    def load_model_checkpoint_and_version(self):
        if self.has('model_version'):
            self.model_version = self.get('model_version')
            model_version_history = json.loads(self.get('model_version_history')) if self.has('model_version_history') else []
            major_model_versions = json.loads(self.get('major_model_versions')) if self.has('major_model_versions') else []
            all_model_versions = model_version_history + major_model_versions
            if all_model_versions:
                # Find the checkpoint path for the current model version
                version_data = next(
                    (entry for entry in all_model_versions if entry[0] == self.model_version), None)
                if version_data:
                    os.environ['checkpoint_file'] = version_data[1] # This is what actually sets the checkpoint file that training and prediction will use
                    ic(f"Using model version: {self.model_version} with checkpoint: {version_data[1]}")
                    return

        # If model version not found, all_model_versions is empty, or model version can't be found in all_model_versions,
        # Set a default model version
        base_model_name = f"[BASE MODEL] {self.__class__.__name__}-v0.0.0"
        major_model_versions = [[base_model_name, os.environ['checkpoint_file']]]
        self.set("major_model_versions", json.dumps(major_model_versions))
        self.set("model_version", base_model_name)
        self.model_version = base_model_name

    def load_weights_from_path(self, path: str):
        """Load model weights from a specified path."""
        if not os.path.exists(path):
            logger.error(f"Checkpoint file {path} does not exist.")
            return False

        # Set the environment variable for the checkpoint file
        os.environ['checkpoint_file'] = path
        self.set("checkpoint_file", path)
        self.save_current_version_as("custom path")
        return True

    def save_current_version_as(self, name):
        partial_version_name = f"[{name.upper()}] {self.__class__.__name__}-"
        version_number = "v0.1.0" # Default version number

        major_model_versions = json.loads(self.get('major_model_versions')) if self.has('major_model_versions') else []
        if major_model_versions:
            last_version = major_model_versions[-1]
            last_version_number = last_version[0].split('-')[-1]
            # Increment the version number
            version_parts = last_version_number.split('.')
            version_parts[1] = str(int(version_parts[1]) + 1)  # Increment minor version
            version_number = '.'.join(version_parts)
        
        full_version_name = f"{partial_version_name}{version_number}"
        major_model_versions.append([full_version_name, os.environ['checkpoint_file']])
        self.set("major_model_versions", json.dumps(major_model_versions))

    def bump_model_version(self, checkpoint_file, max_ckpts=DEFAULT_NUM_CKPTS_TO_KEEP):
        if not os.path.exists(checkpoint_file):
            logger.error(f"Checkpoint file {checkpoint_file} does not exist.")
            return

        timestamp = os.path.getmtime(checkpoint_file)
        # Convert timestamp to a human-readable format
        timestamp_str = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        # Update the model version with the timestamp
        version_name = f"{self.__class__.__name__}-{timestamp_str}"
        model_version_history = json.loads(self.get("model_version_history")) if self.has('model_version_history') else []
        model_version_history.append([version_name, checkpoint_file])
        while len(model_version_history) > max_ckpts:
            remove_entry = model_version_history.pop(0)
            if os.path.exists(remove_entry[1]):
                os.remove(remove_entry[1])
                logger.info(f"Removed checkpoint: {remove_entry[1]}")
        self.set("model_version_history", json.dumps(model_version_history))
        self.set("latest_ckpt", json.dumps([version_name, checkpoint_file]))

        os.environ['checkpoint_file'] = checkpoint_file

        self.set("model_version", version_name)
        self.model_version = version_name


    def _update_memory_bank(self, new_task_ids):
        """Update memory bank with new task IDs, maintaining max size"""
        current_trained = self.memory_bank.get('trained_task_ids', set())
        current_trained = set(current_trained)
        new_task_ids = set(new_task_ids)
        # Add new task IDs
        updated_trained = current_trained.union(new_task_ids)

        # Maintain max size by keeping most recent
        if len(updated_trained) > self.max_memory_size:
            updated_trained = set(list(updated_trained)
                                  [-self.max_memory_size:])

        # self.memory_bank['trained_task_ids'] = updated_trained

        # Store as JSON string with list conversion for sets
        # memory_bank_to_store = {
            # 'trained_task_ids': list(updated_trained)
        # }
        # self.set('memory_bank', json.dumps(memory_bank_to_store))
        logger.info(
            f"Memory bank updated. Now contains {len(updated_trained)} trained tasks")

    def _tasks_to_coco(self, tasks, temp_dir):
        """Convert Label Studio tasks to COCO format, handling train regions."""
        if not SHAPELY_AVAILABLE:
            logger.error("Shapely is not installed, cannot process 'Train region' polygons or shrink them.")
            # Fallback or error - choosing to log and potentially skip region processing
            # Depending on requirements, you might want to raise an Exception here.

        # Invert label map for finding category IDs
        self.label_map_inverted = {v: k for k, v in self.label_map.items()}
        # Create label index map {label_name: category_id}
        self.label_index = {
            label_name: i
            for i, label_name in enumerate(self.label_map_inverted.values()) # Use labels from project config directly
        }
        # ic(self.label_index)
        coco = {
            'images': [],
            'annotations': [],
            'categories': [{'id': i, 'name': l} for i, l in enumerate(self.label_map_inverted.values())]
        }

        # ic(coco['categories'])
        
        ann_id = 1
        coco_img_id = 0 # Use a separate counter for COCO image IDs

        for task in tasks:
            task_id = task['id'] # Use the actual task ID for logging/filenames
            try:
                image_url = self._get_image_url(task)
                img_path = self.get_local_path(image_url)
                if not os.path.exists(img_path):
                    logger.warning(f"Image path not found for task {task_id}: {img_path}. Skipping task.")
                    continue

                original_width, original_height = get_image_size(img_path)
                img = Image.open(img_path).convert("RGB")

                # Determine if training on whole image or regions
                train_on_whole_image = True # Default
                train_region_polygons_percent = [] # Store multiple regions (as percentage points)

                for ann in task.get('annotations', []):
                    for result in ann.get('result', []):
                        # Check for the choice first
                        if result.get('type') == 'choices' and result.get('from_name') == 'train_region_choice':
                            choice = result.get('value', {}).get('choices', [])
                            if choice and choice[0] == 'Train only on train region':
                                train_on_whole_image = False
                        # Collect *all* Train region polygons regardless of choice (needed if choice is region)
                        elif result.get('type') == 'polygonlabels' and 'Train region' in result.get('value', {}).get('polygonlabels', []):
                             # Ensure points exist and are not empty
                            points = result['value'].get('points')
                            if points and len(points) >= 3: # Need at least 3 points for a polygon
                                train_region_polygons_percent.append(points)

                # If choice is 'Train only on train region' but no valid polygons found, skip task or log warning
                if not train_on_whole_image and not train_region_polygons_percent:
                    logger.warning(f"Task {task_id} set to 'Train only on train region' but no valid 'Train region' polygons found. Skipping region processing for this task.")
                    # Option 1: Skip the entire task
                    # continue
                    # Option 2: Fallback to training on the whole image (more robust)
                    train_on_whole_image = False
                    logger.warning(f"Falling back to training on whole image for task {task_id}.")


                # --- Process Annotations ---
                # Extract all relevant annotations from the task first
                task_annotations = []
                # Find required height/width text areas if they exist
                bbox_height_text = None
                bbox_width_text = None
                required_width = 0.5 # Default keypoint width %
                required_height = 0.5 # Default keypoint height %

                for ann in task.get('annotations', []):
                     # Look for text inputs defining bbox size (assuming specific from_names)
                     for result in ann.get('result', []):
                         if result.get('type') == 'textarea' and result.get('from_name') == 'height': # Adjust from_name if needed
                             try:
                                 # Assuming text contains a single number (percentage)
                                 bbox_height_text = float(result['value']['text'][0])
                                 required_height = bbox_height_text if bbox_height_text > 0 else required_height
                             except (ValueError, IndexError, KeyError):
                                 logger.warning(f"Task {task_id}: Could not parse bbox_height_px value.")
                         elif result.get('type') == 'textarea' and result.get('from_name') == 'width': # Adjust from_name if needed
                             try:
                                 bbox_width_text = float(result['value']['text'][0])
                                 required_width = bbox_width_text if bbox_width_text > 0 else required_width
                             except (ValueError, IndexError, KeyError):
                                 logger.warning(f"Task {task_id}: Could not parse bbox_width_px value.")

                     # Now process keypoints
                     for result in ann.get('result', []):
                        if result.get('type') == 'keypointlabels':
                            label = result['value'].get('keypointlabels', [None])[0]
                            # in the dictionary self.label_map, key means LS label and value means mmdetection label
                            mmdet_label = self.label_map.get(label, label) # Map label if needed
                        

                            # Use project config labels for checking and indexing
                            if mmdet_label not in self.label_index:
                                logger.warning(f"Task {task_id}: Label '{mmdet_label}' (original: '{label}') not found in project config labels: {list(self.label_index.keys())}. Skipping annotation.")
                                continue

                            category_id = self.label_index[mmdet_label]

                            # Convert keypoint center % to absolute pixel coordinates
                            x_pct = result['value']['x']
                            y_pct = result['value']['y']
                            center_x = x_pct * original_width / 100
                            center_y = y_pct * original_height / 100

                            # Get bbox dimensions from text inputs or defaults (as percentages)
                            w_pct = required_width # Use specific width if available
                            h_pct = required_height # Use specific height if available
                            
                            # Convert width/height % to absolute pixel dimensions
                            abs_w = w_pct * original_width / 100
                            abs_h = h_pct * original_height / 100

                            # Calculate COCO bbox [x_min, y_min, width, height] in absolute pixels
                            x_min = center_x - abs_w / 2
                            y_min = center_y - abs_h / 2

                            # Clip bbox coordinates to image boundaries
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            abs_w = min(original_width - x_min, abs_w)
                            abs_h = min(original_height - y_min, abs_h)

                            # Ensure width and height are positive
                            if abs_w <= 0 or abs_h <= 0:
                                logger.warning(f"Task {task_id}: Skipping annotation with non-positive dimensions for label '{mmdet_label}'. W={abs_w}, H={abs_h}")
                                continue


                            task_annotations.append({
                                'category_id': category_id,
                                'bbox_abs': [x_min, y_min, abs_w, abs_h], # Store absolute coords on original image
                                'center_abs': (center_x, center_y) # Store absolute center point
                            })


                # --- Generate COCO Data ---
                if train_on_whole_image:
                    # Use the whole image
                    # Add COCO image entry
                    coco['images'].append({
                        'id': coco_img_id,
                        'file_name': img_path, # Use original path
                        'width': original_width,
                        'height': original_height
                    })

                    # Add all annotations associated with this image
                    for ann_data in task_annotations:
                        coco['annotations'].append({
                            'id': ann_id,
                            'image_id': coco_img_id,
                            'category_id': ann_data['category_id'],
                            'bbox': ann_data['bbox_abs'], # Use original absolute bbox
                            'area': ann_data['bbox_abs'][2] * ann_data['bbox_abs'][3],
                            'iscrowd': 0
                        })
                        ann_id += 1
                    coco_img_id += 1 # Increment after processing the whole image

                elif SHAPELY_AVAILABLE: # Process regions only if shapely is available
                    processed_region = False
                    # Iterate through each detected train region polygon
                    for region_index, region_poly_percent in enumerate(train_region_polygons_percent):
                        try:
                            # Convert percentage points to absolute pixel coordinates
                            abs_points = [
                                (p[0] * original_width / 100, p[1] * original_height / 100)
                                for p in region_poly_percent
                            ]

                            if len(abs_points) < 3:
                                logger.warning(f"Task {task_id}, Region {region_index}: Skipping invalid polygon with < 3 points.")
                                continue

                            # Create Shapely Polygon
                            polygon = Polygon(abs_points)

                            # --- Feature 2: Shrink the polygon ---
                            # Calculate shrink distance: 10% of the smaller dimension of the polygon's bounding box
                            min_x, min_y, max_x, max_y = polygon.bounds
                            poly_width = max_x - min_x
                            poly_height = max_y - min_y

                            if poly_width <= 0 or poly_height <= 0:
                                logger.warning(f"Task {task_id}, Region {region_index}: Skipping polygon with non-positive dimensions.")
                                continue

                            # Use a buffer relative to polygon size, preventing excessive shrinking
                            # Heuristic: 5% of the diagonal length? or 10% of min dimension? Let's stick to 10% min dim.
                            # --- New fixed-pixel shrinking ---
                            fixed_shrink_pixels = 20  # Shrink 100 pixels from all edges
                            shrink_distance = fixed_shrink_pixels

                            # Perform negative buffer (shrinking)
                            # Use join_style=2 (MITRE) for sharper corners if desired
                            shrunk_polygon = polygon.buffer(-shrink_distance, join_style=2)

                            # Validate shrunk polygon
                            shrunk_polygon = make_valid(shrunk_polygon) # Ensure geometric validity

                            if shrunk_polygon.is_empty or not shrunk_polygon.is_valid:
                                logger.warning(f"Task {task_id}, Region {region_index}: Polygon became empty or invalid after shrinking by {shrink_distance:.2f} pixels. Skipping region.")
                                continue

                             # Handle potential MultiPolygons resulting from buffering
                            polygons_to_process = []
                            if isinstance(shrunk_polygon, MultiPolygon):
                                # Option: Process each part of the MultiPolygon as a separate crop
                                # polygons_to_process.extend(list(shrunk_polygon.geoms))
                                # Option: Take only the largest polygon
                                largest_poly = max(shrunk_polygon.geoms, key=lambda p: p.area)
                                if largest_poly.area > 0:
                                    polygons_to_process.append(largest_poly)
                                else:
                                     logger.warning(f"Task {task_id}, Region {region_index}: Largest component of MultiPolygon has zero area after shrinking. Skipping.")
                                     continue
                            elif isinstance(shrunk_polygon, Polygon) and shrunk_polygon.area > 0:
                                polygons_to_process.append(shrunk_polygon)
                            else:
                                logger.warning(f"Task {task_id}, Region {region_index}: Invalid geometry type or zero area after shrinking. Type: {type(shrunk_polygon)}. Skipping.")
                                continue

                            # --- Feature 3: Create Crop for each valid (shrunk) polygon ---
                            for poly_idx, final_polygon in enumerate(polygons_to_process):
                                # Get polygon points from the SHRUNK polygon
                                shrunk_points = list(final_polygon.exterior.coords)
                                
                                # Get bounding box of the final (shrunk) polygon
                                min_x, min_y, max_x, max_y = final_polygon.bounds
                                crop_min_x = max(0, int(math.floor(min_x)))
                                crop_min_y = max(0, int(math.floor(min_y)))
                                crop_max_x = min(original_width, int(math.ceil(max_x)))
                                crop_max_y = min(original_height, int(math.ceil(max_y)))

                                # Create mask using SHRUNK polygon coordinates
                                mask = Image.new('L', (original_width, original_height), 0)
                                draw = ImageDraw.Draw(mask)
                                
                                # Convert shrunk polygon coordinates to list of tuples
                                shrunk_poly_points = [(x, y) for x, y in shrunk_points]
                                draw.polygon(shrunk_poly_points, fill=255)  # Use shrunk coordinates

                                # Apply mask and crop
                                img_array = np.array(img)
                                mask_array = np.array(mask)
                                masked_array = np.where(mask_array[..., None] == 255, img_array, 0)
                                cropped_img = Image.fromarray(masked_array).crop(
                                    (crop_min_x, crop_min_y, crop_max_x, crop_max_y)
                                )

                                # Save cropped image
                                crop_filename = f'task_{task_id}_region_{region_index}_poly_{poly_idx}_crop.jpg'
                                cropped_img_path = os.path.join(temp_dir, crop_filename)
                                cropped_img.save(cropped_img_path)

                                
                                # Get actual crop dimensions from the image
                                crop_width, crop_height = cropped_img.size

                                                                # Add COCO image entry for the crop
                                coco['images'].append({
                                    'id': coco_img_id,
                                    'file_name': cropped_img_path, # Absolute path to the crop
                                    'width': crop_width,
                                    'height': crop_height
                                })
                                # Process annotations
                                for ann_data in task_annotations:
                                    # Check if center is in the SHRUNK polygon
                                    center_point = Point(ann_data['center_abs'])
                                    if final_polygon.contains(center_point):
                                        # Convert coordinates RELATIVE TO CROP
                                        orig_bbox = ann_data['bbox_abs']
                                        adj_x_min = orig_bbox[0] - crop_min_x
                                        adj_y_min = orig_bbox[1] - crop_min_y
                                        
                                        # Ensure coordinates are within crop bounds
                                        final_x_min = max(0, adj_x_min)
                                        final_y_min = max(0, adj_y_min)
                                        final_w = min(crop_width - final_x_min, orig_bbox[2])
                                        final_h = min(crop_height - final_y_min, orig_bbox[3])

                                        # Only add if annotation is visible in crop
                                        if final_w > 0 and final_h > 0:
                                            coco['annotations'].append({
                                                'id': ann_id,
                                                'image_id': coco_img_id,
                                                'category_id': ann_data['category_id'],
                                                'bbox': [final_x_min, final_y_min, final_w, final_h],
                                                'area': final_w * final_h,
                                                'iscrowd': 0
                                            })
                                            ann_id += 1

                                # Increment COCO image ID for the next crop/image
                                coco_img_id += 1
                                processed_region = True # Mark that at least one region was processed

                        except GEOSException as geos_e:
                            logger.error(f"Task {task_id}, Region {region_index}: Shapely geometric error: {geos_e}. Skipping region.")
                        except Exception as e:
                            logger.error(f"Task {task_id}, Region {region_index}: Unexpected error processing region: {e}", exc_info=True)
                            # Continue to next region

                    # If regions were supposed to be processed but none were successful, log it
                    if not processed_region and not train_on_whole_image:
                        logger.warning(f"Task {task_id}: Set to train on regions, but no regions were successfully processed and cropped.")


            except FileNotFoundError:
                 logger.warning(f"Image file not found for task {task_id} at path derived from {task['data'].get('image', 'N/A')}. Skipping task.")
            except IOError as e:
                 logger.error(f"Error opening or reading image for task {task_id}: {e}. Skipping task.")
            except Exception as e:
                # Catch broader errors during task processing
                logger.error(f"Error processing task {task_id}: {e}", exc_info=True)

        if not coco['images']:
            logger.warning("COCO generation resulted in zero images. Training cannot proceed.")
        if not coco['annotations']:
            logger.warning("COCO generation resulted in zero annotations. Training might be ineffective.")


        return coco


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
