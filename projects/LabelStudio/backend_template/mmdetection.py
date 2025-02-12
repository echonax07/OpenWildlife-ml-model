# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir

from mmdet.apis import inference_detector, init_detector
from tools.slice_train_imgs import slice_train_images
import torch
from icecream import ic


logger = logging.getLogger(__name__)


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 score_threshold=0.5,
                 device='cpu',
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)
        from icecream import ic
        ic(config_file)
        ic(os.environ['config_file'])
        ic(checkpoint_file)
        ic(os.environ['checkpoint_file'])
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
            self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values',
                                                       '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': key
                    })
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}'  # noqa E501
                )
        return image_url

    def predict(self, tasks, **kwargs):
        label_studio_results = []
        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            model_results = inference_detector(self.model,
                                            image_path).pred_instances
            results = []
            all_scores = []
            img_width, img_height = get_image_size(image_path)
            print(f'>>> model_results: {model_results}')
            print(f'>>> label_map {self.label_map}')
            print(f'>>> self.model.dataset_meta: {self.model.dataset_meta}')
            classes = self.model.dataset_meta.get('classes')
            print(f'Classes >>> {classes}')
            for item in model_results:
                print(f'item >>>>> {item}')
                bboxes, label, scores = item['bboxes'], item['labels'], item[
                    'scores']
                score = float(scores[-1])
                if score < self.score_thresh:
                    continue
                print(f'bboxes >>>>> {bboxes}')
                print(f'label >>>>> {label}')
                output_label = classes[list(self.label_map.get(label, label))[0]]
                print(f'>>> output_label: {output_label}')
                if output_label not in self.labels_in_config:
                    print(output_label + ' label not found in project config.')
                    continue

                for bbox in bboxes:
                    bbox = list(bbox)
                    if not bbox:
                        continue

                    x, y, xmax, ymax = bbox[:4]
                    results.append({
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': [output_label],
                            'x': float(x) / img_width * 100,
                            'y': float(y) / img_height * 100,
                            'width': (float(xmax) - float(x)) / img_width * 100,
                            'height': (float(ymax) - float(y)) / img_height * 100
                        },
                        'score': score
                    })
                    all_scores.append(score)
            avg_score = sum(all_scores) / max(len(all_scores), 1)
            print(f'>>> RESULTS: {results}')
            label_studio_results.append({'result': results, 'score': avg_score})
        return label_studio_results
    
    def fit(self, tasks, workdir=None, **kwargs):
        """Train the model using annotations from event.json"""
        import tempfile
        from mmengine.config import Config
        from mmdet.registry import RUNNERS
        from mmengine.runner import Runner

            
        # Set single-threaded environment
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
            
        result = {
            'model_path': self.checkpoint_file,
            'checkpoints': [],
            'labels': list(self.labels_in_config),
            'error': None
        }

        try:
            # Read event.json from workdir
            event_path = os.path.join(workdir, 'event.json')
            if not os.path.exists(event_path):
                raise FileNotFoundError(f"event.json not found in {workdir}")

            with open(event_path, 'r') as f:
                event_data = json.load(f)

            # Extract relevant data from event.json structure
            annotation_data = event_data['data']['annotation']
            task_data = {
                'data': event_data['data']['task']['data'],
                'annotations': [{
                    'result': annotation_data['result'],
                    'task': event_data['data']['task']['id']
                }]
            }

            # Convert to COCO format
            coco_data = self._event_to_coco([task_data])

            # Create temporary directory for training data
            with tempfile.TemporaryDirectory() as temp_dir:
                ann_path = os.path.join(temp_dir, 'train.json')
                with open(ann_path, 'w') as f:
                    json.dump(coco_data, f, indent=4)

                # Load and modify config
                cfg = Config.fromfile(self.config_file)
                cfg.model.roi_head.bbox_head.num_classes = len(self.labels_in_config)
                
                # Modify train paths
                cfg.data_root = ''
                cfg.train_dataloader.dataset.data_root = ''
                cfg.train_dataloader.dataset.ann_file = os.path.join(temp_dir,'train.json')
                cfg.train_dataloader.num_workers = 0  # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.train_dataloader.persistent_workers = False
                cfg.train_dataloader.prefetch_factor = None
                cfg.train_dataloader.batch_size = 2
                
                # Modify val paths
                cfg.val_dataloader.dataset.data_root = ''
                cfg.val_dataloader.dataset.ann_file = os.path.join(temp_dir,'train.json')
                cfg.val_dataloader.num_workers = 0  # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.val_dataloader.persistent_workers = False
                cfg.val_dataloader.prefetch_factor = None
                cfg.val_evaluator.ann_file = os.path.join(temp_dir, 'train.json')
                cfg.val_evaluator.outfile_prefix = os.path.join(temp_dir, 'prediction_val')
                
                
                # Modify test paths
                cfg.test_dataloader.dataset.data_root = ''
                cfg.test_dataloader.dataset.ann_file = os.path.join(temp_dir,'train.json')
                cfg.test_dataloader.num_workers = 0  # Disable multiprocessing to avoid interference with Label Studio which doesnt allow any child to spawn new processes
                cfg.test_dataloader.persistent_workers = False
                cfg.test_dataloader.prefetch_factor = None
                cfg.test_evaluator.ann_file = os.path.join(temp_dir, 'train.json')
                cfg.test_evaluator.outfile_prefix = os.path.join(temp_dir, 'prediction_test')
                
                # Modify train slices path 
                os.makedirs(os.path.join(temp_dir, 'slices'), exist_ok=True)
                cfg.data_root_slice = os.path.join(temp_dir, 'slices')
                # whole images data_root
                cfg.data_root_whole = temp_dir
                
                # Training parameters
                cfg.train_cfg.max_iters = 10
                cfg.train_cfg.val_interval=1000000 # Disable validation
                cfg.default_hooks.checkpoint.interval = 10 # Save checkpoint every 10 iterations
                # use cuda if available else cpu
                cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                cfg.work_dir = workdir or temp_dir
                cfg.load_from = os.environ['checkpoint_file']
                
                slice_configuration = cfg.get('slice_configuration')
                cfg = slice_train_images(cfg, **slice_configuration)
                            
                # Run training
                runner = Runner.from_cfg(cfg)
                runner.train()

                # read checkpoint path from the cfg.work_dir/'last_checkpoint'
                with open(os.path.join(cfg.work_dir, 'last_checkpoint'), 'r') as f:
                    latest_ckpt = f.read().strip()
                # Update model checkpoint
                # latest_ckpt = os.path.join(cfg.work_dir, 'latest.pth')
                if os.path.exists(latest_ckpt):
                    result['model_path'] = latest_ckpt
                    result['checkpoints'].append(latest_ckpt)
                    self.checkpoint_file = latest_ckpt
                    self.model = init_detector(
                        self.config_file, 
                        self.checkpoint_file, 
                        device=cfg.device
                    )
                    os.environ['checkpoint_file'] = latest_ckpt
                    ic("Training completed successfully")     
                else:
                    raise RuntimeError("Training failed - no checkpoint created")

            return result

        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            result['error'] = str(e)
            return result

    def _event_to_coco(self, tasks):
        """Convert event.json structure to COCO format"""
        coco = {
            'images': [],
            'annotations': [],
            'categories': []
        }

        # Create categories
        categories = {label: idx+1 for idx, label in enumerate(sorted(self.labels_in_config))}
        coco['categories'] = [{'id': v, 'name': k} for k, v in categories.items()]

        ann_id = 1
        for task_id, task in enumerate(tasks):
            try:
                # Get image info
                image_url = task['data']['image']
                image_path = self.get_local_path(image_url)
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                # read height and width of the image from image_path using pillow
                height, width = get_image_size(image_path)

                # Add image entry
                coco['images'].append({
                    'id': task_id + 1,
                    'file_name': image_path,
                    'width': width,
                    'height': height
                })

                # Process annotations
                for annotation in task.get('annotations', []):
                    for result in annotation.get('result', []):
                        if result.get('type') != 'rectanglelabels':
                            continue

                        label = result['value']['rectanglelabels'][0]
                        if label not in categories:
                            continue

                        # Convert to COCO bbox format [x, y, width, height]
                        x = result['value']['x'] * width / 100
                        y = result['value']['y'] * height / 100
                        w = result['value']['width'] * width / 100
                        h = result['value']['height'] * height / 100

                        coco['annotations'].append({
                            'id': ann_id,
                            'image_id': task_id + 1,
                            'category_id': categories[label],
                            'bbox': [x, y, w, h],
                            'area': w * h,
                            'iscrowd': 0,
                            'ignore': 0
                        })
                        ann_id += 1

            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                continue

        return coco
def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
