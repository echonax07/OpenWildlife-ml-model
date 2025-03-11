# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from mmengine.model import BaseModel
from torch import Tensor
import numpy as np
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from ..utils import samplelist_boxtype2tensor
from mmengine.structures import InstanceData
from mmdet.utils.slicing import slice_image
from mmdet.utils.large_image import merge_results_by_nms, shift_predictions

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


class BaseDetector(BaseModel, metaclass=ABCMeta):
    """Base class for detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 sliding_window_inference: Dict = None,
                 init_cfg: OptMultiConfig = None):
        self.sliding_window_inference = sliding_window_inference
        if self.sliding_window_inference == None:
            self.sliding_window_inference = {}
            self.sliding_window_inference['enable'] = False
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            if self.sliding_window_inference.get('enable'):
                from icecream import ic
                print('*'*10)
                ic('sliding window inference')
                print('*'*10)
                _, _, height, width = inputs.shape
                img_np = inputs[0].permute(1, 2, 0).cpu().numpy()
                
                # Create sliced image object
                slice_params = self.sliding_window_inference
                sliced_image_object = slice_image(
                    img_np,
                    slice_height=slice_params['patch_size'],
                    slice_width=slice_params['patch_size'],
                    auto_slice_resolution=False,
                    overlap_height_ratio=slice_params['patch_overlap_ratio'],
                    overlap_width_ratio=slice_params['patch_overlap_ratio'],
                )

                # Create template data sample once
                template_ds = self._create_template_data_sample(
                    data_samples[0], 
                    slice_params['patch_size']
                ) if data_samples else None

                # Prepare slices on CPU
                max_height = slice_params['patch_size']
                max_width = slice_params['patch_size']
                padded_slices = []
                for img in sliced_image_object.images:
                    pad_height = max_height - img.shape[0]
                    pad_width = max_width - img.shape[1]
                    padded_img = np.pad(
                        img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant'
                    )
                    padded_slices.append(padded_img)

                slices_np = np.stack(padded_slices).transpose(0, 3, 1, 2)
                slices_cpu = torch.from_numpy(slices_np).to(dtype=inputs.dtype)
                

                slice_results = []
                batch_size = slice_params.get('slice_batch_size', 1)
                
                # Process in batches
                
                # wrap tqdm around the for
                for i in range(0, len(slices_cpu), batch_size):
                    batch_end = i + batch_size
                    current_batch = slices_cpu[i:batch_end]
                    
                    # Move only this batch to GPU
                    current_batch_gpu = current_batch.to(device=inputs.device)
                    
                    # Prepare batch data samples
                    batch_ds = [template_ds.clone() for _ in range(len(current_batch))] if template_ds else None
                    
                    # Process batch
                    batch_results = self.predict(current_batch_gpu, batch_ds)
                    
                    # Move results to CPU and store
                    for ds in batch_results:
                        if hasattr(ds, 'pred_instances'):
                            ds.pred_instances = ds.pred_instances.to('cpu')
                    slice_results.extend(batch_results)
                    
                    # Cleanup GPU resources
                    del current_batch_gpu
                    torch.cuda.empty_cache()

                # Merge results
                merged_instances = self._merge_slice_results(
                    slice_results, 
                    sliced_image_object.starting_pixels,
                    (height, width)
                )
                
                # Create final data sample
                final_result = data_samples[0].clone() if data_samples else DetDataSample()
                final_result.pred_instances = merged_instances
                return [final_result]
            else:
                return self.predict(inputs, data_samples)

        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _create_template_data_sample(self, original_ds, patch_size):
        """Create a template data sample for slices."""
        template_ds = original_ds.clone()
        meta = {
            'batch_input_shape': (patch_size, patch_size),
            'pad_shape': (patch_size, patch_size),
            'ori_shape': (patch_size, patch_size),
            'img_shape': (patch_size, patch_size),
            'img_id': None,
            'img_path': None
        }
        template_ds.set_metainfo(meta)
        return template_ds

    def _merge_slice_results(self, slice_results, starting_pixels, img_shape):
        """Merge slice results with NMS."""
        if not any(len(r.pred_instances) for r in slice_results):
            return InstanceData()

        shifted_instances = shift_predictions(
            slice_results,
            starting_pixels,
            src_image_shape=img_shape
        )

        return merge_results_by_nms(
            slice_results,
            starting_pixels,
            src_image_shape=img_shape,
            nms_cfg={
                'type': self.sliding_window_inference['merge_nms_type'],
                'iou_threshold': self.sliding_window_inference['merge_iou_thr']
            }
        ).pred_instances

    @abstractmethod
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples
