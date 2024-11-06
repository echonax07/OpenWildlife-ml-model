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
                # inputs dim (B, 3, H, W)
                _, _, height, width = inputs.shape
                sliced_image_object = slice_image(
                    inputs[0].cpu().numpy().transpose(1, 2, 0),
                    slice_height=self.sliding_window_inference['patch_size'],
                    slice_width=self.sliding_window_inference['patch_size'],
                    auto_slice_resolution=False,
                    overlap_height_ratio=self.sliding_window_inference['patch_overlap_ratio'],
                    overlap_width_ratio=self.sliding_window_inference['patch_overlap_ratio'],
                )
                stacked_image_slice = np.stack(
                    sliced_image_object.images, axis=0)
                stacked_image_slice = stacked_image_slice.transpose(0, 3, 1, 2)
                stacked_image_slice = torch.tensor(
                    stacked_image_slice, dtype=inputs.dtype, device=inputs.device)

                clone_data_sample = data_samples[0].clone()
                patch_size = self.sliding_window_inference['patch_size']
                clone_data_sample.set_metainfo(dict(batch_input_shape=(patch_size, patch_size), pad_shape=(
                    patch_size, patch_size), ori_shape=(patch_size, patch_size), img_shape=(patch_size, patch_size), img_id=None, img_path=None))
                gt_instances = clone_data_sample.ignored_instances.clone()
                clone_data_sample.set_data(dict(gt_instances=gt_instances))
                data_samples_sliced = [
                    clone_data_sample.clone() for _ in range(len(sliced_image_object))]

                # Determine the batch size for slices
                slice_batch_size = self.sliding_window_inference.get(
                    'slice_batch_size', -1)
                num_slices = stacked_image_slice.shape[0]

                if slice_batch_size == -1:
                    slice_batch_size = num_slices

                slice_results = []
                # print(f'Doing sliding window inference with {num_slices} slices')
                for i in range(0, num_slices, slice_batch_size):
                    batch_slices = stacked_image_slice[i:i+slice_batch_size]
                    batch_data_samples = data_samples_sliced[i:i +
                                                             slice_batch_size]
                    batch_results = self.predict(
                        batch_slices, batch_data_samples)
                    slice_results.extend(batch_results)

                shifted_instances = shift_predictions(
                    slice_results,
                    sliced_image_object.starting_pixels,
                    src_image_shape=(height, width)
                )

                found_object = any(len(result.pred_instances)
                                   for result in slice_results)

                if found_object:
                    image_result = merge_results_by_nms(
                        slice_results,
                        sliced_image_object.starting_pixels,
                        src_image_shape=(height, width),
                        nms_cfg={
                            'type': self.sliding_window_inference['merge_nms_type'],
                            'iou_threshold': self.sliding_window_inference['merge_iou_thr']
                        })
                    instance_data = InstanceData()
                    instance_data.labels = image_result.pred_instances.labels
                    instance_data.bboxes = image_result.pred_instances.bboxes
                    instance_data.scores = image_result.pred_instances.scores
                else:
                    instance_data = InstanceData()
                    instance_data.labels = slice_results[0].pred_instances.labels
                    instance_data.bboxes = slice_results[0].pred_instances.bboxes
                    instance_data.scores = slice_results[0].pred_instances.scores

                final_data_sample_sliced = slice_results[0].clone()
                # Put back original shape and size
                final_data_sample_sliced.set_metainfo(dict(batch_input_shape=(height, width), pad_shape=(
                    height, width), ori_shape=(height, width), img_shape=(height, width)))
                final_data_sample_sliced.set_data(
                    dict(pred_instances=instance_data))
                data_samples[0].set_data(dict(pred_instances=instance_data))
                return data_samples
            else:
                return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

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
