import os
import matplotlib.pyplot as plt
import json
import logging
from faster_coco_eval import COCO
from faster_coco_eval.extra import Curves
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from icecream import ic
import json
from pycocotools.coco import COCO as COCO2
import numpy as np
from tqdm import tqdm

def calculate_centers(boxes):
    """Calculate the centers ofp multiple bounding boxes."""
    x_centers = boxes[:, 0] + boxes[:, 2] / 2
    y_centers = boxes[:, 1] + boxes[:, 3] / 2
    return np.vstack((x_centers, y_centers)).T


def compute_distances(pred_centers, gt_centers):
    """Compute the Euclidean distances between prediction and ground truth centers."""
    pred_expand = np.expand_dims(pred_centers, axis=1)
    gt_expand = np.expand_dims(gt_centers, axis=0)
    distances = np.linalg.norm(pred_expand - gt_expand, axis=2)
    return distances


def custom_metric(predictions, ground_truths, distance_threshold):
    """
    Compute custom metric for object detection using vectorized operations.

    Parameters:
    - predictions: List of prediction boxes [x, y, h, w]
    - ground_truths: List of ground truth boxes [x, y, h, w]
    - distance_threshold: Threshold for the center distance to consider a match

    Returns:
    - true_positives: Number of true positive detections
    - false_positives: Number of false positive detections
    - false_negatives: Number of false negative detections
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return 0, len(predictions), len(ground_truths)

    pred_boxes = np.array(predictions)
    gt_boxes = np.array(ground_truths)

    pred_centers = calculate_centers(pred_boxes)
    gt_centers = calculate_centers(gt_boxes)

    distances = compute_distances(pred_centers, gt_centers)

    # Match predictions to ground truths based on distance threshold
    matches = distances <= distance_threshold

    true_positives = 0
    matched_gt_indices = set()

    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if matches[i, j] and j not in matched_gt_indices:
                true_positives += 1
                matched_gt_indices.add(j)
                break

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - len(matched_gt_indices)

    return true_positives, false_positives, false_negatives


def calculate_metrics(gt_json_path, pred_json_path, scores, image_ids=None, distance_threshold=10):
    """
    Calculate TP, FP, FN for specified image IDs in the JSON files, with score threshold.

    Parameters:
    - gt_json_path: Path to the ground truth JSON file
    - pred_json_path: Path to the predictions JSON file
    - score_threshold: Minimum score threshold for predictions to consider
    - image_ids: List of image IDs to process. If None, process all images.
    - distance_threshold: Threshold for the center distance to consider a match

    Returns:
    - true_positives: Number of true positive detections across specified images
    - false_positives: Number of false positive detections across specified images
    - false_negatives: Number of false negative detections across specified images
    """
    # Load COCO instances
    coco_gt = COCO2(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)

    if image_ids is None:
        image_ids = coco_gt.getImgIds()

    tp_list = [0]*101
    fp_list = [0]*101
    fn_list = [0]*101
    images_with_predictions_only_list = [0]*101
    fp_in_images_with_predictions_only_list = [0]*101
    
    for img_id in tqdm(image_ids):
        # Get ground truth annotations for current image
        ann_ids_gt = coco_gt.getAnnIds(imgIds=img_id)
        anns_gt = coco_gt.loadAnns(ann_ids_gt)
        gt_boxes = [ann['bbox'] for ann in anns_gt]

        # Get prediction annotations for current image, filtered by score threshold
        ann_ids_pred = coco_pred.getAnnIds(imgIds=img_id)
        anns_pred = coco_pred.loadAnns(ann_ids_pred)
        for idx,scr in enumerate(scores):            
            pred_boxes = [ann['bbox']
                        for ann in anns_pred if ann['score'] >= scr]
            if len(gt_boxes) == 0 and len(pred_boxes) > 0:
                images_with_predictions_only_list[idx] += 1
                fp_in_images_with_predictions_only_list[idx] += len(pred_boxes)
            tp, fp, fn = custom_metric(pred_boxes, gt_boxes, distance_threshold)
            tp_list[idx]+= tp
            fp_list[idx] += fp
            fn_list[idx] += fn 

    return tp_list, fp_list, fn_list, images_with_predictions_only_list, fp_in_images_with_predictions_only_list


def load(file):
    with open(file) as io:
        _data = json.load(io)

    return _data


# def plot_confusion_matrix(gt_path, prediction_path):
#     threshold_iou = 0.1
#     score_threshold = 0
#     prepared_coco_in_dict = load(gt_path)
#     prepared_anns = load(prediction_path)
#     print(f'Total predictions: {len(prepared_anns)}')
#     thresholded_score_list = [
#         1 for ann in prepared_anns if ann['score'] >= score_threshold]

#     print(
#         f'Total prediction after thresholding: {len(thresholded_score_list)}')
#     cocoGt = COCO(prepared_coco_in_dict)
#     cocoDt = cocoGt.loadRes(prepared_anns)
#     cur = Curves(cocoGt, cocoDt, iou_tresh=threshold_iou, iouType='bbox')
#     fig, precision, recall, scores, names = cur.plot_pre_rec(
#         plotly_backend=True)
#     # cur.display_matrix(score_threshold=score_threshold)
#     return cur, precision, recall, scores, names


def load(file):
    with open(file) as io:
        _data = json.load(io)
    return _data


def plot_confusion_matrix(gt_path, prediction_path, threshold_iou=0.5):
    threshold_iou = threshold_iou
    score_threshold = 0
    bbox_difference_threshold = 10000
    prepared_coco_in_dict = load(gt_path)
    prepared_anns = load(prediction_path)
    print(f'Total predictions: {len(prepared_anns)}')
    thresholded_score_list = [
        ann for ann in prepared_anns if ann['score'] >= score_threshold]
    prepared_anns = [
        ann for ann in thresholded_score_list if abs(ann['bbox'][2]-ann['bbox'][3]) <= bbox_difference_threshold]

    print(
        f'Total prediction after thresholding: {len(prepared_anns)}')
    if len(prepared_anns) == 0:
        print('Annotations are empty after threholding')
        return len(prepared_anns), 0, 0, 0, 0, 0
    else:
        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)
        cur = Curves(cocoGt, cocoDt, iou_tresh=threshold_iou, iouType='bbox', useCats=False)
        fig, recall, precision, scores, names = cur.plot_pre_rec(
            plotly_backend=True)
    # cur.display_matrix(score_threshold=score_threshold)
        return len(prepared_anns), cur, recall, precision, scores, names

threshold_iou = 0.5
bbox_distance_threshold = 20
GT_json = [
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_terns_(Already_processed_Hayes)/test.json'
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/AED/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/test.json',
    # # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_michigan/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_monash/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_newmexico/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_palmyra/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_penguins/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_pfeifer/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_seabirdwatch/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_qian_penguin/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/WAID/test/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eikelboom/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_sea_lion_blackout/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/turtle/test.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/NOAA_arctic_seals/test.json',
    # '/home/m32patel/projects/def-dclausi/whale/merged/test/test_140.json',
    # '/home/m32patel/projects/def-dclausi/whale/merged/test/test_2015.json',
    # '/home/m32patel/projects/def-dclausi/whale/merged/test/test_ES_2016.json',
    # '/home/m32patel/projects/def-dclausi/whale/merged/test/test_2017.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/coco_split_val.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/test_15.json',
    '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB_grounded.json',
    # '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/coco_iter12345_val.json'
    # '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/mscoco_completed_test_iter1.json',
    #  '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/mscoco_completed_test_iter2.json',
    #   '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/mscoco_completed_test_iter3.json',
    #    '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/mscoco_completed_test_iter4.json',
    #     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/mscoco_completed_test_iter5.json',
    #     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/mscoco_completed_test_iter6.json',
    # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/test.json',
]

# # List of prediction JSON file paths
pred_json = [
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Aerial_seabird_westafrica_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/birds_izembek_lagoon_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/michigan_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/monash_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/new_mexico_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/palmyra_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/pfeifer_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/seabirdwatch_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/qian_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/WAID_livestock_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_sealion_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/turtle_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_artic_seal_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2014_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",  # whale data based on year
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2015_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
#     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Narwhal_2016_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2017_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/SAVMAP/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/polar_bear/prediction_mm_grounding_dino_nocaption.bbox.json",
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Virunga_garamba_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/DFO_Whale23/prediction_mm_grounding_dino_nocaption.bbox.json'
# '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/penguins_od_finetune/prediction_mm_grounding_dino_finetune_test.bbox.json'
# '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/SAVMAP/prediction_mm_grounding_dino_finetune_val.bbox.json'
    # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Aerial_seabird_westafrica_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
    # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/tern_finetune/prediction_mm_grounding_dino_finetune_val.bbox.json'
    # '/lustre06/project/6075102/whale/mmwhale2/work_dir_grounding_dino/palmyra_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json'
    '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/virunga_garamba/prediction_mm_grounding_dino_finetune_test.bbox.json'
]


# # List of prediction JSON file paths
# pred_json = [
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/test_prediction_iter1.bbox.json',
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/test_prediction_iter2.bbox.json',
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/test_prediction_iter3.bbox.json',
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/test_prediction_iter4.bbox.json',
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/test_prediction_iter5.bbox.json',
#     '/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/Elements/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/mscoco_jsons/test_prediction_iter6.bbox.json',
#     # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune/eiderduck_finetune_syn_plus_real/prediction_mm_grounding_dino_finetune_test.bbox.json'
#     # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Aerial_seabird_westafrica_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/birds_izembek_lagoon_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/michigan_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/monash_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/new_mexico_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/palmyra_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/pfeifer_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/seabirdwatch_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/qian_od_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/WAID_livestock_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_sealion_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/turtle_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_artic_seal_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2014_dataset/prediction_mm_grounding_dino_caption.bbox.json",  # whale data based on year
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2015_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Narwhal_2016_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2017_dataset/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/SAVMAP/prediction_mm_grounding_dino_caption.bbox.json",
# #     "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/polar_bear/prediction_mm_grounding_dino_caption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Virunga_garamba_dataset/prediction_mm_grounding_dino_caption.bbox.json"
        # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/palmyra_od_dataset/prediction_mm_grounding_dino_viz_caption.bbox.json',
# ]

# Extract the names of the runs
names = [os.path.basename(os.path.dirname(path)) for path in pred_json]
# names = ['iter1',
#          'iter2',
#          'iter3',
#          'iter4',
#          'iter5',
#          'iter6',]


fig = make_subplots(rows=1, cols=1, subplot_titles=[
    'Precision-Recall'])

experiments = []

for GT, pred, name in zip(GT_json, pred_json, names):
    try:
        ic(name)
        len_anns, cur, r, p, score, auc = plot_confusion_matrix(GT, pred, threshold_iou)
        # Find the index of the element closest to 0.9
        tp_list = []
        fp_list = []
        fn_list = []
        images_with_predictions_only_list = []
        fp_in_images_with_predictions_only_list = []
        # closest_index = np.argmin(np.abs(r - 0.9))
        # score_at_p_90 = score[closest_index]
        tp_list, fp_list, fn_list, images_with_predictions_only_list, fp_in_images_with_predictions_only_list = calculate_metrics(
            GT, pred, score, distance_threshold=bbox_distance_threshold)
        # print(
        #     f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
        # Calculate element-wise F1 score
        # Add small value to avoid division by zero
        f1 = 2 * (p * r) / (p + r + 1e-10)
        tp = np.array(tp_list)
        fp = np.array(fp_list)
        fn = np.array(fn_list)
        p2 = tp/(tp+fp)
        r2 = tp/(tp+fn)
        f12 = 2*p2*r2/(p2+r2)

        experiment_data = {
            "name": name,
            "p": list(p),
            "r": list(r),
            "r": list(r),
            "f1": list(f1),
            "score": list(score),
            "auc": auc,
            "len_anns": len_anns,
            "tp": tp_list,
            "fp": fp_list,
            "fn": fn_list,
            "p2": list(p2),
            "r2": list(r2),
            "f12": list(f12),
            "no_GT_only_pred": images_with_predictions_only_list,
            "fp_in_images_with_predictions_only": fp_in_images_with_predictions_only_list
        }
        experiments.append(experiment_data)
        if len_anns == 0:
            pass
        else:
            # f1, tp, fp, fn, p2, r2, no_GT_only_pred, fp_in_images_with_predictions_only
            customdata_stack = np.stack((f1, tp, fp, fn, p2, r2, f12, np.array(
                images_with_predictions_only_list), np.array(fp_in_images_with_predictions_only_list)), axis=1)
            fig.add_trace(
                go.Scatter(
                    x=r,
                    y=p,
                    name=f'{name}:{auc}',
                    text=score,  # Keep original score
                    customdata=customdata_stack,  # Add f1 as custom data
                    hovertemplate='Pre: %{y:.3f}<br>' +
                    'Rec: %{x:.3f}<br>' +
                    'F1: %{customdata[0]:.3f}<br>' +  # Use customdata for F1
                    'Score: %{text:.3f}<br>' +
                    'TP: %{customdata[1]:.3f}<br>' +
                    'FP: %{customdata[2]:.3f}<br>' +
                    'FN: %{customdata[3]:.3f}<br>' +
                    'p2: %{customdata[4]:.3f}<br>' +
                    'r2: %{customdata[5]:.3f}<br>' +
                    'f12: %{customdata[6]:.3f}<br>' +
                    'no_GT_only_pred: %{customdata[7]:.3f}<br>' +
                    'fp_in_images_with_predictions_only: %{customdata[8]:.3f}<br>' +
                    f'Name: {name}<extra></extra>',
                    showlegend=True,
                    mode='lines',
                ),
                row=1, col=1
            )

            # Draw a horizontal line at recall = 90%
            fig.add_vline(x=0.9, line_dash="dash", line_color="red",
                          annotation_text="Recall = 90%",
                          annotation_position="bottom right")
            margin = 0.01
            fig.layout.yaxis.range = [0 - margin, 1 + margin]
            fig.layout.xaxis.range = [0 - margin, 1 + margin]

            fig.layout.yaxis.title = 'Precision'
            fig.layout.xaxis.title = 'Recall'

            fig.update_layout(height=600, width=1200)

    except Exception as e:
        print(f'Exception: {e}')
        print(f'file {pred} maynot be found pls check the path')
        continue

experiment_json = {
    "experiments": experiments
}

with open("/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck.json", "w") as outfile:
    json.dump(experiment_json, outfile)
fig.write_image('pr_curve_plotly.png')
fig.show()
fig.write_html(
    "/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz/eider_duck.html")
print("file dumped sucessfully")
