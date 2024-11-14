import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from faster_coco_eval import COCO
from faster_coco_eval.extra import Curves
from pycocotools.coco import COCO as COCO2
from functools import lru_cache

@dataclass
class MetricsResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    images_with_predictions_only: int
    fp_in_images_with_predictions_only: int

@dataclass
class ExperimentResult:
    name: str
    precision: List[float]
    recall: List[float]
    f1: List[float]
    scores: List[float]
    auc: float
    total_annotations: int
    metrics: MetricsResult

class BoundingBoxAnalyzer:
    def __init__(self, distance_threshold: float = 1000):
        self.distance_threshold = distance_threshold

    @staticmethod
    def calculate_centers(boxes: np.ndarray) -> np.ndarray:
        """Calculate centers of bounding boxes using vectorized operations."""
        return np.column_stack((
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2
        ))

    @staticmethod
    def compute_distances(pred_centers: np.ndarray, gt_centers: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between prediction and ground truth centers."""
        return np.linalg.norm(
            pred_centers[:, np.newaxis, :] - gt_centers[np.newaxis, :, :],
            axis=2
        )

    def custom_metric(self, predictions: List, ground_truths: List) -> MetricsResult:
        """Compute custom metrics for object detection."""
        if not predictions or not ground_truths:
            return MetricsResult(0, len(predictions), len(ground_truths), 0, 0)

        pred_boxes = np.array(predictions)
        gt_boxes = np.array(ground_truths)

        distances = self.compute_distances(
            self.calculate_centers(pred_boxes),
            self.calculate_centers(gt_boxes)
        )

        matches = distances <= self.distance_threshold
        matched_gt_indices = set()
        true_positives = 0

        for pred_matches in matches:
            for gt_idx, is_match in enumerate(pred_matches):
                if is_match and gt_idx not in matched_gt_indices:
                    true_positives += 1
                    matched_gt_indices.add(gt_idx)
                    break

        return MetricsResult(
            true_positives=true_positives,
            false_positives=len(predictions) - true_positives,
            false_negatives=len(ground_truths) - len(matched_gt_indices),
            images_with_predictions_only=0,  # Updated in calculate_metrics
            fp_in_images_with_predictions_only=0  # Updated in calculate_metrics
        )

class COCOAnalyzer:
    def __init__(self, gt_path: str, pred_path: str):
        self.gt_path = Path(gt_path)
        self.pred_path = Path(pred_path)
        self.bbox_analyzer = BoundingBoxAnalyzer()
        
    @lru_cache(maxsize=2)
    def load_annotations(self) -> Tuple[COCO2, COCO2]:
        """Load COCO annotations with caching."""
        coco_gt = COCO2(str(self.gt_path))
        coco_pred = coco_gt.loadRes(str(self.pred_path))
        return coco_gt, coco_pred

    def calculate_metrics(self, score_threshold: float = 0.5) -> MetricsResult:
        """Calculate metrics for the dataset."""
        coco_gt, coco_pred = self.load_annotations()
        image_ids = coco_gt.getImgIds()
        
        total_metrics = MetricsResult(0, 0, 0, 0, 0)
        
        for img_id in image_ids:
            gt_boxes = [ann['bbox'] for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))]
            pred_anns = [ann for ann in coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))
                        if ann['score'] >= score_threshold]
            pred_boxes = [ann['bbox'] for ann in pred_anns]

            if not gt_boxes and pred_boxes:
                total_metrics.images_with_predictions_only += 1
                total_metrics.fp_in_images_with_predictions_only += len(pred_boxes)

            metrics = self.bbox_analyzer.custom_metric(pred_boxes, gt_boxes)
            
            # Update totals
            total_metrics.true_positives += metrics.true_positives
            total_metrics.false_positives += metrics.false_positives
            total_metrics.false_negatives += metrics.false_negatives

        return total_metrics

class ExperimentManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_experiments(self, gt_paths: List[str], pred_paths: List[str]) -> None:
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Precision-Recall'])
        experiments = []

        for gt_path, pred_path in zip(gt_paths, pred_paths):
            name = Path(pred_path).parent.name
            # try:
            experiment = self._run_single_experiment(gt_path, pred_path, name)
            if experiment:
                experiments.append(experiment)
                self._add_to_plot(fig, experiment)
            # except Exception as e:
            #     print(f'Error processing {name}: {e}')

        self._save_results(experiments, fig)

    def _run_single_experiment(self, gt_path: str, pred_path: str, name: str) -> Optional[Dict]:
        analyzer = COCOAnalyzer(gt_path, pred_path)
        cocoGt = COCO(self._load_json(gt_path))
        cocoDt = cocoGt.loadRes(self._load_json(pred_path))
        curves = Curves(cocoGt, 
                       cocoDt, 
                       iou_tresh=0.1, 
                       iouType='bbox')
        
        _, precision, recall, scores, _ = curves.plot_pre_rec(plotly_backend=True)
        
        # if not scores:
        #     return None
            
        metrics_list = [analyzer.calculate_metrics(score) for score in scores]
        
        return {
            "name": name,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "scores": scores.tolist(),
            "metrics": metrics_list
        }

    @staticmethod
    def _load_json(path: str) -> Dict:
        with open(path) as f:
            return json.load(f)

    def _save_results(self, experiments: List[Dict], fig: go.Figure) -> None:
        # with open(self.output_dir / "results.json", "w") as f:
        #     json.dump({"experiments": experiments}, f)
        
        fig.write_image(self.output_dir / "pr_curve.png")
        fig.write_html(self.output_dir / "pr_curve.html")

    @staticmethod
    def _add_to_plot(fig: go.Figure, experiment: Dict) -> None:
        """
        Add experiment results to the plotly figure.
        
        Args:
            fig: Plotly figure object
            experiment: Dictionary containing experiment results
        """
        name = experiment["name"]
        r = experiment["recall"]
        p = experiment["precision"]
        scores = experiment["scores"]
        metrics = experiment["metrics"]
        
        # Stack all the metrics data for hover information
        customdata = np.column_stack((
            # Calculate F1 scores
            2 * (np.array(p) * np.array(r)) / (np.array(p) + np.array(r) + 1e-10),
            # True Positives
            [m.true_positives for m in metrics],
            # False Positives
            [m.false_positives for m in metrics],
            # False Negatives
            [m.false_negatives for m in metrics],
            # Precision from metrics (TP/(TP+FP))
            [m.true_positives/(m.true_positives + m.false_positives + 1e-10) for m in metrics],
            # Recall from metrics (TP/(TP+FN))
            [m.true_positives/(m.true_positives + m.false_negatives + 1e-10) for m in metrics],
            # Images with predictions only
            [m.images_with_predictions_only for m in metrics],
            # False positives in images with predictions only
            [m.fp_in_images_with_predictions_only for m in metrics]
        ))

        # Calculate AUC for the legend
        auc = np.trapz(p, r)
        
        # Add trace to figure
        fig.add_trace(
            go.Scatter(
                x=r,
                y=p,
                name=f'{name}:{auc:.3f}',
                text=scores,
                customdata=customdata,
                hovertemplate=(
                    'Precision: %{y:.3f}<br>' +
                    'Recall: %{x:.3f}<br>' +
                    'F1: %{customdata[0]:.3f}<br>' +
                    'Score: %{text:.3f}<br>' +
                    'TP: %{customdata[1]:.0f}<br>' +
                    'FP: %{customdata[2]:.0f}<br>' +
                    'FN: %{customdata[3]:.0f}<br>' +
                    'Metric Precision: %{customdata[4]:.3f}<br>' +
                    'Metric Recall: %{customdata[5]:.3f}<br>' +
                    'Images w/o GT: %{customdata[6]:.0f}<br>' +
                    'FP in Images w/o GT: %{customdata[7]:.0f}<br>' +
                    f'Name: {name}<extra></extra>'
                ),
                showlegend=True,
                mode='lines',
            ),
            row=1,
            col=1
        )

        # Add reference line at recall = 0.9
        fig.add_vline(
            x=0.9,
            line_dash="dash",
            line_color="red",
            annotation_text="Recall = 90%",
            annotation_position="bottom right"
        )

        # Update layout
        margin = 0.01
        fig.update_layout(
            height=600,
            width=1200,
            yaxis=dict(
                title='Precision',
                range=[-margin, 1 + margin]
            ),
            xaxis=dict(
                title='Recall',
                range=[-margin, 1 + margin]
            )
        )
if __name__ == "__main__":
    output_dir = Path("/home/m32patel/projects/def-dclausi/whale/mmwhale2/result_viz")
    manager = ExperimentManager(output_dir)

    GT_json = [
        '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/AED/test.json',
        # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/test.json',
        # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial-livestock-dataset/test/test.json',
        # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/test.json',
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
        # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/savmap_dataset_v2/test.json',
        # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/test_filtered.json',
        # '/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB_grounded.json'
    ]
    # List of prediction JSON file paths
    pred_json = [
        "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/AED_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Aerial_seabird_westafrica_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/aerial_livestock_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/birds_izembek_lagoon_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/michigan_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/monash_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/new_mexico_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/palmyra_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/penguins_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/pfeifer_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/seabirdwatch_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/qian_od_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/WAID_livestock_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Eikelboom_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_sealion_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/turtle_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/NOAA_artic_seal_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2014_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",  # whale data based on year
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Beluga_2015_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Narwhal_2016_dataset/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/SAVMAP/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/polar_bear/prediction_mm_grounding_dino_nocaption.bbox.json",
        # "/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/Virunga_garamba_dataset/prediction_mm_grounding_dino_nocaption.bbox.json"
    ]
    manager.run_experiments(GT_json, pred_json)