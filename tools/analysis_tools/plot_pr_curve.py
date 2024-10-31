from icecream import ic
import os
import matplotlib.pyplot as plt
import json
import logging
from faster_coco_eval import COCO
from faster_coco_eval.extra import Curves
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def load(file):
    with open(file) as io:
        _data = json.load(io)
    return _data


def plot_confusion_matrix(gt_path, prediction_path):
    threshold_iou = 0.5
    score_threshold = 0
    bbox_difference_threshold = 10000
    prepared_coco_in_dict = load(gt_path)
    prepared_anns = load(prediction_path)
    print(f'Total predictions: {len(prepared_anns)}')
    thresholded_score_list = [
        ann for ann in prepared_anns if ann['score'] >= score_threshold]
    prepared_anns = [
        ann for ann in thresholded_score_list if abs(ann['bbox'][2]-ann['bbox'][3]) <= bbox_difference_threshold]

    print(f'Total prediction after thresholding: {len(prepared_anns)}')
    if len(prepared_anns) == 0:
        print('Annotations are empty after thresholding')
        return len(prepared_anns), 0, 0, 0, 0, 0
    else:
        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)
        cur = Curves(cocoGt, cocoDt, iou_tresh=threshold_iou, iouType='bbox')
        fig, recall, precision, scores, names = cur.plot_pre_rec(
            plotly_backend=False)
        return len(prepared_anns), cur, recall, precision, scores, names


def get_plotting_data(all_exp_data, name):
    for exp in all_exp_data['experiments']:
        if name == exp['name']:
            return exp['len_anns'], exp['r'], exp['p'], exp['score'], exp['auc'], exp['fp'], exp['fn'], exp['fp_in_images_with_predictions_only'], exp['r2'], exp['p2'], exp['no_GT_only_pred']


GT_json = '/home/m32patel/projects/def-dclausi/whale/merged/test/test.json'
pred_json = [

    # 'work_dirs/DFO_whale_deformable_detr_256/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_512/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_1024/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_2048/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_4096/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_resize_0.5/prediction.bbox.json',

    # 'work_dirs/DFO_whale_deformable_detr_256_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_512_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_1024_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_2048_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_4096_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_resize_0.5_swin_T/prediction.bbox.json',

    # 'work_dirs/DFO_whale_deformable_detr_256_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_512_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_1024_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_2048_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_4096_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_resize_0.5_convnext_T/prediction.bbox.json',

    # 'work_dirs/DFO_whale_deformable_detr_256_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_512_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_1024_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_2048_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_4096_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_deformable_detr_resize_0.5_mamba/prediction.bbox.json',

    # '/home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dirs/whale_faster_rcnn_clean_256/sliced_prediction_iter_20000.json',
    # 'work_dirs/DFO_whale_faster_rcnn_512/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_1024/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_2048/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_4096/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_resize_0.6/prediction.bbox.json',

    # 'work_dirs/DFO_whale_faster_rcnn_256_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_512_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_1024_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_2048_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_4096_swin_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_resize_0.5_swin_T/prediction.bbox.json',

    # 'work_dirs/DFO_whale_faster_rcnn_256_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_512_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_1024_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_2048_convnext_T/prediction.bbox.json',
    'work_dirs/DFO_whale_faster_rcnn_4096_convnext_T/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_resize_0.5_convnext_T/prediction.bbox.json',

    # 'work_dirs/DFO_whale_faster_rcnn_256_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_512_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_1024_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_2048_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_4096_mamba/prediction.bbox.json',
    # 'work_dirs/DFO_whale_faster_rcnn_resize_0.5_mamba/prediction.bbox.json',


    # 'work_dirs/DFO_whale_yolov8s_256/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8s_512/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8s_1024/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8s_2048/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8s_4096/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8s_resize_0.5/prediction.bbox.json',


    # 'work_dirs/DFO_whale_yolov8l_256/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8l_512/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8l_1024/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8l_2048/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8l_4096/prediction.bbox.json',
    # 'work_dirs/DFO_whale_yolov8l_resize_0.5/prediction.bbox.json',


    #  '../../work_dirs/DFO_whale_faster_rcnn_1024_lr_0.00002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_1024/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_1024_lr_0.0002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_1024_lr_0.02/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_2048_lr_0.00002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_2048_lr_0.0002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_2048_lr_0.02/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_2048/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_256_lr_0.00002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_256_lr_0.02/prediction.bbox.json',
    # #  '../../work_dirs/DFO_whale_faster_rcnn_256/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_4096_lr_0.0002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_4096/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_512_lr_0.00002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_512_lr_0.0002/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_512_lr_0.02/prediction.bbox.json',
    #  '../../work_dirs/DFO_whale_faster_rcnn_512/prediction.bbox.json',
]
names = [os.path.basename(os.path.dirname(path)) for path in pred_json]

add_suffix = ['whale_faster_rcnn_clean_256',
              'DFO_whale_faster_rcnn_512',
              'DFO_whale_faster_rcnn_1024',
              'DFO_whale_faster_rcnn_2048',
              'DFO_whale_faster_rcnn_4096',
              'DFO_whale_faster_rcnn_resize_0.6'
              'DFO_whale_deformable_detr_256',
              'DFO_whale_deformable_detr_512',
              'DFO_whale_deformable_detr_1024',
              'DFO_whale_deformable_detr_2048',
              'DFO_whale_deformable_detr_4096',
              'DFO_whale_deformable_detr_resize_0.5',]


all_exp_data = load(
    'configs/DFO_whale_different_backbones/all_experiments.json')

ic(names)

fig = make_subplots(rows=1, cols=1, subplot_titles=['Precision-Recall'])

experiments = []
plotting_data = []

for pred, name in zip(pred_json, names):
    try:
        len_anns, r, p, score, auc, fp, fn, fp_in_images_with_predictions_only, r2, p2, no_GT_only_pred = get_plotting_data(
            all_exp_data, name)
        auc_float = float(auc.replace("auc: ", ""))
        if len_anns == 0:
            continue
        if name in add_suffix:
            if name == 'whale_faster_rcnn_clean_256':
                name = 'DFO_whale_faster_rcnn_256'
            name = name + '_resnet50'
        name = name.replace('DFO_whale_', '')
        plotting_data.append(
            (name, r, p, score, auc, fp, fn, auc_float, fp_in_images_with_predictions_only, r2, p2, no_GT_only_pred))
    except Exception as e:
        print(f'Exception {e} found')
        print(f'File {pred} may not be found, please check the path')
        continue

plotting_data.sort(key=lambda x: x[7], reverse=True)
# List of distinct colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for index, (name, r, p, score, auc, fp, fn, auc_float, fp_in_images_with_predictions_only, r2, p2, no_GT_only_pred) in enumerate(plotting_data):
    fig.add_trace(
        go.Scatter(
            x=r,
            y=p,
            name=f'{name}:{auc}',
            text=score,
            hovertemplate='Pre: %{y:.3f}<br>' +
            'Rec: %{x:.3f}<br>' +
            'Score: %{text:.3f}<br>' +
            f'FP: {fp}<br>' +
            f'FN: {fn}<br>' +
            f'r2: {r2}<br>' +
            f'p2: {p2}<br>' +
            f'images_without_GT_but_with_predictions: {no_GT_only_pred}<br>' +
            f'fp_in_images_without_GT_but_with_predictions: {fp_in_images_with_predictions_only}<br>' +
            f'Name: {name}<extra></extra>',
            showlegend=True,
            mode='lines',
            line=dict(color=colors[index])
        ),
        row=1, col=1
    )

fig.add_vline(x=0.9, line_dash="dash", line_color="red",
              annotation_text="Recall = 90%",
              annotation_position="bottom right")
margin = 0.01
fig.layout.yaxis.range = [0 - margin, 1 + margin]
fig.layout.xaxis.range = [0 - margin, 1 + margin]

fig.layout.yaxis.title = 'Precision'
fig.layout.xaxis.title = 'Recall'
fig.update_layout(height=600, width=1200)

# Spider Plot Function


def spider(df, *, id_column, title=None, max_values=None, padding=1.25):
    # categories = df._get_numeric_data().columns.tolist()
    categories = df.dtypes[(df.dtypes == 'float') | (
        df.dtypes == 'int')].index.tolist()
    ic(categories)
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding*max(value) for key, value in data.items()}

    normalized_data = {key: np.array(
        value) / max_values[key] for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.15)
        for _x, _y, t in zip(angles, values, actual_values):
            t = f'{t:.2f}' if isinstance(t, float) else str(t)
            # ax.text(_x, _y, t, size='x-small')

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    for i in range(len(tiks)):
        if tiks[i] in ['Images w/o GT but w/ Pred', 'FP in Images w/o GT but w/ Pred']:
            tiks[i] += '\n (Lower is better)'
    ax.set_xticklabels(tiks)
    ic(tiks)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    if title is not None:
        plt.suptitle(title)
    plt.show()
    plt.savefig('spider_plot.png', bbox_inches='tight', dpi=400)


# Prepare data for spider plot
spider_data = {
    'name': [],
    'R2': [],
    'P2': [],
    'Images w/o GT but w/ Pred': [],
    'FP in Images w/o GT but w/ Pred': []
}

for name, r, p, score, auc, fp, fn, auc_float, fp_in_images_with_predictions_only, r2, p2, no_GT_only_pred in plotting_data:
    spider_data['name'].append(name)
    spider_data['R2'].append(r2)
    spider_data['P2'].append(p2)
    spider_data['Images w/o GT but w/ Pred'].append(no_GT_only_pred)
    spider_data['FP in Images w/o GT but w/ Pred'].append(
        fp_in_images_with_predictions_only)

spider_df = pd.DataFrame(spider_data)

def spider_plotly(df, *, id_column, title=None, max_values=None, padding=1.25):
    categories = df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()

    if max_values is None:
        max_values = {key: padding * max(value) for key, value in data.items()}

    normalized_data = {key: np.array(
        value) / max_values[key] for key, value in data.items()}

    fig = go.Figure()

    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in categories]
        actual_values = [data[key][i] for key in categories]

        fig.add_trace(go.Scatterpolar(
            r=values + values[:1],
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name,
            text=[f'{v:.2f}' if isinstance(v, float) else str(
                v) for v in actual_values] + [''],
            textposition='middle center',
            hoverinfo='text+name'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=title
    )

    fig.show()
    # fig.write_image("spider_plot.png")

# # Plot the spider plot
spider(spider_df, id_column='name',
       title='')
# spider_plotly(spider_df, id_column='name', title='Spider Plot')
# Plot the spider plot
# spider_plotly(spider_df, id_column='name', title='Spider Plot for Whale Detection Models', padding=1.1)
fig.show()
