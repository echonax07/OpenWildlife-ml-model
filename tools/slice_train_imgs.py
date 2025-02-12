import argparse
import os
import json
from mmengine.config import Config
from sahi.slicing import slice_coco
from typing import Any, Dict, Optional

def slice_train_images(cfg, enable, slice_height, slice_width, overlap_height_ratio=0, overlap_width_ratio=0, save_only_positive_slices=False):
    """Perform slicing of training images and save the patches."""
    if enable:
        print(f'Doing slicing with patch size: {slice_height}, {slice_width}!')
        
        if is_directory_empty(os.path.join(cfg['data_root_slice'], 'train')):
            do_slicing_and_merge_annotation(cfg, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, save_only_positive_slices)
        else:
            print(f'Folder: {os.path.join(cfg["data_root_slice"], "train")} found with sliced images and annotation')
            print('Skipping slicing...')
        
        cfg['train_dataloader']['dataset']['data_root'] = ''
        cfg['train_dataloader']['dataset']['ann_file'] = os.path.abspath(os.path.join(cfg['data_root_slice'], 'train', 'train.json'))
        cfg['train_dataloader']['dataset']['data_prefix']['img'] = ''
        return cfg
    else:
        return cfg

def do_slicing_and_merge_annotation(cfg, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, save_only_positive_slices):
    coco_annotation_file_path = os.path.abspath(os.path.join(cfg['data_root_whole'], cfg['train_dataloader']['dataset']['ann_file']))
    image_dir = os.path.abspath(os.path.join(cfg['data_root_whole'], cfg['train_dataloader']['dataset']['data_prefix']['img']))
    output_coco_annotation_file_name = os.path.abspath(os.path.join(cfg['data_root_slice'], 'train', 'train_slice.json'))
    output_dir = os.path.join(cfg['data_root_slice'], 'train')

    # Do slicing
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=image_dir,
        output_coco_annotation_file_name=output_coco_annotation_file_name,
        ignore_negative_samples=save_only_positive_slices,
        output_dir=output_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        min_area_ratio=0.1,
        verbose=False,
        save_only_positive_slices=save_only_positive_slices
    )

    # Merge annotations with saving the paths in both the files
    full_img_dir =  os.path.abspath(os.path.join(cfg['data_root_whole'], cfg['train_dataloader']['dataset']['data_prefix']['img']))
    slice_img_dir = os.path.abspath(os.path.join(cfg['data_root_slice'], 'train'))
    output_coco_annotation_file_name = output_coco_annotation_file_name.replace('train_slice.json', 'train_slice.json_coco.json')
    final_output_file = output_coco_annotation_file_name.replace('train_slice.json_coco.json', 'train.json')

    final_output_file = coco_merge(
        input_extend=coco_annotation_file_path,
        input_add=output_coco_annotation_file_name,
        extend_dir_path=full_img_dir,
        add_dir_path=slice_img_dir,
        output_file=final_output_file
    )
    return final_output_file

def coco_merge(
    input_extend: str,
    input_add: str,
    extend_dir_path: str,
    add_dir_path: str,
    output_file: str,
    indent: Optional[int] = None,
) -> str:
    """Merge COCO annotation files."""
    with open(input_extend, "r") as f:
        data_extend = json.load(f)
    with open(input_add, "r") as f:
        data_add = json.load(f)
    
    for image in data_extend.get('images', []):
        if 'file_name' in image:
            image['file_name'] = os.path.join(extend_dir_path, image['file_name'])

    for image in data_add.get('images', []):
        if 'file_name' in image:
            image['file_name'] = os.path.join(add_dir_path, image['file_name'])

    output: Dict[str, Any] = {
        k: data_extend[k] for k in data_extend if k not in ("images", "annotations")
    }

    output["images"], output["annotations"] = [], []

    for i, data in enumerate([data_extend, data_add]):
        print(f"Input {i + 1}: {len(data['images'])} images, {len(data['annotations'])} annotations")
        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

    print(f"Result: {len(output['images'])} images, {len(output['annotations'])} annotations")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=indent, ensure_ascii=False)

    return output_file

def is_directory_empty(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    return len(os.listdir(directory_path)) == 0

def check_file_exists(file_path):
    return os.path.isfile(file_path)

def main():
    parser = argparse.ArgumentParser(description='Slice training images and save the patches.')
    parser.add_argument('config', help='Path to the config file')
    args = parser.parse_args()

    # Load configuration
    cfg = Config.fromfile(args.config)

    slice_configuration = cfg.slice_configuration
    slice_train_images(cfg, **slice_configuration)

    # Save the modified configuration back to the file
    cfg.dump(args.config)

if __name__ == '__main__':
    main()
