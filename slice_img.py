import argparse
import os
import json
from sahi.slicing import slice_coco
from typing import Any, Dict, Optional

''' USAGE
python slice_img.py --enable=True --data_root_whole=/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Aerial_Seabirds_West_Africa/ --data_root_slice=/home/m32patel/scratch/animal_patches/Aerial_Seabirds_West_Africa/ --ann_file=train.json --img_dir=''  --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True 


python slice_img.py --enable=True --data_root_whole=/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba --data_root_slice=/home/m32patel/scratch/animal_patches/Virunga_Garamba/ --ann_file=groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json --img_dir='train'  --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True


python slice_img.py --enable=True --data_root_whole=/home/m32patel/projects/def-dclausi/whale/merged/train --data_root_slice=/home/m32patel/scratch/animal_patches/2017_Beluga/ --ann_file=/home/m32patel/projects/def-dclausi/whale/merged/train/split_dataset_2017.json --img_dir=''  --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True

sahi coco slice --image_dir /home/m32patel/projects/def-dclausi/whale/merged/train --dataset_json_path /home/m32patel/projects/def-dclausi/whale/merged/train/split_dataset_201117.json --slice_size 1024 --overlap_ratio 0 --output_dir /lustre07/scratch/m32patel/animal_patches/2017_Beluga_all_patches/

sahi coco slice --image_dir /home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated --dataset_json_path /home/m32patel/projects/rrg-dclausi/wildlife/datasets/polar_bear_annotated/train_5.json --slice_size 1024 --overlap_ratio 0 --output_dir /lustre07/scratch/m32patel/animal_patches/polar_bear

sahi coco slice --image_dir /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/train --dataset_json_path /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Virunga_Garamba/groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json --slice_size 1024 --overlap_ratio 0 --output_dir /lustre07/scratch/m32patel/animal_patches/Virunga_Garamba_all_patches


python slice_img.py --enable=True --data_root_whole='' --data_root_slice=/home/m32patel/scratch/animal_patches/DFO_whale_23/ --ann_file=/home/m32patel/projects/rrg-dclausi/whale/dataset/2023_Survey_DFO/DFO_2023_Survey_annotation/High_Arctic_Survey/Plane1/corrected_tasks/coco_iter12345_train.json --img_dir='' --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True

python slice_img.py --enable=True --data_root_whole='/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images' --data_root_slice=/home/m32patel/scratch/animal_patches/SAVMAP/ --ann_file=/home/m32patel/projects/rrg-dclausi/wildlife/datasets/SAVMAP_test/images/coco_split_train.json --img_dir='' --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True


python slice_img.py --enable=True --data_root_whole=/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/images --data_root_slice=/home/m32patel/scratch/animal_patches/eider_duck_patches --ann_file=/home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/coco_eider_ducks_v4.json --img_dir=''  --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True 


'''



def slice_train_images(enable, data_root_whole, data_root_slice, slice_height, slice_width, ann_file, img_dir, overlap_height_ratio=0, overlap_width_ratio=0, save_only_positive_slices=False):
    """Perform slicing of training images and save the patches."""
    if enable:
        print(f'Doing slicing with patch size: {slice_height}, {slice_width}!')
        
        if is_directory_empty(os.path.join(data_root_slice, 'train')):
            do_slicing_and_merge_annotation(data_root_whole, data_root_slice, ann_file, img_dir, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, save_only_positive_slices)
        else:
            print(f'Folder: {os.path.join(data_root_slice, "train")} found with sliced images and annotation')
            print('Skipping slicing...')
        
        ann_file_path = os.path.abspath(os.path.join(data_root_slice, 'train', 'train.json'))
        return ann_file_path
    else:
        return ann_file

def do_slicing_and_merge_annotation(data_root_whole, data_root_slice, ann_file, img_dir, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, save_only_positive_slices):
    coco_annotation_file_path = os.path.abspath(os.path.join(data_root_whole, ann_file))
    image_dir = os.path.abspath(os.path.join(data_root_whole, img_dir))
    output_coco_annotation_file_name = os.path.abspath(os.path.join(data_root_slice, 'train', 'train_slice.json'))
    output_dir = os.path.join(data_root_slice, 'train')

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
    full_img_dir = os.path.abspath(os.path.join(data_root_whole, img_dir))
    slice_img_dir = os.path.abspath(os.path.join(data_root_slice, 'train'))
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

def coco_merge(input_extend: str, input_add: str, extend_dir_path: str, add_dir_path: str, output_file: str, indent: Optional[int] = None) -> str:
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

    # print(f"Result: {len(output['images'])} images, {len(output['annotations'])} annotations")

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
    parser.add_argument('--enable', type=bool, default=True, help='Enable slicing')
    parser.add_argument('--data_root_whole', type=str, required=True, help='Root directory for whole images and annotations')
    parser.add_argument('--data_root_slice', type=str, required=True, help='Directory where sliced images will be saved')
    parser.add_argument('--slice_height', type=int, default=512, help='Height of image slices')
    parser.add_argument('--slice_width', type=int, default=512, help='Width of image slices')
    parser.add_argument('--overlap_height_ratio', type=float, default=0, help='Height overlap ratio between slices')
    parser.add_argument('--overlap_width_ratio', type=float, default=0, help='Width overlap ratio between slices')
    parser.add_argument('--save_only_positive_slices', type=bool, default=False, help='Whether to save only slices with annotations')
    parser.add_argument('--ann_file', type=str, required=True, help='Annotation file for the training data (COCO format)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory for training images')

    args = parser.parse_args()

    # Call the slicing function
    slice_train_images(
        args.enable, 
        args.data_root_whole, 
        args.data_root_slice, 
        args.slice_height, 
        args.slice_width, 
        args.ann_file, 
        args.img_dir, 
        args.overlap_height_ratio, 
        args.overlap_width_ratio, 
        args.save_only_positive_slices
    )

if __name__ == '__main__':
    main()


