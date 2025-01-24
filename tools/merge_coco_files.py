import json
from typing import Any, Dict, List, Optional
from loguru import logger
import argparse
import os

''' Example
python tools/merge_coco_files.py \
--inputs /home/m32patel/scratch/animal_patches/eider_duck_patches/train/train_slice_filtered.json_coco.json /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/dense/annotations.json /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/sparse/annotations.json \
--folders /home/m32patel/scratch/animal_patches/eider_duck_patches/train/ /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/dense/ /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/eider_duck_synth_images/aerial-duck-counting/synthesized_combined/sparse/ \
--output /home/m32patel/projects/rrg-dclausi/wildlife/datasets/Eider_duck_labelstudio-export/real_sparse_dense.json
'''


@logger.catch(reraise=True)
def merge_coco_files(
    input_files: List[str],
    folder_paths: List[str],
    output_file: str,
    indent: Optional[int] = None,
) -> str:
    """Merge multiple COCO annotation files, updating image paths.

    Args:
        input_files: List of paths to input COCO annotation files to be merged.
        folder_paths: List of corresponding folder paths for images in each input file.
        output_file: Path to output file with merged annotations.
        indent: Argument passed to `json.dump`. See https://docs.python.org/3/library/json.html#json.dump.
    """
    if not input_files or len(input_files) < 2:
        raise ValueError("You must provide at least two input files to merge.")

    if len(input_files) != len(folder_paths):
        raise ValueError("Each input file must have a corresponding folder path.")

    # Load the first file to start the merge
    with open(input_files[0], "r") as f:
        output = json.load(f)

    # Prepare structure for merged content
    output["images"] = []
    output["annotations"] = []

    for file_index, (file_path, folder_path) in enumerate(zip(input_files, folder_paths)):
        with open(file_path, "r") as f:
            data = json.load(f)

        logger.info(
            f"Processing file {file_index + 1}/{len(input_files)}: {file_path}"
        )
        logger.info(
            f"Input {file_index + 1}: {len(data['images'])} images, {len(data['annotations'])} annotations"
        )

        # Map categories between the current file and the merged output
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

        # Map images and annotations
        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            # Update the filename to include the full path
            image["file_name"] = os.path.join(folder_path, image["file_name"])

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

    logger.info(
        f"Result: {len(output['images'])} images, {len(output['annotations'])} annotations"
    )

    # Save the merged annotations to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=indent, ensure_ascii=False)

    return output_file


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO annotation files into a single file, updating image paths."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to input COCO annotation files to be merged. Provide at least two files.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="Folder paths corresponding to the images in each input file. Provide the same number of folders as input files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output COCO annotation file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indent level for the output JSON file (default: 4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        merge_coco_files(args.inputs, args.folders, args.output, args.indent)
        logger.info(f"Merged COCO annotations saved to {args.output}")
    except Exception as e:
        logger.error(f"Error merging COCO files: {e}")
