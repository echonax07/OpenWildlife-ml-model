#!/usr/bin/env python3
"""
Convert LabelStudio format annotations to MSCOCO format.

Fixed categories and numeric annotation IDs.
"""

import json
import argparse
import os
from datetime import datetime
from typing import Dict, Any


class LabelStudioToCOCO:
    def __init__(self):
        # Predefined categories
        self.coco_data = {
            "info": {
                "description": "",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "LabelStudio to COCO Converter",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Arctic Eider Society",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "female duck"},
                {"id": 1, "name": "male duck"},
                {"id": 2, "name": "Ice"},
                {"id": 3, "name": "Juvenile duck"},
                {"id": 4, "name": "duck"}
            ]
        }
        # Map labels to their fixed category ID
        self.category_mapping = {c["name"]: c["id"] for c in self.coco_data["categories"]}
        self.image_id_mapping = {}
        self.annotation_id = 1
        self.image_id = 1

    def extract_filename_from_path(self, path: str) -> str:
        return os.path.basename(path)

    def convert_keypoint_to_coco(self, x: float, y: float, width: float, height: float,
                                 img_width: int, img_height: int) -> Dict[str, Any]:
        abs_width = width * img_width / 100.0
        abs_height = height * img_height / 100.0
        abs_x = x * img_width / 100.0 - abs_width / 2
        abs_y = y * img_height / 100.0 - abs_height / 2

        return {
            "bbox": [abs_x, abs_y, abs_width, abs_height]
        }

    def process_image(self, ls_item: Dict) -> tuple[int, int, int]:
        image_path = ls_item["data"]["image"]
        filename = self.extract_filename_from_path(image_path)

        if filename in self.image_id_mapping:
            raise ValueError(f"Duplicate image filename found: {filename}")

        width = height = None
        if ls_item["annotations"] and ls_item["annotations"][0]["result"]:
            first_result = ls_item["annotations"][0]["result"][0]
            width = first_result.get("original_width")
            height = first_result.get("original_height")
        else:
            raise KeyError(f"No annotations found for image {filename}.")

        image_entry = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1,
            "date_captured": ls_item.get("created_at", "")
        }

        self.coco_data["images"].append(image_entry)
        self.image_id_mapping[filename] = self.image_id
        current_image_id = self.image_id
        self.image_id += 1

        return current_image_id, width, height

    def process_annotation(self, result: Dict, image_id: int, img_width: int, img_height: int,
                           bbox_width: float, bbox_height: float):
        annotation_type = result["type"]
        value = result["value"]

        if annotation_type != "keypointlabels":
            return

        label = value["keypointlabels"][0] if "keypointlabels" in value else "duck"
        category_id = self.category_mapping.get(label, 4)  # Default to "duck" if unknown

        # Convert from percentages to absolute bbox
        rect_data = self.convert_keypoint_to_coco(
            value["x"], value["y"], bbox_width, bbox_height,
            img_width, img_height
        )

        abs_x, abs_y, abs_w, abs_h = rect_data["bbox"]
        area = abs_w * abs_h

        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [abs_x, abs_y, abs_w, abs_h],
            "area": area,
            "iscrowd": 0
        }

        self.coco_data["annotations"].append(annotation)
        self.annotation_id += 1


    def convert(self, labelstudio_file: str, output_file: str):
        print(f"Loading LabelStudio file: {labelstudio_file}")

        with open(labelstudio_file, 'r', encoding='utf-8') as f:
            ls_data = json.load(f)

        print(f"Processing {len(ls_data)} items...")

        for item in ls_data:
            image_id, img_width, img_height = self.process_image(item)

            bbox_width = bbox_height = -1.0
            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    if result["type"] == "textarea":
                        if result["from_name"] == "width":
                            bbox_width = float(result["value"]["text"][0])
                        elif result["from_name"] == "height":
                            bbox_height = float(result["value"]["text"][0])
                    if bbox_width > 0 and bbox_height > 0:
                        break

            if bbox_width <= 0 or bbox_height <= 0:
                print(f"FATAL: Bounding box information could not be found for task {item['id']}")
                continue

            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    self.process_annotation(result, image_id, img_width, img_height, bbox_width, bbox_height)

        print(f"Saving COCO format to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, indent=2)

        print(f"\nConversion completed!")
        print(f"Images: {len(self.coco_data['images'])}")
        print(f"Annotations: {len(self.coco_data['annotations'])}")
        print(f"Categories: {len(self.coco_data['categories'])}")
        print(f"Categories: {[cat['name'] for cat in self.coco_data['categories']]}")


def main():
    parser = argparse.ArgumentParser(description="Convert LabelStudio annotations to MSCOCO format")
    parser.add_argument("input_file", help="Input LabelStudio JSON file")
    parser.add_argument("-o", "--output", help="Output COCO JSON file", default="coco_annotations.json")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1

    converter = LabelStudioToCOCO()
    converter.convert(args.input_file, args.output)


if __name__ == "__main__":
    exit(main())
