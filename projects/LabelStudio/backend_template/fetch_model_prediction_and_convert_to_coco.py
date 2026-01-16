#!/usr/bin/env python3
"""
Fetch Label Studio model predictions via API and convert to COCO format.
Filters images by specified string patterns in filenames.
"""

import json
import os
import requests
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import urlparse, unquote
from label_studio_sdk import Client


# Configuration
LS_URL = 'http://129.97.250.147:8080'
API_TOKEN = 'ebdc6fa5f2c3abcd502b55d5ccc1dc0e4ae9f68d'
PROJECT_ID = 110

# Image filter strings
IMAGE_FILTER_STRINGS = [
    "8865", "8795", "8762", "8756", "8715", "8714", "8713", "8633",
    "8499", "7492", "4665", "4652", "4648", "4641", "4638", "4605",
    "4595", "4592", "4591", "4585", "4581", "4579", "4575", "4573",
    "4563", "4561", "4510", "4476", "4466", "4462", "4454", "4449",
    "4437", "4432", "4425", "4421", "4412", "4408", "4405", "4386",
    "4384", "4383", "4369", "4367", "4364", "4361", "4355", "4242",
    "4176"
]

# Output directory and file
OUTPUT_DIR = 'model_predictions_coco_annotations'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'annotations.json')


class LabelStudioPredictionsToCOCO:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Predefined categories matching Label Studio configuration
        self.coco_data = {
            "info": {
                "description": "Filtered duck model predictions from Label Studio",
                "url": LS_URL,
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "LabelStudio Predictions to COCO Converter",
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

    def extract_filename_from_url(self, image_url: str) -> str:
        """Extract filename from Label Studio image URL"""
        is_uploaded_file = image_url.startswith("/data/upload")
        is_local_storage_file = image_url.startswith("/data/local-files") and "?d=" in image_url
        is_cloud_storage_file = (
            image_url.startswith("s3:") or
            image_url.startswith("gs:") or
            image_url.startswith("azure-blob:")
        )

        if is_uploaded_file:
            filename = os.path.basename(urlparse(image_url).path)
        elif is_local_storage_file:
            filename = os.path.basename(unquote(image_url.split('?d=')[1]))
        elif is_cloud_storage_file:
            # Extract filename from cloud storage URL
            filename = os.path.basename(image_url.split('/')[-1])
        else:
            # Fallback to basename
            filename = os.path.basename(image_url)

        return filename

    def should_include_image(self, filename: str, filter_strings: List[str]) -> bool:
        """Check if filename contains any of the filter strings"""
        return any(filter_str in filename for filter_str in filter_strings)

    def download_image(self, image_url: str, filename: str) -> bool:
        """Download image from Label Studio and save to output directory"""
        try:
            # Handle relative URLs
            if image_url.startswith('/'):
                full_url = LS_URL.rstrip('/') + image_url
            else:
                full_url = image_url

            # Download image
            headers = {'Authorization': f'Token {API_TOKEN}'}
            response = requests.get(full_url, headers=headers)
            response.raise_for_status()

            # Save image to output directory
            image_path = os.path.join(self.output_dir, filename)
            with open(image_path, 'wb') as f:
                f.write(response.content)

            return True
        except Exception as e:
            print(f"  Error downloading image: {e}")
            return False

    def convert_keypoint_to_bbox(self, x: float, y: float, width: float, height: float,
                                  img_width: int, img_height: int) -> List[float]:
        """Convert Label Studio keypoint percentages to COCO bbox format"""
        abs_width = width * img_width / 100.0
        abs_height = height * img_height / 100.0
        abs_x = x * img_width / 100.0 - abs_width / 2
        abs_y = y * img_height / 100.0 - abs_height / 2

        return [abs_x, abs_y, abs_width, abs_height]

    def process_task(self, task: Dict) -> bool:
        """
        Process a single Label Studio task and add predictions to COCO format.
        Returns True if task was processed, False if skipped.
        """
        # Get image URL
        image_url = task.get('data', {}).get('image')
        if not image_url:
            print(f"Task {task['id']}: No image URL found")
            return False

        # Extract filename
        filename = self.extract_filename_from_url(image_url)

        # Check if filename should be included
        if not self.should_include_image(filename, IMAGE_FILTER_STRINGS):
            return False

        # Download the image
        print(f"Task {task['id']} ({filename}): Downloading image...")
        if not self.download_image(image_url, filename):
            print(f"Task {task['id']} ({filename}): Failed to download image, skipping")
            return False

        # Check if task has predictions
        if not task.get('predictions') or not task['predictions']:
            print(f"Task {task['id']} ({filename}): No predictions found")
            return False

        # Get the latest prediction (or you can choose based on model_version, score, etc.)
        prediction = task['predictions'][-1]
        results = prediction.get('result', [])

        if not results:
            print(f"Task {task['id']} ({filename}): Empty prediction results")
            return False

        # Extract image dimensions from first result with original dimensions
        img_width = img_height = None
        for result in results:
            if 'original_width' in result and 'original_height' in result:
                img_width = result['original_width']
                img_height = result['original_height']
                break

        if img_width is None or img_height is None:
            print(f"Task {task['id']} ({filename}): No image dimensions found")
            return False

        # Extract bbox dimensions from textarea predictions
        bbox_width = bbox_height = None
        for result in results:
            if result.get('type') == 'textarea':
                if result.get('from_name') == 'width':
                    try:
                        bbox_width = float(result['value']['text'][0])
                    except (KeyError, IndexError, ValueError):
                        pass
                elif result.get('from_name') == 'height':
                    try:
                        bbox_height = float(result['value']['text'][0])
                    except (KeyError, IndexError, ValueError):
                        pass

        if bbox_width is None or bbox_height is None or bbox_width <= 0 or bbox_height <= 0:
            print(f"Task {task['id']} ({filename}): Invalid or missing bbox dimensions")
            return False

        # Add image entry to COCO format
        image_entry = {
            "id": self.image_id,
            "width": img_width,
            "height": img_height,
            "file_name": filename,
            "license": 1,
            "date_captured": task.get('created_at', ''),
            "ls_task_id": task['id'],  # Store original task ID for reference
            "prediction_model_version": prediction.get('model_version', 'unknown')
        }

        self.coco_data["images"].append(image_entry)
        self.image_id_mapping[filename] = self.image_id
        current_image_id = self.image_id
        self.image_id += 1

        # Process keypoint predictions
        keypoint_count = 0
        for result in results:
            if result.get('type') == 'keypointlabels':
                value = result.get('value', {})

                # Get label
                label = value.get('keypointlabels', ['duck'])[0]
                category_id = self.category_mapping.get(label, 4)  # Default to "duck"

                # Get confidence score if available
                score = result.get('score', 1.0)

                # Convert to COCO bbox format
                x = value.get('x', 0)
                y = value.get('y', 0)
                w = value.get('width', 0)
                h = value.get('height', 0)

                bbox = self.convert_keypoint_to_bbox(
                    x, y, bbox_width, bbox_height,
                    img_width, img_height
                )

                area = bbox[2] * bbox[3]

                # Create COCO annotation
                annotation_entry = {
                    "id": self.annotation_id,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "score": score  # Add confidence score for predictions
                }

                self.coco_data["annotations"].append(annotation_entry)
                self.annotation_id += 1
                keypoint_count += 1

        print(f"Task {task['id']} ({filename}): Processed {keypoint_count} predictions")
        return True

    def fetch_and_convert(self, output_file: str):
        """Main function to fetch tasks from Label Studio and convert predictions to COCO format"""
        print("Connecting to Label Studio...")

        # Initialize Label Studio client
        ls = Client(url=LS_URL, api_key=API_TOKEN)

        try:
            # Get project
            project = ls.get_project(PROJECT_ID)
            print(f"Successfully connected to project: {project.title}")

            # Get all tasks
            tasks = project.get_tasks()
            print(f"Found {len(tasks)} total tasks in project")
            print(f"Filtering for images containing: {IMAGE_FILTER_STRINGS}")
            print()

            # Process each task
            processed_count = 0
            for i, task in enumerate(tasks):
                if self.process_task(task):
                    processed_count += 1

            print()
            print(f"Processed {processed_count} out of {len(tasks)} tasks")
            print(f"Total images in COCO format: {len(self.coco_data['images'])}")
            print(f"Total predictions in COCO format: {len(self.coco_data['annotations'])}")

            # Save COCO format JSON
            print(f"\nSaving COCO predictions to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.coco_data, f, indent=2)

            print("\nConversion completed successfully!")
            print(f"Images: {len(self.coco_data['images'])}")
            print(f"Predictions: {len(self.coco_data['annotations'])}")
            print(f"Categories: {[cat['name'] for cat in self.coco_data['categories']]}")

            # Print summary by category
            if self.coco_data['annotations']:
                print("\nPredictions by category:")
                category_counts = {}
                for ann in self.coco_data['annotations']:
                    cat_id = ann['category_id']
                    cat_name = next(c['name'] for c in self.coco_data['categories'] if c['id'] == cat_id)
                    category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

                for cat_name, count in sorted(category_counts.items()):
                    print(f"  {cat_name}: {count}")

                # Print average confidence score
                avg_score = sum(ann.get('score', 1.0) for ann in self.coco_data['annotations']) / len(self.coco_data['annotations'])
                print(f"\nAverage confidence score: {avg_score:.3f}")

        except Exception as e:
            print(f"Error: {e}")
            print("Make sure your Label Studio server is running and accessible")
            raise


def main():
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Annotations will be saved to: {OUTPUT_FILE}")
    print()

    converter = LabelStudioPredictionsToCOCO(OUTPUT_DIR)
    converter.fetch_and_convert(OUTPUT_FILE)


if __name__ == "__main__":
    main()
