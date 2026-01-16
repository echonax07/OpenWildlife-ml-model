#!/usr/bin/env python3
"""
Create a new Label Studio project and copy specific tasks (with annotations/predictions)
from an existing project based on filename filters.
"""

import json
from typing import List, Dict, Any
from label_studio_sdk import Client
from urllib.parse import urlparse, unquote
import os


# Configuration
LS_URL = 'http://129.97.250.147:8080'
API_TOKEN = 'ebdc6fa5f2c3abcd502b55d5ccc1dc0e4ae9f68d'
SOURCE_PROJECT_ID = 110

# New project configuration
NEW_PROJECT_TITLE = "Heavy correction"
NEW_PROJECT_DESCRIPTION = "Project with Heavy correction tasks copied from project 110"

# Filename filter strings
FILENAME_FILTERS = [
    "8723", "8729", "8651", "8650", "8649", "8648", "8724", "8627",
    "8615", "8592", "8584", "8581", "8576", "8506", "7491", "7483",
    "4515", "4399", "4389", "4388", "4387", "4200", "4129", "4125",
    "4121", "4118", "4112", "BEL_03_001", "BEL_03_007"
]


class LabelStudioProjectCopier:
    def __init__(self, url: str, api_key: str, source_project_id: int):
        self.client = Client(url=url, api_key=api_key)
        self.source_project_id = source_project_id
        self.source_project = None
        self.new_project = None

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
            filename = os.path.basename(image_url.split('/')[-1])
        else:
            filename = os.path.basename(image_url)

        return filename

    def should_include_task(self, task: Dict, filter_strings: List[str]) -> bool:
        """Check if task should be included based on filename filters"""
        image_url = task.get('data', {}).get('image', '')
        if not image_url:
            return False

        filename = self.extract_filename_from_url(image_url)
        return any(filter_str in filename for filter_str in filter_strings)

    def get_source_project_config(self) -> str:
        """Get the label configuration XML from source project"""
        try:
            self.source_project = self.client.get_project(self.source_project_id)
            label_config = self.source_project.get_params().get('label_config', '')
            return label_config
        except Exception as e:
            print(f"Error getting source project config: {e}")
            raise

    def create_new_project(self, title: str, description: str, label_config: str) -> int:
        """Create a new Label Studio project with the same configuration"""
        try:
            print(f"Creating new project: '{title}'")

            # Create project with label config
            self.new_project = self.client.start_project(
                title=title,
                label_config=label_config
            )

            # Update description if API supports it
            try:
                # Some Label Studio versions support updating description
                project_params = self.new_project.get_params()
                project_params['description'] = description
                self.new_project.set_params(project_params)
            except:
                pass

            print(f"New project created with ID: {self.new_project.id}")
            return self.new_project.id

        except Exception as e:
            print(f"Error creating new project: {e}")
            raise

    def copy_task_to_new_project(self, task: Dict) -> Dict:
        """Copy a single task with its annotations and predictions to the new project"""
        try:
            # Prepare complete task data including annotations and predictions
            task_data = {
                'data': task.get('data', {})
            }

            # Include annotations if they exist
            if task.get('annotations'):
                task_data['annotations'] = task['annotations']

            # Include predictions if they exist
            if task.get('predictions'):
                task_data['predictions'] = task['predictions']

            # Import the task with all its data
            # Label Studio should automatically handle annotations and predictions
            result = self.new_project.import_tasks([task_data])

            return {
                'success': True,
                'num_annotations': len(task.get('annotations', [])),
                'num_predictions': len(task.get('predictions', []))
            }

        except Exception as e:
            print(f"  Error copying task: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def copy_filtered_tasks(self, filter_strings: List[str]):
        """Main function to copy filtered tasks to new project"""
        print("=" * 80)
        print("Label Studio Project Copy Tool")
        print("=" * 80)
        print()

        # Get source project and its configuration
        print(f"Connecting to source project (ID: {self.source_project_id})...")
        label_config = self.get_source_project_config()
        print(f"Source project: {self.source_project.title}")
        print(f"Retrieved label configuration ({len(label_config)} characters)")
        print()

        # Create new project
        new_project_id = self.create_new_project(
            NEW_PROJECT_TITLE,
            NEW_PROJECT_DESCRIPTION,
            label_config
        )
        print()

        # Get all tasks from source project
        print("Fetching tasks from source project...")
        source_tasks = self.source_project.get_tasks()
        print(f"Found {len(source_tasks)} total tasks in source project")
        print()

        # Filter and copy tasks
        print(f"Filtering tasks for filenames: {filter_strings}")
        print()

        tasks_to_copy = []
        for task in source_tasks:
            if self.should_include_task(task, filter_strings):
                image_url = task.get('data', {}).get('image', '')
                filename = self.extract_filename_from_url(image_url)
                tasks_to_copy.append((task, filename))

        print(f"Found {len(tasks_to_copy)} tasks matching filter criteria")
        print()

        if not tasks_to_copy:
            print("No tasks to copy. Exiting.")
            return

        # Copy each task
        print("Copying tasks to new project...")
        print("-" * 80)

        copied_count = 0
        total_annotations = 0
        total_predictions = 0

        for i, (task, filename) in enumerate(tasks_to_copy, 1):
            task_id = task['id']
            num_annotations = len(task.get('annotations', []))
            num_predictions = len(task.get('predictions', []))

            print(f"[{i}/{len(tasks_to_copy)}] Task {task_id} ({filename})")
            print(f"  Annotations: {num_annotations}, Predictions: {num_predictions}")

            result = self.copy_task_to_new_project(task)

            if result['success']:
                copied_count += 1
                total_annotations += result['num_annotations']
                total_predictions += result['num_predictions']
                print(f"  ✓ Successfully copied")
            else:
                print(f"  ✗ Failed to copy: {result.get('error', 'Unknown error')}")

            print()

        # Summary
        print("=" * 80)
        print("Copy Complete!")
        print("=" * 80)
        print(f"Source Project ID: {self.source_project_id}")
        print(f"New Project ID: {new_project_id}")
        print(f"Tasks copied: {copied_count}/{len(tasks_to_copy)}")
        print(f"Total annotations: {total_annotations}")
        print(f"Total predictions: {total_predictions}")
        print(f"\nNew project URL: {LS_URL}/projects/{new_project_id}")
        print("=" * 80)


def main():
    copier = LabelStudioProjectCopier(
        url=LS_URL,
        api_key=API_TOKEN,
        source_project_id=SOURCE_PROJECT_ID
    )

    copier.copy_filtered_tasks(FILENAME_FILTERS)


if __name__ == "__main__":
    main()
