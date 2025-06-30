import os
import json
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from label_studio_sdk import Client
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import io
import base64

# LABEL STUDIO CONFIG
LS_URL = 'http://129.97.250.147:8080/'
API_TOKEN = 'ebdc6fa5f2c3abcd502b55d5ccc1dc0e4ae9f68d'
PROJECT_ID = 95

class ImageCropSplitter:
    def __init__(self, ls_url, api_token, project_id):
        self.client = Client(url=ls_url, api_key=api_token)
        self.project_id = project_id
        self.project = self.client.get_project(project_id)
        
    def get_task_by_id(self, task_id):
        """Get a specific task by ID"""
        try:
            task = self.project.get_task(task_id)
            return task
        except Exception as e:
            print(f"Error fetching task {task_id}: {str(e)}")
            return None
    
    def download_image(self, image_url, task_id):
        """Download image from Label Studio"""
        try:
            # Handle both relative and absolute URLs
            if image_url.startswith('/'):
                full_url = f"{LS_URL.rstrip('/')}{image_url}"
            else:
                full_url = image_url
            
            headers = {'Authorization': f'Token {API_TOKEN}'}
            response = requests.get(full_url, headers=headers)
            response.raise_for_status()
            
            # Save temporarily
            temp_path = f"temp_image_{task_id}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            return temp_path
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return None
    
    def get_annotations_in_region(self, annotations, x_start, y_start, x_end, y_end, img_width, img_height):
        """Get annotations that fall within a specific region"""
        region_annotations = []
        
        for annotation in annotations:
            for result in annotation.get('result', []):
                if result.get('type') == 'keypointlabels':
                    value = result.get('value', {})
                    
                    # Convert percentage coordinates to pixel coordinates
                    x_percent = value.get('x', 0)
                    y_percent = value.get('y', 0)
                    
                    x_pixel = (x_percent / 100) * img_width
                    y_pixel = (y_percent / 100) * img_height
                    
                    # Check if point is within the crop region
                    if x_start <= x_pixel <= x_end and y_start <= y_pixel <= y_end:
                        # Create new annotation with adjusted coordinates
                        new_result = result.copy()
                        
                        # Adjust coordinates relative to crop
                        crop_width = x_end - x_start
                        crop_height = y_end - y_start
                        
                        new_x_percent = ((x_pixel - x_start) / crop_width) * 100
                        new_y_percent = ((y_pixel - y_start) / crop_height) * 100
                        
                        new_result['value']['x'] = new_x_percent
                        new_result['value']['y'] = new_y_percent
                        
                        # Update to_name to match new task structure
                        new_result['to_name'] = 'image'
                        
                        region_annotations.append({
                            'result': [new_result],
                            'was_cancelled': annotation.get('was_cancelled', False),
                            'ground_truth': annotation.get('ground_truth', False),
                            'created_username': annotation.get('created_username', ''),
                            'updated_username': annotation.get('updated_username', ''),
                        })
        
        return region_annotations
    
    def create_preview_grid(self, image_path, v_blocks, h_blocks):
        """Create a preview of how the image will be split"""
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(f'Image Split Preview: {v_blocks}x{h_blocks} grid ({v_blocks * h_blocks} crops)')
        
        # Calculate block dimensions
        block_width = img_width / h_blocks
        block_height = img_height / v_blocks
        
        # Draw grid lines
        for i in range(1, h_blocks):
            x = i * block_width
            ax.axvline(x=x, color='red', linewidth=2, alpha=0.7)
        
        for i in range(1, v_blocks):
            y = i * block_height
            ax.axhline(y=y, color='red', linewidth=2, alpha=0.7)
        
        # Add crop numbers
        crop_num = 0
        for v in range(v_blocks):
            for h in range(h_blocks):
                crop_num += 1
                center_x = (h + 0.5) * block_width
                center_y = (v + 0.5) * block_height
                
                ax.text(center_x, center_y, str(crop_num), 
                       ha='center', va='center', fontsize=16, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontweight='bold')
        
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Flip Y axis for image coordinates
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def split_image_and_create_tasks(self, task_id, v_blocks, h_blocks):
        """Split image into crops and create new tasks"""
        # Get original task
        original_task = self.get_task_by_id(task_id)
        if not original_task:
            return False
        
        # Get image info
        task_data = original_task.get('data', {})
        image_path_key = 'image' if 'image' in task_data else 'img'
        image_url = task_data.get(image_path_key)
        
        if not image_url:
            print("No image found in task data")
            return False
        
        # Download image
        temp_image_path = self.download_image(image_url, task_id)
        if not temp_image_path:
            return False
        
        try:
            img = Image.open(temp_image_path)
            img_width, img_height = img.size
            
            # Get original filename
            original_filename = Path(urlparse(image_url).path).name
            base_name = Path(original_filename).stem
            extension = Path(original_filename).suffix
            
            # Calculate block dimensions
            block_width = img_width // h_blocks
            block_height = img_height // v_blocks
            
            # Get original annotations
            original_annotations = original_task.get('annotations', [])
            
            created_tasks = []
            crop_num = 0
            
            print(f"Creating {v_blocks * h_blocks} crops...")
            
            for v in range(v_blocks):
                for h in range(h_blocks):
                    crop_num += 1
                    
                    # Calculate crop boundaries
                    x_start = h * block_width
                    y_start = v * block_height
                    x_end = min((h + 1) * block_width, img_width)
                    y_end = min((v + 1) * block_height, img_height)
                    
                    # Crop image
                    crop = img.crop((x_start, y_start, x_end, y_end))
                    
                    # Save crop
                    crop_filename = f"{base_name}_crop_{crop_num:03d}{extension}"
                    crop_path = f"crops/{crop_filename}"
                    os.makedirs("crops", exist_ok=True)
                    crop.save(crop_path)
                    
                    # Get annotations for this region
                    region_annotations = self.get_annotations_in_region(
                        original_annotations, x_start, y_start, x_end, y_end, 
                        img_width, img_height
                    )
                    
                    # Upload crop to Label Studio
                    try:
                        # Upload the crop file
                        with open(crop_path, 'rb') as f:
                            upload_response = self.client.make_request(
                                'POST',
                                '/api/projects/{}/import'.format(self.project_id),
                                files={'file': (crop_filename, f, 'image/jpeg')}
                            )
                        
                        if upload_response.status_code != 201:
                            print(f"Failed to upload crop {crop_num}: {upload_response.text}")
                            continue
                        
                        # Create task data
                        task_data = {
                            'data': {'image': f'/data/upload/{self.project_id}/{crop_filename}'}
                        }
                        
                        # Create new task
                        create_response = self.client.make_request(
                            'POST',
                            f'/api/projects/{self.project_id}/tasks/',
                            json=task_data
                        )
                        
                        if create_response.status_code != 201:
                            print(f"Failed to create task for crop {crop_num}: {create_response.text}")
                            continue
                        
                        new_task = create_response.json()
                        new_task_id = new_task['id']
                        
                        # Add annotations to the new task
                        for annotation in region_annotations:
                            annotation_response = self.client.make_request(
                                'POST',
                                f'/api/tasks/{new_task_id}/annotations/',
                                json=annotation
                            )
                            
                            if annotation_response.status_code != 201:
                                print(f"Failed to add annotation to crop {crop_num}: {annotation_response.text}")
                        
                        created_tasks.append({
                            'crop_num': crop_num,
                            'task_id': new_task_id,
                            'filename': crop_filename,
                            'annotations_count': len(region_annotations)
                        })
                        
                        print(f"✓ Created crop {crop_num}/{v_blocks * h_blocks}: {crop_filename} "
                              f"(Task ID: {new_task_id}, Annotations: {len(region_annotations)})")
                        
                    except Exception as e:
                        print(f"Error creating task for crop {crop_num}: {str(e)}")
                        continue
            
            print(f"\nSuccessfully created {len(created_tasks)} crop tasks!")
            return created_tasks
            
        finally:
            # Clean up temp files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    def preview_and_confirm_split(self, task_id, v_blocks, h_blocks):
        """Show preview and ask for confirmation before splitting"""
        # Get task and download image for preview
        task = self.get_task_by_id(task_id)
        if not task:
            return False
        
        task_data = task.get('data', {})
        image_path_key = 'image' if 'image' in task_data else 'img'
        image_url = task_data.get(image_path_key)
        
        temp_image_path = self.download_image(image_url, task_id)
        if not temp_image_path:
            return False
        
        try:
            # Show preview
            fig = self.create_preview_grid(temp_image_path, v_blocks, h_blocks)
            
            # Add confirmation buttons
            ax_confirm = plt.axes([0.4, 0.02, 0.1, 0.05])
            ax_cancel = plt.axes([0.51, 0.02, 0.1, 0.05])
            
            btn_confirm = Button(ax_confirm, 'Confirm')
            btn_cancel = Button(ax_cancel, 'Cancel')
            
            confirmed = [False]
            cancelled = [False]
            
            def on_confirm(event):
                confirmed[0] = True
                plt.close(fig)
            
            def on_cancel(event):
                cancelled[0] = True
                plt.close(fig)
            
            btn_confirm.on_clicked(on_confirm)
            btn_cancel.on_clicked(on_cancel)
            
            plt.show()
            
            if confirmed[0]:
                print("Confirmed! Starting image split...")
                return self.split_image_and_create_tasks(task_id, v_blocks, h_blocks)
            else:
                print("Cancelled.")
                return False
                
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

def main():
    """Main function to run the image splitter"""
    print("Label Studio Image Crop Splitter")
    print("=================================")
    
    splitter = ImageCropSplitter(LS_URL, API_TOKEN, PROJECT_ID)
    
    try:
        # Get user input
        task_id = int(input("Enter the task ID to split: "))
        v_blocks = int(input("Enter number of vertical blocks: "))
        h_blocks = int(input("Enter number of horizontal blocks: "))
        
        print(f"\nTask ID: {task_id}")
        print(f"Grid: {v_blocks} vertical × {h_blocks} horizontal = {v_blocks * h_blocks} crops")
        
        # Preview and confirm
        result = splitter.preview_and_confirm_split(task_id, v_blocks, h_blocks)
        
        if result:
            print("\n" + "="*50)
            print("SPLIT SUMMARY")
            print("="*50)
            for crop_info in result:
                print(f"Crop {crop_info['crop_num']:3d}: {crop_info['filename']} "
                      f"(Task ID: {crop_info['task_id']}, "
                      f"Annotations: {crop_info['annotations_count']})")
            print(f"\nTotal crops created: {len(result)}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()