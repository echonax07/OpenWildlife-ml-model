# Label Studio Image Crop Splitter - Jupyter Notebook Version (Optimized with Local Path Option)
# Run each cell sequentially

# Cell 1: Import libraries and setup
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
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
import base64
from icecream import ic
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_data_dir, get_local_path
# Enable inline plotting
# %matplotlib inline

# Configuration
LS_URL = 'http://129.97.250.147:8080/'
API_TOKEN = 'ebdc6fa5f2c3abcd502b55d5ccc1dc0e4ae9f68d'
PROJECT_ID = 108

print("✓ Libraries imported and configuration set")

# Cell 2: Initialize the optimized ImageCropSplitter class
class ImageCropSplitter:
    def __init__(self, ls_url, api_token, project_id):
        self.client = Client(url=ls_url, api_key=api_token)
        self.project_id = project_id
        self.project = self.client.get_project(project_id)
        print(f"✓ Connected to Label Studio project {project_id}")
        
    def get_task_by_id(self, task_id):
        """Get a specific task by ID"""
        try:
            task = self.project.get_task(task_id)
            return task
        except Exception as e:
            print(f"❌ Error fetching task {task_id}: {str(e)}")
            return None
    
    def download_image(self, image_url, task_id):
        """Download image from Label Studio"""
        try:
            if image_url.startswith('/'):
                full_url = f"{LS_URL.rstrip('/')}{image_url}"
            else:
                full_url = image_url
            
            headers = {'Authorization': f'Token {API_TOKEN}'}
            response = requests.get(full_url, headers=headers)
            response.raise_for_status()
            
            temp_path = f"temp_image_{task_id}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            return temp_path
        except Exception as e:
            print(f"❌ Error downloading image: {str(e)}")
            return None
    
    def get_annotations_in_region(self, annotations, x_start, y_start, x_end, y_end, img_width, img_height):
        """Get annotations that fall within a specific region"""
        region_annotations = []
        
        for annotation in annotations:
            for result in annotation.get('result', []):
                if result.get('type') == 'keypointlabels':
                    value = result.get('value', {})
                    
                    x_percent = value.get('x', 0)
                    y_percent = value.get('y', 0)
                    
                    x_pixel = (x_percent / 100) * img_width
                    y_pixel = (y_percent / 100) * img_height
                    
                    if x_start <= x_pixel <= x_end and y_start <= y_pixel <= y_end:
                        new_result = result.copy()
                        
                        crop_width = x_end - x_start
                        crop_height = y_end - y_start
                        
                        new_x_percent = ((x_pixel - x_start) / crop_width) * 100
                        new_y_percent = ((y_pixel - y_start) / crop_height) * 100
                        
                        new_result['value']['x'] = new_x_percent
                        new_result['value']['y'] = new_y_percent
                        new_result['to_name'] = 'image'
                        
                        region_annotations.append(new_result)
        
        return region_annotations
    
    def create_preview_grid(self, image_path, v_blocks, h_blocks):
        """Create a preview of how the image will be split"""
        ic(image_path)
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(img)
        ax.set_title(f'Image Split Preview: {v_blocks}×{h_blocks} grid ({v_blocks * h_blocks} crops)', 
                    fontsize=16, fontweight='bold')
        
        # Calculate block dimensions
        block_width = img_width / h_blocks
        block_height = img_height / v_blocks
        
        # Draw grid lines
        for i in range(1, h_blocks):
            x = i * block_width
            ax.axvline(x=x, color='red', linewidth=3, alpha=0.8)
        
        for i in range(1, v_blocks):
            y = i * block_height
            ax.axhline(y=y, color='red', linewidth=3, alpha=0.8)
        
        # Add crop numbers
        crop_num = 0
        for v in range(v_blocks):
            for h in range(h_blocks):
                crop_num += 1
                center_x = (h + 0.5) * block_width
                center_y = (v + 0.5) * block_height
                
                ax.text(center_x, center_y, str(crop_num), 
                       ha='center', va='center', fontsize=20, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9),
                       fontweight='bold')
        
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        ax.axis('on')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return img_width, img_height
    
    def split_image_and_create_tasks(self, task_id, v_blocks, h_blocks, upload_crops=True):
        """Split image into crops and create new tasks with optimized API calls"""
        original_task = self.get_task_by_id(task_id)
        if not original_task:
            return False
        
        task_data = original_task.get('data', {})
        image_path_key = 'image' if 'image' in task_data else 'img'
        image_url = task_data.get(image_path_key)
        
        if not image_url:
            print("❌ No image found in task data")
            return False
        
        temp_image_path = self.download_image(image_url, task_id)
        ic(temp_image_path)
        if not temp_image_path:
            return False
        
        try:
            img = Image.open(temp_image_path)
            img_width, img_height = img.size
            
            original_filename = get_local_path(image_url, hostname=LS_URL, access_token=API_TOKEN)
            ic(original_filename)
            base_name = Path(original_filename).stem
            extension = Path(original_filename).suffix
            
            block_width = img_width // h_blocks
            block_height = img_height // v_blocks
            
            original_annotations = original_task.get('annotations', [])
            
            created_tasks = []
            crop_num = 0
            total_crops = v_blocks * h_blocks
            
            print(f"🚀 Creating {total_crops} crops...")
            
            # Create progress bar
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=total_crops,
                description='Progress:',
                bar_style='info',
                style={'bar_color': 'blue'},
                orientation='horizontal'
            )
            display(progress)
            
            for v in range(v_blocks):
                for h in range(h_blocks):
                    crop_num += 1
                    progress.value = crop_num
                    
                    x_start = h * block_width
                    y_start = v * block_height
                    x_end = min((h + 1) * block_width, img_width)
                    y_end = min((v + 1) * block_height, img_height)
                    
                    crop = img.crop((x_start, y_start, x_end, y_end))
                    
                    crop_filename = f"{base_name}_crop_row{v:02d}_col{h:02d}{extension}"
                    crop_path = f"crops/{crop_filename}"
                    os.makedirs("crops", exist_ok=True)
                    ic(crop_path)
                    crop.save(crop_path)
                    
                    # Get annotations for this region
                    region_annotations = self.get_annotations_in_region(
                        original_annotations, x_start, y_start, x_end, y_end, 
                        img_width, img_height
                    )
                    
                    try:
                        if upload_crops:
                            # Upload the image file
                            with open(crop_path, 'rb') as f:
                                upload_response = self.client.make_request(
                                    'POST',
                                    '/api/projects/{}/import'.format(self.project_id),
                                    files={'file': (crop_filename, f, 'image/jpeg')}
                                )
                            
                            if upload_response.status_code != 201:
                                print(f"❌ Failed to upload crop {crop_num}")
                                continue
                            
                            image_path = f'/data/upload/{self.project_id}/{crop_filename}'
                        else:
                            # Use local path
                            image_path = f'/data/local-files/?d=crops/{crop_filename}'
                        
                        # Create task with annotations in single request
                        task_payload = {
                            'data': {'image': image_path}
                        }
                        
                        # Add annotations to the task payload if any exist
                        if region_annotations:
                            task_payload['annotations'] = [{
                                'result': region_annotations,
                                'was_cancelled': False,
                                'ground_truth': False,
                                'created_username': original_annotations[0].get('created_username', '') if original_annotations else '',
                                'updated_username': original_annotations[0].get('updated_username', '') if original_annotations else '',
                            }]
                        
                        # Single API call to create task with annotations
                        create_response = self.client.make_request(
                            'POST',
                            f'/api/projects/{self.project_id}/tasks/',
                            json=task_payload
                        )
                        
                        if create_response.status_code != 201:
                            print(f"❌ Failed to create task for crop {crop_num}: {create_response.text}")
                            continue
                        
                        new_task = create_response.json()
                        new_task_id = new_task['id']
                        
                        created_tasks.append({
                            'crop_num': crop_num,
                            'task_id': new_task_id,
                            'filename': crop_filename,
                            'annotations_count': len(region_annotations)
                        })
                        
                        if crop_num % 10 == 0 or crop_num == total_crops:
                            print(f"✅ Processed {crop_num}/{total_crops} crops")
                        
                    except Exception as e:
                        print(f"❌ Error creating task for crop {crop_num}: {str(e)}")
                        continue
            
            progress.bar_style = 'success'
            print(f"\n🎉 Successfully created {len(created_tasks)} crop tasks!")
            print(f"📊 API calls reduced from {len(created_tasks) * 3} to {len(created_tasks) * (2 if upload_crops else 1)}")
            return created_tasks
            
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

# Initialize the splitter
splitter = ImageCropSplitter(LS_URL, API_TOKEN, PROJECT_ID)

# Cell 3: Create interactive input widgets
print("📋 Enter Split Parameters:")

task_id_widget = widgets.IntText(
    value=1,
    description='Task ID:',
    style={'description_width': 'initial'}
)

v_blocks_widget = widgets.IntText(
    value=2,
    description='Vertical Blocks:',
    style={'description_width': 'initial'}
)

h_blocks_widget = widgets.IntText(
    value=4,
    description='Horizontal Blocks:',
    style={'description_width': 'initial'}
)

upload_crops_widget = widgets.Checkbox(
    value=True,
    description='Upload crops to Label Studio (uncheck to use local paths)',
    style={'description_width': 'initial'}
)

preview_button = widgets.Button(
    description='Preview Split',
    button_style='info',
    icon='eye'
)

confirm_button = widgets.Button(
    description='Confirm & Process',
    button_style='success',
    icon='check'
)

output_area = widgets.Output()

# Display widgets
display(widgets.VBox([
    widgets.HBox([task_id_widget, v_blocks_widget, h_blocks_widget]),
    upload_crops_widget,
    widgets.HBox([preview_button, confirm_button]),
    output_area
]))

# Cell 4: Define widget event handlers
def on_preview_clicked(b):
    with output_area:
        clear_output()
        task_id = task_id_widget.value
        v_blocks = v_blocks_widget.value
        h_blocks = h_blocks_widget.value
        
        print(f"🔍 Previewing split for Task ID: {task_id}")
        print(f"📐 Grid: {v_blocks} vertical × {h_blocks} horizontal = {v_blocks * h_blocks} crops")
        
        task = splitter.get_task_by_id(task_id)
        if not task:
            return
        
        task_data = task.get('data', {})
        image_path_key = 'image' if 'image' in task_data else 'img'
        image_url = task_data.get(image_path_key)
        
        temp_image_path = splitter.download_image(image_url, task_id)
        if temp_image_path:
            try:
                img_width, img_height = splitter.create_preview_grid(temp_image_path, v_blocks, h_blocks)
                print(f"📊 Original image size: {img_width} × {img_height} pixels")
                print(f"📦 Each crop will be approximately: {img_width//h_blocks} × {img_height//v_blocks} pixels")
                print(f"✅ Preview complete! Click 'Confirm & Process' to create the crops.")
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

def on_confirm_clicked(b):
    with output_area:
        clear_output()
        task_id = task_id_widget.value
        v_blocks = v_blocks_widget.value
        h_blocks = h_blocks_widget.value
        upload_crops = upload_crops_widget.value
        
        print(f"🚀 Starting optimized image split process...")
        print(f"📋 Task ID: {task_id}")
        print(f"📐 Grid: {v_blocks} × {h_blocks} = {v_blocks * h_blocks} crops")
        print(f"📤 Upload crops: {upload_crops}")
        print(f"⚡ Using optimized API calls ({2 if upload_crops else 1} request{'s' if upload_crops else ''} per crop)")
        print("-" * 50)
        
        result = splitter.split_image_and_create_tasks(task_id, v_blocks, h_blocks, upload_crops)
        
        if result:
            print("\n" + "="*60)
            print("📈 SPLIT SUMMARY")
            print("="*60)
            
            # Group results by annotation count for summary
            annotation_counts = {}
            for crop_info in result:
                count = crop_info['annotations_count']
                if count not in annotation_counts:
                    annotation_counts[count] = 0
                annotation_counts[count] += 1
            
            print(f"✅ Total crops created: {len(result)}")
            print(f"📊 Annotation distribution:")
            for count, num_crops in sorted(annotation_counts.items()):
                print(f"   • {num_crops} crops with {count} annotations each")
            
            print(f"\n📁 Crop files saved in: ./crops/")
            print(f"🎯 All tasks created in Label Studio project {PROJECT_ID}")
            if not upload_crops:
                print(f"ℹ️ Using local paths: /data/local-files/?d=crops/")
            print("🎉 Process completed successfully!")

# Attach event handlers
preview_button.on_click(on_preview_clicked)
confirm_button.on_click(on_confirm_clicked)

print("🎮 Interactive controls ready! Use the widgets above to split your image.")
print("⚡ This optimized version reduces API calls by up to 66% when using local paths!")