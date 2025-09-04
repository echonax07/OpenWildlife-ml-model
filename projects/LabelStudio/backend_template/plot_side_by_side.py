import cv2
import os
import numpy as np
# Paths to your folders
folder_gt = "/home/pc2041/VIP_lab/labelstudio/mmwhale2/AES fully Manual"
folder_hitl = "/home/pc2041/VIP_lab/labelstudio/mmwhale2/AES HITL part2"
output_folder = "/home/pc2041/VIP_lab/labelstudio/mmwhale2/10_Images_ManualvsHITL"

os.makedirs(output_folder, exist_ok=True)

# Get sorted list of images
images_gt = sorted([f for f in os.listdir(folder_gt) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
images_hitl = sorted([f for f in os.listdir(folder_hitl) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for gt_img_name, hitl_img_name in zip(images_gt, images_hitl):
    gt_path = os.path.join(folder_gt, gt_img_name)
    hitl_path = os.path.join(folder_hitl, hitl_img_name)

    gt_img = cv2.imread(gt_path)
    hitl_img = cv2.imread(hitl_path)

    # Resize images to same height
    if gt_img.shape[0] != hitl_img.shape[0]:
        height = min(gt_img.shape[0], hitl_img.shape[0])
        gt_img = cv2.resize(gt_img, (int(gt_img.shape[1] * height / gt_img.shape[0]), height))
        hitl_img = cv2.resize(hitl_img, (int(hitl_img.shape[1] * height / hitl_img.shape[0]), height))

    # Combine images side by side
    combined = cv2.hconcat([gt_img, hitl_img])

    # Create a white canvas on top for the title
    title_height = 100
    title_canvas = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255

    # Add labels to the canvas
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # Bigger font
    color = (0, 0, 0)  # Black
    thickness = 3

    # Position text roughly at the center of each image
    gt_x = gt_img.shape[1] // 2 - 50
    hitl_x = gt_img.shape[1] + hitl_img.shape[1] // 2 - 70
    y_pos = title_height // 2 + 20

    cv2.putText(title_canvas, 'GT', (gt_x, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(title_canvas, 'HITL', (hitl_x, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    # Stack the title canvas above the combined image
    final_img = cv2.vconcat([title_canvas, combined])

    # Save the result
    output_path = os.path.join(output_folder, gt_img_name)
    cv2.imwrite(output_path, final_img)

print("Done! Combined images with titles saved in", output_folder)
