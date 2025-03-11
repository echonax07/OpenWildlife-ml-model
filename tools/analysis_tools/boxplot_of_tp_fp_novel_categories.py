import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory containing all the dataset folders
root_dir = '/home/m32patel/projects/def-dclausi/whale/mmwhale2/tp_fp_comparision'

# Initialize lists to store all TP and FP scores
all_tp_scores = []
all_fp_scores = []

# Traverse through each dataset folder
for dataset_folder in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_folder)
    
    # Check if it's a directory and contains the required CSV file
    if os.path.isdir(dataset_path):
        csv_file_path = os.path.join(dataset_path, 'tp_fp_confidence_scores.csv')
        
        if os.path.isfile(csv_file_path):
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Append TP and FP scores to the respective lists
            all_tp_scores.extend(df['score_of_tp_without_class_confusion'].tolist())
            all_fp_scores.extend(df['score_of_fp_without_class_confusion'].tolist())

# Create the box plot
plt.figure(figsize=(2, 3))  # Adjust width to fit two plots
plt.boxplot([all_tp_scores, all_fp_scores], vert=True, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            positions=[1, 2])  # Two positions for TP and FP

# Customize the plot with larger fonts
plt.title("")  # Remove the title to save space
plt.xticks([1, 2], ["TP", "FP"], fontsize=14)  # Increased font size for x-axis labels
plt.yticks(np.linspace(0, 1, 6), fontsize=14)  # Increased font size for y-axis ticks
plt.ylim(0, 1)  # Ensure y-axis range is fixed from 0 to 1
plt.ylabel("Score", fontsize=12)  # Larger font size for y-axis label
plt.tight_layout()

# Show the plot
plt.savefig('tp_fp_novel_class')