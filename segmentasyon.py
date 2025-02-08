#!pip install transformers
#!pip install torch
#
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

#image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open("WIN_20240414_13_49_48_Pro.jpg")
# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# logits = outputs.logits.cpu()

# upsampled_logits = nn.functional.interpolate(
#     logits,
#     size=image.size[::-1],
#     mode="bilinear",
#     align_corners=False,
# )

# pred_seg = upsampled_logits.argmax(dim=1)[0]
# plt.imshow(pred_seg)

import os
# Load model and processor
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Define the base directory where the images are stored
base_directory = "C:\\Users\\melis\\Desktop\\staj\\omer-melisa staj\\static"

# Iterate through folders
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)
    
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        
        # Iterate through images in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            # Open and process the image
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            
            
            # Perform model inference
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Upsample logits and get predictions
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            
            # Convert to numpy and plot
            pred_seg = pred_seg.cpu().numpy()
            
            # Plot and show the segmentation result
            plt.figure(figsize=(10, 10))
            plt.imshow(pred_seg, cmap="viridis")
            plt.title(f"Segmentation for {folder_name}/{image_name}")
            plt.axis('off')  # Hide axis
            plt.show()