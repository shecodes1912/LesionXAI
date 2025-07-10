"""
Module Name: main.py
Description: Main pipeline to run the Grad-CAM model on apple leaf images and compare with ground truth masks.
import os
Author: Adwita Jain
Date: 2025-07-07
"""


import os
import pandas as pd
from dataloader import load_image, load_mask
from model import get_resnet_model
from gradcam import generate_gradcam, binarize_heatmap
from metrics import iou
from visualisation import save_visualization
import numpy as np


def run_pipeline(image_dir, mask_dir, threshold=0.1):
    # Load the pre-trained model
    model = get_resnet_model()
    results = []  # List to store IoU results for each image

    # Iterate through all files in the image directory
    for fname in os.listdir(image_dir):
        # Skip files that are not images
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]  # removes .jpg/.png
        mask_jpg = os.path.join(mask_dir, f"{base_name}_mask.jpg")
        mask_png = os.path.join(mask_dir, f"{base_name}_mask.png")

        # Check if either mask file exists
        mask_path = mask_jpg if os.path.exists(mask_jpg) else (mask_png if os.path.exists(mask_png) else None)

        # Skip if no mask is found
        if mask_path is None:
            print(f" Missing mask for {fname}, skipping.")
            continue

        # Load and preprocess the image and mask
        img_pre, img_orig = load_image(img_path)
        mask = load_mask(mask_path)

        # Generate Grad-CAM heatmap and binarize it
        heatmap, pred_index = generate_gradcam(model, img_pre)
        bin_mask = binarize_heatmap(heatmap, threshold)

        # Compute IoU score
        score = iou(bin_mask, mask)
        results.append({"filename": fname, "iou": score})

        # Save visualization for this image
        save_visualization(img_orig, mask, heatmap, bin_mask, score, fname.split(".")[0], "results/visualizations")

    # Save all IoU results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv("results/metrics.csv", index=False)
    print("Pipeline complete. Results saved to 'results/'.")

if __name__ == "__main__":
    # Run the pipeline with specified directories and threshold
    run_pipeline("dataset/images", "dataset/masks", threshold=0.1)

