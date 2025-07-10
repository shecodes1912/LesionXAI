"""
Module Name: visualisation.py
Description: Visualizes the results of the Grad-CAM model.
Author: Adwita Jain
Date: 2025-07-07
"""
import matplotlib.pyplot as plt
import os

def save_visualization(img, true_mask, heatmap, binary_mask, iou_score, filename, output_dir):
    # Create a figure with 4 subplots in a row
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    # Show the original image
    axs[0].imshow(img.astype("uint8"))
    axs[0].set_title("Original")
    axs[0].axis('off')

    # Show the ground truth mask
    axs[1].imshow(true_mask, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    # Overlay the Grad-CAM heatmap on the original image
    axs[2].imshow(img.astype("uint8"))
    axs[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axs[2].set_title("Grad-CAM Overlay")
    axs[2].axis('off')

    # Show the binarized heatmap with IoU score in the title
    axs[3].imshow(binary_mask, cmap='gray')
    axs[3].set_title(f"Binarized Heatmap\nIoU: {iou_score:.2f}")
    axs[3].axis('off')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the figure to the specified directory with a custom filename
    plt.savefig(os.path.join(output_dir, f"{filename}_visual.png"))
    # Close the figure to free memory
    plt.close()
