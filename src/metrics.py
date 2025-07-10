"""
Module Name: metrics.py
Description: Computes Intersection over Union (IoU) between predicted and ground truth masks.
Author: Adwita Jain
Date: 2025-10-10
"""

import numpy as np

def iou(pred_mask, true_mask):
    # Calculate the intersection: pixels where both masks are True
    intersection = np.logical_and(pred_mask, true_mask).sum()
    # Calculate the union: pixels where either mask is True
    union = np.logical_or(pred_mask, true_mask).sum()

    # Handle the case where both masks are empty (no positive pixels)
    if union == 0:
        print("Empty union — likely both masks are blank!")
        return 0.0

    # Return the IoU score (intersection over union)
    return intersection / union


