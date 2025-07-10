"""
Module Name: model.py
Description: Defines the model architecture for the segmentation task.
Author: Adwita Jain
Date: 2025-10-10
"""

from tensorflow.keras.applications import ResNet50

def get_resnet_model():
    # Load the ResNet50 model with pre-trained ImageNet weights and include the top classification layer
    return ResNet50(weights='imagenet', include_top=True)
