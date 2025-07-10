"""

Module Name: dataloader.py
Description: Loads images and masks for the Grad-CAM model.
Author: Adwita Jain
Date: 2025-10-10
"""


import os
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2


IMG_WIDTH, IMG_HEIGHT = 224, 224  # Set image dimensions for model input

def load_image(img_path):
    # Load image and resize to model input size
    img = keras_image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    # Convert image to numpy array
    img_array = keras_image.img_to_array(img)
    # Preprocess for ResNet50 and add batch dimension, also return original array
    return preprocess_input(np.expand_dims(img_array, axis=0)), img_array

def load_mask(mask_path):
    # Load mask image as grayscale and resize
    mask = keras_image.load_img(mask_path, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
    # Convert mask to numpy array and remove single channel dimension
    mask_array = keras_image.img_to_array(mask).squeeze()

    # Normalize and binarize: convert all nonzero values to 1
    binary_mask = (mask_array >= 1).astype(np.uint8)

    return binary_mask

