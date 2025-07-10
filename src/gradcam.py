"""
Module Name: gradcam.py
Description: Generates Grad-CAM heatmaps for visualizing model predictions.
Author: Adwita Jain
Date: 2025-07-07
"""

import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model


def generate_gradcam(model, img_array, last_conv_layer_name="conv5_block3_out", classifier_layer_name="predictions", pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer and the output
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # If no prediction index is provided, use the class with highest score
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # Get the score for the target class
        class_output = predictions[:, pred_index]

    # Compute gradients of the class output value with respect to the feature map
    grads = tape.gradient(class_output, conv_outputs)[0]
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    # Weight the feature maps by the pooled gradients
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def binarize_heatmap(heatmap, threshold=0.5):
    # Resize heatmap to match input image size (224x224)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    # Binarize the heatmap using the given threshold
    return (heatmap_resized > threshold).astype(np.uint8)
