"""
TODO: data_loader.py

This file is typically responsible for loading and preprocessing your data.
It can handle tasks such as reading data from files, 
performing data augmentation, data normalization, 
and creating data loaders for efficient batching of the data during training and evaluation.

Obserservations:
Images from the dataset are of different sizes.
"""

import os
from PIL import Image


def resize_image(image_dir, output_dir, target_size=(256, 256)):
    """
    Resize images to target size and save them to output directory.
    Currently only supports .jpg and .png images.
    It's worth noting that resize will stretch and sometimes rotate Images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path)

            resized_image = image.resize(target_size)

            output_path = os.path.join(output_dir, image_name)
            resized_image.save(output_path)
