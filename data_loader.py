"""
TODO: data_loader.py

This file is typically responsible for loading and preprocessing your data.
It can handle tasks such as reading data from files, 
performing data augmentation, data normalization, 
and creating data loaders for efficient batching of the data during training and evaluation.
"""

import os
from PIL import Image
import random
import configparser
import numpy as np


def get_files_from_subfolders(folder_path: str):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths)


def extract_path_from_config(config_file: str):
    config = configparser.ConfigParser()
    config.read(config_file)

    unmasked_images_folder = config.get("Paths", "umasked_images_folder")
    masked_images_folder = config.get("Paths", "masked_images_folder")

    return unmasked_images_folder, masked_images_folder


def make_training_and_validation_set(dataset_size, validation_split):
    """
    Returns a training set and validation set with labels not taking 
    the distribution between masked and unmasked images into account.
    This is done in one method to ensure that the same images are not used in both sets
    """
    training_set = []
    validation_set = []
    training_labels = []
    validation_labels = []

    # Using Config for better control over paths on different machines
    # see example_config.ini and rename file to labels.ini for this method to work
    unmasked_images_folder, masked_images_folder = extract_path_from_config(
        "labels.ini")

    unmasked_image_files = get_files_from_subfolders(unmasked_images_folder)
    masked_image_files = get_files_from_subfolders(masked_images_folder)

    for i in range(dataset_size):
        if random.random() < 0.5:
            image_path = random.choice(unmasked_image_files)
            label = 0  # 0 for unmasked
        else:
            image_path = random.choice(masked_image_files)
            label = 1  # 1 for masked

        image = Image.open(image_path)

        if random.random() < validation_split:
            validation_set.append(image)
            validation_labels.append(label)
        else:
            training_set.append(image)
            training_labels.append(label)

    training_set = np.array(training_set)
    training_labels = np.array(training_labels)
    validation_set = np.array(validation_set)
    validation_labels = np.array(validation_labels)

    return training_set, training_labels, validation_set, validation_labels


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
