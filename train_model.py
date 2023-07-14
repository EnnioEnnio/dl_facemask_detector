"""
TODO: train_model.py
This file is responsible for training your model using the training data. It
typically includes the training loop, which iterates over the training dataset,
feeds the data through the model, computes the loss, performs backpropagation,
and updates the model parameters using an optimizer. It can also include code
for monitoring and logging training progress, saving checkpoints of the model,
and performing early stopping if necessary.
"""

from architecture import LeNetty
import torch
import random
import numpy as np
import torch
import os
import configparser
from PIL import Image
from tqdm import tqdm

# NOTE: I feel that this methods could be moved to data_loader.py for better code organization


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


def train_model(model,
                training_set=None,
                training_labels=None,
                epochs=200,
                batch_size=128,
                lr=0.1,
                loss=torch.nn.BCELoss(),
                optimizer=torch.optim.RMSprop):
    """INFO: training for 200 epochs with a batch_size of 128 will train on a total of 25,600 images"""
    if training_set is None or training_labels is None:
        print("[INFO] No training set and/or labels provided.")

    neural_net = model
    optimizer = optimizer(neural_net.parameters(), lr=lr)
    num_batches = len(training_set) // batch_size
    total_loss = 0.0

    # Training Loop TODO: implement early stopping and checkpointing (
    # don't know what the latter is but chatGPT suggested it ^^)
    for index_epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        # TODO: implement out of bounds check if epoch is larger than dataset
        start_index = index_epoch * batch_size
        end_index = start_index + index_epoch

        # Prepare mini-batch data and labels
        mini_batch_data = training_set[start_index:end_index]
        mini_batch_labels = training_labels[start_index:end_index]

        # Convert mini-batch data and labels to PyTorch tensors
        mini_batch_data = torch.tensor(mini_batch_data)
        mini_batch_labels = torch.tensor(mini_batch_labels)

        batch_loss = 0.0
        # Batch Loop
        for image_index in tqdm(range(batch_size), desc="Batch Progress", unit="image", leave=False):
            # Forward pass for each image in the mini-batch
            image_data = mini_batch_data[image_index]
            image_label = mini_batch_labels[image_index]

            # Forward pass for the current image
            output = neural_net(image_data)

            # Calculate loss for the current image
            image_loss = loss(output, image_label)

            # Accumulate loss for the mini-batch
            batch_loss += image_loss

            # Backward pass and parameter update for the current image
            optimizer.zero_grad()
            image_loss.backward()
            optimizer.step()

        # Print average batch loss
        batch_loss /= batch_size
        tqdm.write(f"Epoch {index_epoch+1}/{epochs}, Batch Loss: {batch_loss}")

        # Accumulate loss for the entire training set
        total_loss += batch_loss

    # Print total loss
    total_loss /= num_batches
    tqdm.write(f"Total Loss: {total_loss}")

    # Return the trained model
    return neural_net


if __name__ == "__main__":
    model = LeNetty()
    (training_set, training_labels, validation_set, validation_labels) = make_training_and_validation_set(
        dataset_size=25600, validation_split=0.2)
    model.train(True)
    train_model(model, training_set=training_set,
                training_labels=training_labels)
