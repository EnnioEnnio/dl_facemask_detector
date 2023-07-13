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

# NOTE: I feel that this methods could be moved to data_loader.py for better code organization


def get_files_from_subfolders(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths)


def make_training_set(batch_size):
    training_set = []
    labels = []
    # Using Config for better control over paths on different machines
    # see example_config.ini and rename file to labels.ini for this method to work
    config = configparser.ConfigParser()
    config.read("labels.ini")

    unmasked_images_folder = config.get("Paths", "umasked_images_folder")
    masked_images_folder = config.get("Paths", "masked_images_folder")

    unmasked_image_files = get_files_from_subfolders(unmasked_images_folder)
    masked_image_files = get_files_from_subfolders(masked_images_folder)

    for i in range(batch_size):
        if random.random() < 0.5:
            image_path = random.choice(unmasked_image_files)
            label = 0
        else:
            image_path = random.choice(masked_image_files)
            label = 1

        image = Image.open(image_path)
        training_set.append(image)
        labels.append(label)

    training_set = np.array(training_set)
    labels = np.array(labels)

    return training_set, labels


def train_model(model, epochs=200, batch_size=128, lr=0.1, loss=torch.nn.BCELoss(), optimizer=None,):
    neural_net = model
    optimizer = (
        torch.optim.RMSprop(neural_net.parameters(), lr=lr)
        if optimizer is None
        else optimizer(neural_net.parameters(), lr=lr)
    )
    total_loss = 0.0
    # NOTE: I'm not sure if this is the correct way to make a training set since it only happenes once
    # I think this should happen in the training loop for Stochastic Gradient Descent. Not sure though.
    training_loader = make_training_set(batch_size)

    # TODO: implement early stopping and checkpointing and monitoring using tqdm
    for epoch in range(epochs):
        print(f"[INFO] running epoch {epoch}")
        global inputs, labels
        epoch_loss = 0.0
        for i, mini_batch in enumerate(training_loader):
            # generate our mini batch
            inputs, labels = mini_batch
            optimizer.zero_grad()
            # forward pass
            outputs = neural_net(inputs)
            # compute loss for current batch and calculate gradients
            loss = loss(outputs, labels)
            print(f"[INFO] (epoch {epoch}) loss for batch {i}: {loss.item()}")
            loss.backward()
            # step optimizer, adjust weights
            optimizer.step()
            # compute epoch / total loss
            epoch_loss += loss.item()
        print(f"[INFO] (epoch {epoch}): total loss: {loss}")
        total_loss += epoch_loss


if __name__ == "__main__":
    model = LeNetty()
    model.train(True)
    # train_model(model)
