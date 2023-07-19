"""
This file is responsible for training your model using the training data. It
typically includes the training loop, which iterates over the training dataset,
feeds the data through the model, computes the loss, performs backpropagation,
and updates the model parameters using an optimizer. It can also include code
for monitoring and logging training progress, saving checkpoints of the model,
and performing early stopping if necessary.
"""

from architecture import LeNetty
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import configparser
import logging as log
import numpy as np
import os
import torch
import wandb

log.basicConfig(level=log.INFO, format="[%(levelname)s] %(message)s")


def train_model(
    model,
    data_set,
    epochs=200,
    batch_size=128,
    learning_rate=0.1,
    validation_split=0.2,
    early_stopping_patience=10,
    checkpointing=False,
    loss_function=torch.nn.BCELoss,
    optimizer=torch.optim.RMSprop,
):
    log.info(f"Validation split: {validation_split}")

    training_set, validation_set = random_split(
        data_set, [1 - validation_split, validation_split]
    )

    log.info(f"Training dataset: {len(training_set)} samples")
    log.info(f"Validation dataset: {len(validation_set)} samples")
    log.info(f"Detected classes: {data_set.class_to_idx}")

    # creates a balanced sampler for subsets of a torch Dataset object (ex ImageFolder)
    def balanced_sampler(set):
        indices = set.indices
        class_labels = [dataset.targets[i] for i in indices]
        class_sample_count = np.bincount(class_labels)
        class_weights = 1.0 / class_sample_count
        sample_weights = np.array([class_weights[t] for t in class_labels])
        return WeightedRandomSampler(
            torch.from_numpy(sample_weights), len(sample_weights)
        )

    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=balanced_sampler(training_set),
    )

    validation_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=balanced_sampler(validation_set),
    )

    # init model, cuda, optimizer, and loss function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info(
            f"Using computation device: {torch.cuda.get_device_name()} * {torch.cuda.device_count()}"
        )
    else:
        log.info("Using computation device: cpu")

    log.debug(f"Model: {model}")
    neural_net = model.to(device)
    optimizer = optimizer(neural_net.parameters(), lr=learning_rate)
    loss = loss_function()

    # initialize weights and biases logging (wandb)
    wandb.init(
        project="dl_facemask_detection",
        config={
            "architecture": model.__class__.__name__,
            "dataset": "Real-World-Masked-Face-Dataset (RMFD)",
            "learning_rate": learning_rate,
            "epoch": epochs,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer.__class__.__name__,
        },
    )

    # init counters for early stopping, logging, etc.
    best_loss = np.inf
    epochs_without_improvement = 0
    num_epochs = epochs
    num_training_batches = len(training_loader)
    num_validation_batches = len(validation_loader)

    # main training Loop
    epochs = tqdm(range(epochs), desc="Epoch progress", unit="epochs", total=epochs)
    for epoch in epochs:
        epoch_idx = epoch + 1
        epoch_loss = 0.0
        training_batches = tqdm(
            enumerate(training_loader),
            desc=f"Epoch {epoch_idx}: Training progress",
            unit="batches",
            leave=False,
            total=num_training_batches,
        )
        for batch, (input, label) in training_batches:
            input = input.to(device)
            label = label.to(device).float()
            output = neural_net(input).reshape(batch_size)

            # backprop
            current_batch_loss = loss(output, label)
            optimizer.zero_grad()
            current_batch_loss.backward()
            optimizer.step()

            # accumulate loss for the current epoch
            epoch_loss += (current_batch_loss / num_training_batches).item()

            wandb.log({"batch_loss": current_batch_loss.item()})
            tqdm.write(
                f"Epoch {epoch_idx:03}/{num_epochs:03},"
                f"Batch: {batch:4},"
                f"Batch loss: {current_batch_loss:12f},"
                f"Running epoch loss: {epoch_loss:12f},"
                f"Split: {np.bincount(Tensor.cpu(label)) / batch_size}"
            )

        # validation testing
        tqdm.write(f"Epoch {epoch_idx:03}/{num_epochs:03}, performing validation")
        validation_loss = 0.0
        validation_batches = tqdm(
            enumerate(validation_loader),
            desc=f"Epoch {epoch_idx}: Validation progress",
            unit="batches",
            leave=False,
            total=num_validation_batches,
        )
        for batch, (input, label) in validation_batches:
            input = input.to(device)
            label = label.to(device).float()
            output = neural_net(input).reshape(batch_size)
            current_batch_loss = loss(output, label)
            validation_loss += (current_batch_loss / num_validation_batches).item()
        tqdm.write(
            f"Epoch {epoch_idx:03}/{num_epochs:03}, Validation loss: {validation_loss:12f}"
        )
        wandb.log({"validation_loss": validation_loss})

        # end of current epoch, log current loss and perform checkpointing / early stopping if needed
        wandb.log({"loss": epoch_loss})
        tqdm.write(f"Epoch {epoch_idx:03}/{num_epochs:03}, Loss: {epoch_loss:12f}")

        # Early Stopping (and Checkpointing)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            if checkpointing:
                tqdm.write(f"Epoch {epoch_idx:03}/{num_epochs:03}, saving checkpoint")
                torch.save(
                    trained_model.state_dict(),
                    f"${model.__class__.__name__}-checkpoint-{epoch_idx}.pt",
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > early_stopping_patience:
            tqdm.write(f"Early stopping after {epoch_idx} epochs without improvement")
            break

    # Close weights and biases logging (needed for Jupyter Notebooks)
    wandb.finish()
    # Return the trained model
    return neural_net


if __name__ == "__main__":
    model = LeNetty()

    config = configparser.ConfigParser()
    config.read("config.ini")
    config_value = config.get("Paths", "dataset", fallback=None)
    dataset_path = os.getenv("DATASET_PATH") or config_value or "./dataset"

    dataset_path = os.path.abspath(dataset_path)
    log.info(f"Dataset path: {dataset_path}")

    img_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            # transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(
        root=dataset_path,
        transform=img_transform,
    )

    model.train(True)
    trained_model = train_model(model, dataset)
    torch.save(trained_model.state_dict(), f"{model.__class__.__name__}-trained.pt")
