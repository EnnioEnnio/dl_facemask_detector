"""
This file is responsible for training your model using the training data. It
typically includes the training loop, which iterates over the training dataset,
feeds the data through the model, computes the loss, performs backpropagation,
and updates the model parameters using an optimizer. It can also include code
for monitoring and logging training progress, saving checkpoints of the model,
and performing early stopping if necessary.
"""

from architecture import LeNetty
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import logging as log
import numpy as np
import os
import torch
import wandb

log.basicConfig(level=log.INFO, format="[%(levelname)s] %(message)s")


def train_model(
    model,
    training_set,
    epochs=200,
    batch_size=128,
    learning_rate=0.1,
    early_stopping_patience=10,
    checkpointing=False,
    loss_function=torch.nn.BCELoss,
    optimizer=torch.optim.RMSprop,
):
    log.info(f"Number of training samples: {len(training_set)}")
    log.info(f"Detected classes: {training_set.class_to_idx}")

    # init dataloader from training dataset
    dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # init model, optimizer and loss function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using computation device: {device}\n")
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

    # init counters for early stopping & logging
    best_loss = np.inf
    epochs_without_improvement = 0
    total_loss = 0.0
    num_batches = len(dataloader)

    # main training Loop
    epochs = tqdm(
        range(epochs),
        desc="Training Progress",
        unit="epochs",
        total=epochs,
    )
    for epoch in epochs:
        for batch, (input, label) in tqdm(
            enumerate(dataloader),
            desc="Current epoch progress",
            unit="batches",
            leave=False,
            total=num_batches,
        ):
            batch_loss = 0.0
            input = input.to(device)
            label = label.to(device).float()
            output = neural_net(input)
            output = output.reshape(batch_size)

            log.debug(f"label: {label}")
            log.debug(f"output: {output}")

            image_loss = loss(output, label)
            optimizer.zero_grad()
            image_loss.backward()
            optimizer.step()

            batch_loss += (image_loss / batch_size).item()
            wandb.log({"batch_loss": batch_loss})
            tqdm.write(
                f"Epoch {epoch+1}/{len(epochs)}, Batch: {batch}, Batch Loss: {batch_loss}"
            )

            # Early Stopping (and Checkpointing)
            if batch_loss < best_loss:
                best_loss = batch_loss
                epochs_without_improvement = 0
                if checkpointing:
                    tqdm.write(f"Epoch {epoch+1}/{epochs}, saving checkpoint")
                    torch.save(
                        trained_model.state_dict(),
                        f"${model.__class__.__name__}-checkpoint-{epoch}.pt",
                    )
            else:
                epochs_without_improvement += 1
            total_loss += batch_loss

        if epochs_without_improvement > early_stopping_patience:
            tqdm.write(f"Early stopping after {epoch+1} epochs without improvement")
            break

    # Accumulate loss for the entire training set
    # Print and log total loss
    total_loss /= num_batches
    wandb.log({"total_loss": total_loss})
    tqdm.write(f"total Loss: {total_loss}")
    # Close weights and biases logging (needed for Jupyter Notebooks)
    wandb.finish()
    # Return the trained model
    return neural_net


if __name__ == "__main__":
    model = LeNetty()

    dataset_path = os.getenv("DATASET_PATH") or "./dataset"
    dataset_path = os.path.abspath(dataset_path)
    log.debug(f"Dataset path: {dataset_path}")

    img_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(
        root=dataset_path,
        transform=img_transform,
    )

    validation_split = 0.2
    log.debug(f"Validation split: {validation_split}")

    training_dataset, validation_dataset = random_split(
        dataset, [1 - validation_split, validation_split]
    )

    log.debug(f"Training dataset: {len(training_dataset)}")
    log.debug(f"Validation dataset: {len(validation_dataset)}")

    model.train(True)
    trained_model = train_model(model, dataset)
    torch.save(trained_model.state_dict(), f"{model.__class__.__name__}-trained.pt")
