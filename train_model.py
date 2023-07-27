"""
This file is responsible for training your model using the training data. It
typically includes the training loop, which iterates over the training dataset,
feeds the data through the model, computes the loss, performs backpropagation,
and updates the model parameters using an optimizer. It can also include code
for monitoring and logging training progress, saving checkpoints of the model,
and performing early stopping if necessary.
"""

from architecture import Model1
from torch import Tensor
import gc
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import logging as log
import numpy as np
import os
import torch
import wandb

from config import Config

log.basicConfig(level=log.INFO, format="[%(levelname)s] %(message)s")


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info(
            f"Using computation device: {torch.cuda.get_device_name()} * {torch.cuda.device_count()}"
        )
    else:
        log.info("Using computation device: cpu")
    return device


def make_balanced_dataloader(set, batch_size):
    # creates a balanced sampler for subsets of a torch Dataset object (ex ImageFolder)
    indices = set.indices
    class_labels = [dataset.targets[i] for i in indices]
    class_sample_count = np.bincount(class_labels)
    class_weights = 1.0 / class_sample_count
    sample_weights = np.array([class_weights[t] for t in class_labels])
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), len(sample_weights)
    )
    return DataLoader(
        set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )


def train_model(
    model,
    data_set,
    epochs=200,
    batch_size=128,
    learning_rate=0.01,
    momentum=0.9,
    validation_split=0.2,
    early_stopping_patience=10,
    checkpointing=False,
    loss_function=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.SGD,
    save_model=True,
):
    training_set, validation_set = random_split(
        data_set, [1 - validation_split, validation_split]
    )
    training_loader = make_balanced_dataloader(training_set, batch_size)
    validation_loader = make_balanced_dataloader(validation_set, batch_size)
    log.info(f"Validation split: {validation_split}")
    log.info(f"Training dataset: {len(training_set)} samples")
    log.info(f"Validation dataset: {len(validation_set)} samples")
    log.info(f"Detected classes: {data_set.class_to_idx}")

    # init model, cuda, optimizer, and loss function
    device = get_device()
    log.debug(f"Model: {model}")
    neural_net = model.to(device)
    optimizer = optimizer(
        neural_net.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
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
            total=len(training_loader),
        )
        for batch, (input, label) in training_batches:
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            input = input.to(device)
            label = label.to(device).float()
            output = neural_net(input).squeeze()

            # backprop
            batch_loss = loss(output, label)
            batch_loss.backward()
            optimizer.step()

            # accumulate loss for the current epoch
            epoch_loss += (batch_loss / len(training_loader)).item()

            wandb.log({"batch_loss": batch_loss.item()})
            tqdm.write(
                f"Epoch {epoch_idx:03}/{num_epochs:03},"
                f"Batch: {batch:4},"
                f"Batch loss: {batch_loss:12f},"
                f"Running epoch loss: {epoch_loss:12f},"
                f"Split: {np.bincount(Tensor.cpu(label)) / batch_size}"
            )

            del input, label, batch_loss
            gc.collect()

        # validation loop
        tqdm.write(f"Epoch {epoch+1:03}/{len(epochs):03}," f"Performing validation")
        validation_batches = tqdm(
            enumerate(validation_loader), total=len(validation_loader)
        )
        with torch.no_grad():
            validation_loss = 0.0
            num_correct = 0
            num_samples = len(validation_loader) * batch_size
            for batch, (images, labels) in validation_batches:
                labels = labels.to(device).float()
                outputs = model(images.to(device)).squeeze()
                batch_vloss = loss(outputs, labels)
                validation_loss += (batch_vloss / len(validation_loader)).item()

                predictions = torch.round(torch.sigmoid(outputs))
                num_correct += torch.sum(predictions == labels).item()

            validation_accuracy = num_correct / num_samples

        tqdm.write(
            f"Epoch {epoch_idx:03}/{num_epochs:03},"
            f"Validation loss: {validation_loss:12f},"
            f"Validation accuracy: {validation_accuracy:12f}"
        )
        wandb.log(
            {
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

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
                    neural_net.state_dict(),
                    f"${model.__class__.__name__}-checkpoint-{epoch_idx}.pt",
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > early_stopping_patience:
            tqdm.write(f"Early stopping after {epoch_idx} epochs without improvement")
            break

    if save_model:
        file = f"{model.__class__.__name__}-trained.pt"
        tqdm.write(f"Saving model to {file}")
        torch.save(neural_net.state_dict(), file)
        wandb.save(file)
    # Close weights and biases logging (needed for Jupyter Notebooks)
    wandb.finish()
    return neural_net


if __name__ == "__main__":
    model = Model1()
    config = Config()
    dataset_path = os.path.abspath(
        os.getenv("DATASET_PATH")
        or config.get("Paths", "training_set")
        or "./datasets/training_set"
    )
    log.info(f"Dataset path: {dataset_path}")

    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = ImageFolder(
        root=dataset_path,
        transform=img_transform,
    )

    model.train(True)
    train_model(model, dataset)
