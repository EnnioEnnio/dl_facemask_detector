from architecture import LeNetty
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from data_loader import make_training_and_validation_set
import wandb


def train_model(model,
                training_set=None,
                training_labels=None,
                epochs=200,
                batch_size=128,
                learning_rate=0.1,
                early_stopping_patience=10,
                checkpointing=False,
                loss_function=torch.nn.BCELoss(),
                optimizer=torch.optim.RMSprop):

    # initialize weights and biases logging (wandb)
    wandb.init(
        project="dl_facemask_detection",
        config={
            "architecture": model.__class__.__name__,
            "dataset": "Real-World-Masked-Face-Dataset (RMFD)",
            "learning_rate": learning_rate,
            "epoch": epochs,
            "loss_function": loss_function.__class__.__name__,
            "optimizer": optimizer.__name__,
        }
    )

    # check if training set and labels are provided
    if training_set is None or training_labels is None:
        print("[INFO] No training set and/or labels provided.")

    # Check if number of epochs is too large for dataset for chosen batch size
    possible_number_of_batches = len(training_set) // batch_size
    if possible_number_of_batches > epochs:
        print(
            "[WARNING] Number of epochs is too large for dataset for chosen batch size. ")
        print(f"Number of max. possible epochs: {possible_number_of_batches}")

    # initialize model, optimizer and loss function
    neural_net = model
    optimizer = optimizer(neural_net.parameters(), lr=learning_rate)
    num_batches = len(training_set) // batch_size
    total_loss = 0.0

    # Initialize Early Stopping
    best_loss = np.inf
    epochs_without_improvement = 0

    # Training Loop
    for index_epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        start_index = index_epoch * batch_size
        end_index = start_index + batch_size

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
            image_loss = loss_function(output, image_label)

            # Log image loss to weights and biases
            wandb.log({"image_loss": image_loss})

            # Accumulate loss for the mini-batch
            batch_loss += image_loss

            # Backward pass and parameter update for the current image
            optimizer.zero_grad()
            image_loss.backward()
            optimizer.step()

        # Print and log average batch loss
        batch_loss /= batch_size
        wandb.log({"batch_loss": batch_loss})
        tqdm.write(f"Epoch {index_epoch+1}/{epochs}, Batch Loss: {batch_loss}")

        # Early Stopping (and Checkpointing)
        if batch_loss < best_loss:
            best_loss = batch_loss
            epochs_without_improvement = 0
            if checkpointing:
                torch.save(neural_net.state_dict(),
                           f"checkpoint_{index_epoch}.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > early_stopping_patience:
            tqdm.write(
                f"Early stopping after {index_epoch+1} epochs without improvement.")
            break  # Stop training

        # Accumulate loss for the entire training set
        total_loss += batch_loss

    # Print and log total loss
    total_loss /= num_batches
    wandb.log({"total_loss": total_loss})
    tqdm.write(f"Total Loss: {total_loss}")

    # Return the trained model
    return neural_net


if __name__ == "__main__":
    model = LeNetty()
    model.train(True)

    (training_set, training_labels, validation_set, validation_labels) = make_training_and_validation_set(
        dataset_size=25600, validation_split=0.2)

    trained_model = train_model(model, training_set=training_set,
                                training_labels=training_labels)
    torch.save(trained_model.state_dict(),
               f"{model.__class__.__name__}_trained.pt")
