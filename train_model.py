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


def make_training_set(batch_size):
    training_set = []
    return torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )


def train_model(
    model,
    epochs=200,
    batch_size=128,
    lr=0.1,
    loss=torch.nn.BCELoss(),
    optimizer=None,
):
    net = model
    optimizer = (
        torch.optim.RMSprop(net.parameters(), lr=lr)
        if optimizer is None
        else optimizer(net.parameters(), lr=lr)
    )
    total_loss = 0.0
    training_loader = make_training_set(batch_size)

    for epoch in range(epochs):
        print(f"[INFO] running epoch {epoch}")
        global inputs, labels
        epoch_loss = 0.0
        for i, mini_batch in enumerate(training_loader):
            # generate our mini batch
            inputs, labels = mini_batch
            optimizer.zero_grad()
            # forward pass
            outputs = net(inputs)
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
