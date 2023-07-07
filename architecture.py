import torch
import torch.nn as nn

# Define the network architecture
# This is a CNN used for classification with 2 classes.
# The architecture is inspired by LeNet-5 (http://yann.lecun.com/exdb/lenet/)
# The input is a 3-channel RGB image of size 256x256
# The output is a 2-element vector with the probabilities of each class (mask or no mask)


# Reshaping layer to transform the input from a 2D image to a 4D tensor
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 3, 256, 256)


class LeNetty(nn.Module):
    """An Interpretation of the LeNet-5 model."""

    def __init__(self, lr=0.1, num_classes=2, momentum=0.9, weight_decay=5e-4, batch_size=128, epochs=200):
        super(LeNetty, self).__init__()

        self.features = nn.Sequential(
            Reshape(),
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            # output size: 6x256x256
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # output size: 6x128x128
            nn.Conv2d(6, 16, kernel_size=5, stride=2),
            # output size: 16x62x62
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # output size: 16x31x31
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            # output size: 32x14x14
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
            # output size: 32x7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # output size: 32*7*7=1568
            nn.Linear(32*7*7, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 2)
        )

        def forward(x):
            x = self.features(x)
            x = self.classifier(x)
            return x
