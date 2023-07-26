import torch.nn as nn

# Define the network architecture
# This is a CNN used for classification with 2 classes.
# The architecture is inspired by LeNet-5 (http://yann.lecun.com/exdb/lenet/)
# The input is a 3-channel RGB image of size 256x256
# The output is a 1-element vector with the probabilities of a person being masked 1 or not 0


class LeNetty(nn.Module):
    """An Interpretation of the LeNet-5 model."""

    def __init__(self):
        super(LeNetty, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 61 * 61, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
