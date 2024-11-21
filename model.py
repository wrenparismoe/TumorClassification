import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # Activation for conv1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # Activation for conv2

        # Fully connected layers
        self.flattened_size = 56 * 56 * 64  # Precomputed as before
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu_fc1 = nn.ReLU()  # Activation for fc1

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.conv1(x)
        x = self.relu1(x)  # Apply ReLU activation
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu2(x)  # Apply ReLU activation
        x = self.pool(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)  # Apply ReLU activation
        x = self.fc2(x)
        return x
