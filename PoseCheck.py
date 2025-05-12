import torch
import torch.nn as nn
import torch.optim as optim

class PoseCheck(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)  # Conv1d with padding to keep sequence length
        self.bn1 = nn.BatchNorm1d(32)  # Batch Normalization
        self.relu = nn.ReLU()  # ReLU activation
        self.pool = nn.MaxPool1d(2)  # Max pooling to reduce sequence length

        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Increase depth to 64 filters
        self.bn2 = nn.BatchNorm1d(64)  # Batch Normalization

        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # Increase depth to 128 filters
        self.bn3 = nn.BatchNorm1d(128)  # Batch Normalization

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4, 256)  # Linear layer after flattening
        self.fc2 = nn.Linear(256, num_classes)  # Output layer for classification

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Reshape x from [batch_size, 99] to [batch_size, 3, 33]
        x = x.view(x.size(0), 4, 33)  # [batch_size, 3, 33]

        # First convolutional block
        x = self.relu(self.bn1(self.conv1(x)))  # [batch_size, 32, 33]
        x = self.pool(x)  # [batch_size, 32, 16]

        # Second convolutional block
        x = self.relu(self.bn2(self.conv2(x)))  # [batch_size, 64, 16]
        x = self.pool(x)  # [batch_size, 64, 8]

        # Third convolutional block
        x = self.relu(self.bn3(self.conv3(x)))  # [batch_size, 128, 8]
        x = self.pool(x)  # [batch_size, 128, 4]

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer: [batch_size, 128*4] = [batch_size, 512]

        # First fully connected layer
        x = self.relu(self.fc1(x))  # [batch_size, 256]
        x = self.dropout(x)  # Apply dropout

        # Output layer
        x = self.fc2(x)  # [batch_size, num_classes]
  

        return x