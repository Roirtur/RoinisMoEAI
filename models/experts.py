import torch
import torch.nn as nn

class ExpertLayer(nn.Module):
    """
    The Expert: Specialized CNN.
    Input: (Batch_Subset, 3, 32, 32)
    Output: (Batch_Subset, num_classes)
    """
    def __init__(self, input_channels=3, num_classes=100):
        super().__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8

        self.flatten = nn.Flatten()
        
        # 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x