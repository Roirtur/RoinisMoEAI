import torch.nn as nn
import torch.nn.functional as F

class SimpleBaseline(nn.Module):
    """
    A Simple CNN Baseline that matches the architecture of the ExpertLayer.
    Scalable via width_multiplier to match parameter counts to test larger models.
    """
    def __init__(self, input_shape=(3, 32, 32), num_classes=10, width_multiplier=1.0):
        super(SimpleBaseline, self).__init__()
        
        self.planes1 = int(32 * width_multiplier)
        self.planes2 = int(64 * width_multiplier)
        self.hidden_size = int(512 * width_multiplier)
        
        # Layer 1
        self.conv1 = nn.Conv2d(input_shape[0], self.planes1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.planes1)
        self.pool1 = nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        
        # Layer 2
        self.conv2 = nn.Conv2d(self.planes1, self.planes2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.planes2)
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8
        
        self.flatten = nn.Flatten()
        
        self.flat_size = self.planes2 * 8 * 8
        
        self.fc1 = nn.Linear(self.flat_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_baseline(input_shape, num_classes, width_multiplier=1.0):
    return SimpleBaseline(input_shape=input_shape, num_classes=num_classes, width_multiplier=width_multiplier)

