import torch.nn as nn
import torch.nn.functional as F
class GatingNetwork(nn.Module):
    """
    The Router: Lightweight CNN.
    Input: (Batch, 3, 32, 32)
    Output: (Batch, num_experts) -> Softmax probabilities
    """
    def __init__(self, input_channels=3, num_experts=4):
        super().__init__()
        
        # 32x32 -> 16x16
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # 32 * 8 * 8 = 2048
        self.linear = nn.Linear(2048, num_experts)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        logits = self.linear(x)
        
        return F.softmax(logits, dim=1)