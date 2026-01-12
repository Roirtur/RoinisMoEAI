import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    Gating Network (The "Router"):
    A very lightweight CNN that assesses the input image to determine which experts should handle it.
    
    Role:
    - Rapidly classify the "type" of the image (Cheap computation).
    - Output a probability vector (size = num_experts) representing the weight for each expert.
    
    Recommended Architecture:
    - Tiny CNN to be fast.
    - Example: Conv2d (stride=2 for quick downsampling) -> ReLU -> Flatten -> Linear(num_experts) -> Softmax.
    """
    def __init__(self, input_channels = 3, num_experts = 4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels =  input_channels, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        #in_features = 32 (out conv2) * 8 *8 = 2048
        self.linear = torch.nn.Linear(in_features = 2048, out_features = num_experts)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return torch.nn.functional.softmax(logits, dim = 1)
