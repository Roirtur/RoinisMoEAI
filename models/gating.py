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
    def __init__(self, input_channels=3, num_experts=4):
        super().__init__()
        # TODO: Define the layers for the Tiny CNN router
        # Goal: Rapidly look at the image and decide: "Is this a vehicle, a fish, or a flower?"
        pass

    def forward(self, x):
        # TODO: Return expert weights
        # 1. Pass image through the tiny CNN
        # 2. Apply Softmax to get probabilities (weights)
        return x
