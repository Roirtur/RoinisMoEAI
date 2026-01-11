import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    Create GatingNetwork that takes input x and outputs weights.
    Implement the logic for Softmax output (probabilities).
    """
    def __init__(self, input_dim, num_experts):
        super().__init__()
        # TODO: Define layers for gating
        pass

    def forward(self, x):
        # TODO: Return expert weights (and optionally routing decisions for hard routing)
        pass
