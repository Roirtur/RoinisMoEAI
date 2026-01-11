import torch
import torch.nn as nn

class ExpertLayer(nn.Module):
    """
    Define a class ExpertLayer (e.g., a simple 2-layer MLP with ReLU).
    Ensure it handles batch inputs correctly.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: Implement the expert architecture (e.g. Linear -> ReLU -> Linear)
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass
