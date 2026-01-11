import torch
import torch.nn as nn

class DenseBaseline(nn.Module):
    """
    Create a standard sequential network.
    CRUCIAL: Calculate parameter count to ensure 'equivalent capacity' to MoE.
    Options: 
        - Iso-FLOPs (active params): Fair for speed comparison
        - Iso-Param (total params): Fair for storage comparison
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO: Define a dense network comparable to the MoE
        pass

    def forward(self, x):
        pass
