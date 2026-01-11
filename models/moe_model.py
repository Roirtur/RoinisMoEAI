import torch
import torch.nn as nn
from .experts import ExpertLayer
from .gating import GatingNetwork

class MoEModel(nn.Module):
    """
    Assemble MoE: Combine Experts and Gating.
    Implement the weighted sum: y = sum(g_i(x) * E_i(x)).
    Allow number of experts to be a parameter.
    Support Soft and Hard routing.
    """
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim, routing_type='soft'):
        super().__init__()
        # TODO: Initialize experts and gating network
        pass

    def forward(self, x):
        # TODO: Implement MoE forward pass
        # 1. Get gating weights
        # 2. Get expert outputs
        # 3. Combine outputs
        pass
