import torch
import torch.nn as nn
from .experts import ExpertLayer
from .gating import GatingNetwork

class MoEModel(nn.Module):
    """
    MoE Model (The "Manager"):
    Coordinates the Gating Network and the Experts to perform Conditional Computation.
    
    Concept:
    1. The Router (GatingNetwork) looks at the image and decides which Expert is best.
    2. The Manager (MoEModel) dispatches the image ONLY to that specific Expert (Hard Routing).
    3. The Result is combined.
    
    Architecture:
    - 1 GatingNetwork
    - N ExpertLayers (ModuleList)
    """
    def __init__(self, num_experts=4, num_classes=100):
        super().__init__()
        # TODO: Initialize the GatingNetwork (The Router)
        # TODO: Initialize a ModuleList of ExpertLayers (The Specialists)
        pass

    def forward(self, x):
        # TODO: Implement True Conditional Computation logic
        
        # 1. Get routing weights from the Gating Network
        #    weights = self.gate(x)
        
        # 2. Hard Routing (Top-1 Selection):
        #    Find which expert has the highest score for each image in the batch.
        #    top_weights, top_indices = torch.max(weights, dim=1)
        
        # 3. Initialize a final output tensor (zeros)
        
        # 4. Dispatch Loop (The Efficient Part):
        #    Iterate through each expert (0 to num_experts-1):
        #       a. Identify which images in the batch are assigned to this expert.
        #          indices = (top_indices == i).nonzero()
        #       
        #       b. IF there are images for this expert:
        #           i.   Select those specific images.
        #           ii.  Run ONLY the selected expert on those images.
        #           iii. Store the results in the final output tensor.
        #           iv.  (Optional) Scale by the gating weight if desired.
        
        return x # Placeholder return
