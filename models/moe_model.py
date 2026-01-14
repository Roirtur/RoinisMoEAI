import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ExpertLayer
from models import GatingNetwork

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
    def __init__(self, num_experts=4, num_classes=10, input_channels = 3, top_k = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = GatingNetwork(input_channels=input_channels, num_experts=num_experts)

        self.experts = nn.ModuleList([
            ExpertLayer(input_channels=input_channels, num_classes=num_classes) 
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Forward pass of the Mixture of Experts.
        
        Steps:
        1. Router calculates probabilities for each expert.
        2. Select top-k experts for each image.
        3. Dispatch images to selected experts only (sparse execution).
        4. Accumulate weighted outputs from experts.
        5. Calculate auxiliary load balancing loss.

        :param x: Input tensor of shape (Batch_Size, 3, 32, 32)
        :return: A tuple containing:

            - final_output: (Batch_Size, num_classes) -> Final classification scores.
            - router_probs: (Batch_Size, num_experts) -> Router's raw confidence scores.
            - aux_loss: Scalar tensor -> Load balancing loss to ensure expert diversity.
        """

        #Probs vector, contains the probs that each experts is competent for each image
        router_probs = self.router(x)
        
        # best top-k experts for each image (Top-1 default)
        topk_probs, topk_indexes = torch.topk(router_probs, self.top_k, dim=1)
        
        # empty container for finals results
        final_output = torch.zeros(x.size(0), self.num_classes, device=x.device)

        #--- Distributing work to experts ---#
        for i, expert in enumerate(self.experts):
            batch_indexes, k_rank = (topk_indexes == i).nonzero(as_tuple=True)
            
            # if len(batch_indexes) = 0 expert hasnt been chosen for any image
            if len(batch_indexes) > 0:

                #images concerned by current i expert  
                selected_inputs = x[batch_indexes]
                
                # forward pass through the specific expert
                expert_output = expert(selected_inputs)
                
                # weight the output by the gating probability
                scaling_factor = topk_probs[batch_indexes, k_rank].unsqueeze(1)
                
                # Accumulate weighted expert predictions into the final output tensor
                final_output[batch_indexes] += expert_output * scaling_factor
        
        # load balancing auxiliary loss
        # encourages average uniform usage of experts
        mean_probs = router_probs.mean(dim=0)
        target_probs = torch.ones_like(mean_probs) / self.num_experts
        aux_loss = F.mse_loss(mean_probs, target_probs)
        
        return final_output, router_probs, aux_loss