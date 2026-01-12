import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experts import ExpertLayer
from models.gating import GatingNetwork

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
    def __init__(self, num_experts=4, num_classes=100, input_channels = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        self.router = GatingNetwork(input_channels=input_channels, num_experts=num_experts)

        self.experts = nn.ModuleList([
            ExpertLayer(input_channels=input_channels, num_classes=num_classes) 
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # probabilities from Router
        router_probs = self.router(x)
        
        # best expert for each image (Top-1)
        best_weights, best_indices = torch.max(router_probs, dim=1)
        
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self.num_classes, device=x.device)

        for i, expert in enumerate(self.experts):
            indices = (best_indices == i).nonzero(as_tuple=True)[0]
            
            if len(indices) > 0:
                selected_inputs = x[indices]
                
                # forward pass through the specific expert
                expert_output = expert(selected_inputs)
                
                # weight the output by the gating probability
                scaling_factor = best_weights[indices].unsqueeze(1)
                
                final_output[indices] = expert_output * scaling_factor
        
        # load balancing auxiliary loss
        # encourages average uniform usage of experts
        mean_probs = router_probs.mean(dim=0)
        target_probs = torch.ones_like(mean_probs) / self.num_experts
        aux_loss = F.mse_loss(mean_probs, target_probs)
        
        return final_output, router_probs, aux_loss

if __name__ == "__main__":
    # Quick Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    x = torch.randn(10, 3, 32, 32).to(device)
    model = MoEModel(num_experts=4, num_classes=100).to(device)
    
    outputs, probs, aux_loss = model(x)
    
    print(f"Output shape: {outputs.shape}") # Should be [10, 100]
    print(f"Prob shape:   {probs.shape}")   # Should be [10, 4]
    print(f"Aux Loss:     {aux_loss.item()}")
    print("Success.")