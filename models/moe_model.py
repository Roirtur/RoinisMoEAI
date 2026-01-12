import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, num_experts=4, num_classes=100, input_channels = 3, top_k = 1):
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
        router_probs = self.router(x) # -> [0.1, 0.8, 0.05, 0.05]
        
        # best top-k expert for each image (Top-1 default)
        topk_prob, topk_indexes = torch.topk(router_probs, self.top_k, dim=1)
        
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self.num_classes, device=x.device)

        for i, expert in enumerate(self.experts):
            batch_indexes, k_rank = (topk_indexes == i).nonzero(as_tuple=True)
            
            if len(batch_indexes) > 0:
                selected_inputs = x[batch_indexes]
                
                # forward pass through the specific expert
                expert_output = expert(selected_inputs)
                
                # weight the output by the gating probability
                scaling_factor = topk_prob[batch_indexes, k_rank].unsqueeze(1)
                
                final_output[batch_indexes] += expert_output * scaling_factor
        
        # load balancing auxiliary loss
        # encourages average uniform usage of experts
        mean_probs = router_probs.mean(dim=0)
        target_probs = torch.ones_like(mean_probs) / self.num_experts
        aux_loss = F.mse_loss(mean_probs, target_probs)
        
        return final_output, router_probs, aux_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(10, 3, 32, 32).to(device)
    
    # Test avec Top-2
    print("Testing Top-2 routing...")
    model = MoEModel(num_experts=4, num_classes=100, top_k=2).to(device)
    outputs, probs, aux_loss = model(x)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Prob shape:   {probs.shape}")
    print("Success.")