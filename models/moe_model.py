import torch
import torch.nn as nn
from experts import ExpertLayer
from gating import GatingNetwork

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
        self.router = GatingNetwork(input_channels= input_channels, num_experts= num_experts)

        #moduleList sinon pytorch ne gère pas
        self.experts = nn.ModuleList([
            ExpertLayer(input_channels=input_channels, num_classes=num_classes) 
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # TODO: Implement True Conditional Computation logic
        
        # 1. Get routing weights from the Gating Network
        weights = self.router(x)
        
        # 2. Hard Routing (Top-1 Selection):
        #    Find which expert has the highest score for each image in the batch.
        best_weights_weights, best_index = torch.max(weights, dim=1)
        num_classes = self.experts[0].linear.out_features 
        
        # 3. Initialize a final output tensor (zeros)
        batch_size = x.size(0)
        num_classes = self.experts[0].linear.out_features
        final_output = torch.zeros((batch_size, num_classes), device=x.device)

        # 4. Dispatch Loop (The Efficient Part):
        #    Iterate through each expert (0 to num_experts-1):
        for i , expert in enumerate(self.experts):
        #       a. Identify which images in the batch are assigned to this expert.
        #          indices = (top_indices == i).nonzero()
            indexes = (best_index == i).nonzero(as_tuple=True)[0]                  
        #       b. IF there are images for this expert:
            if len(indexes) > 0:
        #           i.   Select those specific images.
                    selected_x = x[indexes]
        #           ii.  Run ONLY the selected expert on those images.
                    expert_output = expert(selected_x)
        #           iii. Store the results in the final output tensor.
                    final_output[indexes] = expert_output
        #           iv.  (Optional) Scale by the gating weight if desired.
        
        return final_output, weights

#quick dimension test might delete later
if __name__ == "__main__":
    # Test unitaire rapide
    print("Testing MoEModel...")
    
    # 1. Configuration
    batch_size = 10
    num_experts = 4
    num_classes = 100
    
    # 2. Données factices (sur CPU pour le test)
    x = torch.randn(batch_size, 3, 32, 32)
    
    # 3. Instanciation du Manager
    model = MoEModel(num_experts=num_experts, num_classes=num_classes)
    print(f"Modèle créé avec {len(model.experts)} experts.")

    # 4. Passage Forward
    output, gating_probs = model(x)
    
    print(f"Output shape: {output.shape}")      # Attendu: [10, 100]
    print(f"Gating props shape: {gating_probs.shape}")   # Attendu: [10, 4]

    # 5. Vérification basique
    if output.shape == (batch_size, num_classes):
        # Vérifions qu'il n'y a pas que des zéros (ce qui voudrait dire que le dispatch a échoué)
        if torch.abs(output).sum() > 0:
            print("✅ SUCCÈS : Le flux de données complet fonctionne !")
        else:
            print("❌ ATTENTION : La sortie ne contient que des zéros.")
    else:
        print("❌ ERREUR de dimension.")