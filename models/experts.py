import torch
import torch.nn as nn

class ExpertLayer(nn.Module):
    """
    Expert Network (The "Specialist"):
    A self-contained "Small CNN" designed to handle specific visual patterns.
    
    Role:
    - Receive a subset of images from the MoE Manager.
    - Output classification scores (e.g., 100 classes for CIFAR-100).
    
    Current Architecture:
    - A standard small CNN for CIFAR.
    - Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> Linear -> Output (100 classes).
    """
    def __init__(self, input_channels=3, num_classes=100):
        super().__init__()
        #in_channels = 3 (r,g b), out_channels = 32 (might change)  
        self.conv1 = torch.nn.Conv2d(in_channels = input_channels, out_channels = 32, kernel_size = 3, padding = 1)
        self.relu = torch.nn.ReLU()
        #32x32 -> 16x16 divide hw/2
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding= 1)
        #convert to 1D vector
        self.flatten = torch.nn.Flatten()
        #out_features = 100 (Cifar 100)
        self.linear = torch.nn.Linear(in_features = 4096, out_features= num_classes)

        '''
        Image départ : 32 × 32
        Après Pool 1 : 16×16
        Après Pool 2 : 8×8
        Nombre de canaux (sortie conv2) : 64
        Calcul : 64 × 8 × 8 = 4096
        '''
     
    def forward(self, x): #return Logits
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
