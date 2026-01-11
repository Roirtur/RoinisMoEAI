import torch
import torch.optim as optim
import argparse

def train(model, dataloader, optimizer, criterion, device):
    """
    Build a generic training function taking model, optimizer, and criterion.
    Implement logging for Loss and Accuracy.
    Note: Ensure you can mask loss based on gating decisions to track "Loss per Expert".
    Extension: Add logic to track "Expert Usage" (frequency of selection) during training.
    """
    # TODO: Training loop
    # return metrics
    pass

def evaluate(model, dataloader, criterion, device):
    # TODO: Evaluation loop
    pass

def main():
    # TODO: Parse arguments (epochs, batch_size, experts, etc.)
    # TODO: Setup device, data, model, optimizer
    # TODO: Run training experiments
    #   - Run 1: Dense Baseline
    #   - Run 2: MoE Soft
    #   - Run 3: MoE Hard
    #   - Run 4: Varying experts
    pass

if __name__ == "__main__":
    main()
