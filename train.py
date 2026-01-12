import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm
from utils.data_loader import get_dataloaders
from models.dense_baseline import get_baseline
from models.moe_model import MoEModel

def train_baseline(model, train_loader, val_loader, test_loader, epochs, device, save_path):
    """
    Training loop specifically for the Dense Baseline model.
    """
    print(f"Starting Dense Baseline training on {device}...")
    
    criterion = nn.CrossEntropyLoss()
    # Standard SGD for ResNet on CIFAR
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        train_acc = 100. * correct / total
        
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
            
        scheduler.step()

    print(f"Training finished. Final Test Accuracy: {test_acc:.2f}%")

def train_moe(model, train_loader, val_loader, test_loader, epochs, device, save_path):
    """
    Training loop specifically for the Mixture of Experts model.
    """
    print(f"Starting MoE training on {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train MoE]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs, router_probs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        train_acc = 100. * correct / total
        
        val_acc = evaluate_moe(model, val_loader, device)
        test_acc = evaluate_moe(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
            
        scheduler.step()

    print(f"MoE Training finished. Final Test Accuracy: {test_acc:.2f}%")


def evaluate_moe(model, dataloader, device):
    """
    Evaluation loop specifically for MoE (handling tuple output).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs) # Ignore router_probs
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def evaluate(model, dataloader, device):
    """
    Generic evaluation loop.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Train MoE or Dense Baseline on CIFAR-100')
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'moe'],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--width_multiplier', type=float, default=1.0, 
                        help='Width multiplier for Dense Baseline (to match MoE parameters/FLOPs)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    train_loader, val_loader, test_loader, num_classes, img_size = get_dataloaders(batch_size=args.batch_size)
    
    if args.model_type == 'baseline':
        print(f"Initializing Dense Baseline (Width x{args.width_multiplier})...")
        model = get_baseline(input_shape=img_size, num_classes=num_classes, width_multiplier=args.width_multiplier)
        model = model.to(device)
        
        # Define save path based on config
        save_name = f"baseline_w{args.width_multiplier}.pth"
        save_path = os.path.join(args.save_dir, save_name)
        
        train_baseline(model, train_loader, val_loader, test_loader, args.epochs, device, save_path)
        
    elif args.model_type == 'moe':
        model = MoEModel(num_experts=8, num_classes=num_classes, input_channels=img_size[0])
        model = model.to(device)
        
        save_path = os.path.join(args.save_dir, "moe_model.pth")
        
        train_moe(model, train_loader, val_loader, test_loader, args.epochs, device, save_path)

if __name__ == "__main__":
    main()
