import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from utils.data_loader import get_dataloaders
from models.dense_baseline import get_baseline
from models.moe_model import MoEModel
from utils import HistoryLogger, get_expert_class_distribution

def train_baseline(model, train_loader, val_loader, test_loader, epochs, device, save_path):
    """
    Training loop specifically for the Dense Baseline model.
    """
    print(f"Starting Dense Baseline training on {device}...")
    
    logger = HistoryLogger()
    history_path = save_path.replace('.pth', '_history.json')

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
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        train_acc = 100. * correct / total
        
        val_acc, val_loss = evaluate(model, val_loader, device)
        test_acc, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Log metrics
        logger.log_epoch(running_loss/total, train_acc, val_loss, val_acc)
        
        # Save model and history
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.save(history_path)
            
        scheduler.step()

    print(f"Training finished. Final Test Accuracy: {test_acc:.2f}%")

def train_moe(model, train_loader, val_loader, test_loader, epochs, device, save_path, aux_weight=5.0):
    """
    Training loop specifically for the Mixture of Experts model.
    """
    print(f"Starting MoE training on {device} with aux_weight={aux_weight}, Top-K={model.top_k}")
    
    logger = HistoryLogger()
    history_path = save_path.replace('.pth', '_history.json')
    
    criterion = nn.CrossEntropyLoss()
    criterion_noreduce = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # expert usage for this epoch
        epoch_expert_counts = torch.zeros(model.num_experts, device=device)
        
        # expert loss tracking
        epoch_expert_loss_sum = torch.zeros(model.num_experts, device=device)
        epoch_expert_loss_count = torch.zeros(model.num_experts, device=device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs, router_probs, aux_loss = model(inputs)
            
            # Count expert selections for logging
            with torch.no_grad():
                # indices of the top-k experts selected
                _, topk_indices = torch.topk(router_probs, k=model.top_k, dim=1) # (Batch, k)
                
                flat_indices = topk_indices.view(-1)
                
                # usage count
                for i in range(model.num_experts):
                    epoch_expert_counts[i] += (flat_indices == i).sum()
                    
                # loss attribution
                sample_losses = criterion_noreduce(outputs, targets)
                top1_expert = topk_indices[:, 0]
                
                for i in range(model.num_experts):
                    mask = (top1_expert == i)
                    if mask.any():
                        epoch_expert_loss_sum[i] += sample_losses[mask].sum()
                        epoch_expert_loss_count[i] += mask.sum()

            loss = criterion(outputs, targets) + aux_weight * aux_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        train_acc = 100. * correct / total
        
        # average loss per expert
        safe_counts = epoch_expert_loss_count.clone()
        safe_counts[safe_counts == 0] = 1.0
        expert_losses = (epoch_expert_loss_sum / safe_counts).cpu().tolist()
        
        val_acc, val_loss = evaluate_moe(model, val_loader, device)
        test_acc, _ = evaluate_moe(model, test_loader, device)
        
        # distribution of data treated by experts
        expert_class_dist = get_expert_class_distribution(model, val_loader, device)
        
        total_selections = total * model.top_k
        
        # percentage of total capacity used by each expert
        usage_str = " | ".join([f"E{i}:{(c/total_selections)*100:.1f}%" for i, c in enumerate(epoch_expert_counts)])
        
        print(f"Epoch {epoch+1}: Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%")
        print(f"   Usage: [{usage_str}]")
        
        logger.log_epoch(running_loss/total, train_acc, val_loss, val_acc, 
                         expert_counts=epoch_expert_counts,
                         expert_losses=expert_losses,
                         expert_class_dist=expert_class_dist)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.save(history_path)
            
        scheduler.step()

    print(f"MoE Training finished. Final Test Accuracy: {test_acc:.2f}%")

def evaluate_moe(model, dataloader, device, aux_weight=1.0):
    """
    Evaluation loop specifically for MoE (handling tuple output).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, aux_loss = model(inputs)
            loss = criterion(outputs, targets) + aux_weight * aux_loss
            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total, running_loss / total

def evaluate(model, dataloader, device):
    """
    Generic evaluation loop.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total, running_loss / total

def main():
    parser = argparse.ArgumentParser(description='Train MoE or Dense Baseline on CIFAR-10')
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'moe'],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--width_multiplier', type=float, default=1.0, 
                        help='Width multiplier for Dense Baseline (to match MoE parameters/FLOPs)')
    parser.add_argument('--aux_weight', type=float, default=5.0, 
                        help='Auxiliary loss weight for load balancing')
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
        
        train_moe(model, train_loader, val_loader, test_loader, args.epochs, device, save_path, aux_weight=args.aux_weight)

if __name__ == "__main__":
    main()
