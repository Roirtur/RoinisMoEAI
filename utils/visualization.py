import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import torch

class HistoryLogger:
    """
    Simple logger to track metrics during training.
    """
    def __init__(self):
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'expert_usage': [],
            'expert_loss': [],
            'expert_class_distribution': []
        }

    def log_epoch(self, train_loss, train_acc, val_loss, val_acc, expert_counts=None, expert_losses=None, expert_class_dist=None):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        if expert_counts is not None:
            if isinstance(expert_counts, torch.Tensor):
                expert_counts = expert_counts.cpu().tolist()
            self.history['expert_usage'].append(expert_counts)
        if expert_losses is not None:
            self.history['expert_loss'].append(expert_losses)
        if expert_class_dist is not None:
            self.history['expert_class_distribution'].append(expert_class_dist)

    def save(self, filepath):
        """Save history to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f)
    
    @staticmethod
    def load(filepath):
        """Load history from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger = HistoryLogger()
        logger.history = data
        return logger

def plot_learning_curves(history_baseline, history_moe, save_dir):
    """
    Plots comparison of Baseline vs MoE.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if hasattr(history_baseline, 'history'):
        history_baseline = history_baseline.history
    if hasattr(history_moe, 'history'):
        history_moe = history_moe.history

    epochs_moe = range(1, len(history_moe['train_acc']) + 1)
    epochs_base = range(1, len(history_baseline['train_acc']) + 1)
    
    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_base, history_baseline['val_acc'], label='Baseline (Val)', linestyle='--')
    plt.plot(epochs_moe, history_moe['val_acc'], label='MoE (Val)', linewidth=2)
    plt.plot(epochs_moe, history_moe['train_acc'], label='MoE (Train)', alpha=0.5)
    
    plt.title('Learning Curve: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'comparison_accuracy.png'))
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_base, history_baseline['val_loss'], label='Baseline (Val)', linestyle='--')
    plt.plot(epochs_moe, history_moe['val_loss'], label='MoE (Val)', linewidth=2)
    plt.plot(epochs_moe, history_moe['train_loss'], label='MoE (Train)', alpha=0.5)
    
    plt.title('Learning Curve: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'comparison_loss.png'))
    plt.close()

def plot_expert_utilization(history_moe, save_dir):
    """
    Plots how expert usage evolves over time.
    history_moe: dict containing 'expert_usage' (list of lists)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if hasattr(history_moe, 'history'):
        history_moe = history_moe.history

    if not history_moe.get('expert_usage'):
        print("No expert usage data found in history.")
        return

    usage_data = np.array(history_moe['expert_usage']) # shape: [Epochs, Num_Experts]
    epochs = range(1, usage_data.shape[0] + 1)
    num_experts = usage_data.shape[1]
    
    row_sums = usage_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    usage_pct = (usage_data / row_sums) * 100
    
    plt.figure(figsize=(10, 6))
    
    pal = sns.color_palette("tab10", num_experts)
    plt.stackplot(epochs, usage_pct.T, labels=[f'Expert {i}' for i in range(num_experts)], colors=pal, alpha=0.8)
    
    plt.title('Expert Utilization Over Training (Stacked)')
    plt.xlabel('Epochs')
    plt.ylabel('Usage (%)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expert_utilization_evolution.png'))
    plt.close()

def plot_expert_heatmap(model, dataloader, device, save_path):
    """
    Generates the Expert vs Class Heatmap.
    This shows which experts handle which classes
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    num_experts = model.num_experts
    num_classes = model.num_classes
    
    heatmap = np.zeros((num_experts, num_classes))
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, router_probs, _ = model(inputs)
            
            _, topk_indices = torch.topk(router_probs, k=model.top_k, dim=1) # (Batch, k)
            
            topk_indices = topk_indices.cpu().numpy()
            targets = targets.cpu().numpy()
            
            for i in range(len(targets)):
                cls = targets[i]
                for k in range(model.top_k):
                    expert = topk_indices[i, k]
                    heatmap[expert, cls] += 1
                
    # normalize per class
    col_sums = heatmap.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    heatmap_norm = heatmap / col_sums
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_norm, cmap="viridis", cbar_kws={'label': 'Probability'})
    plt.title('Expert Specialization by Class')
    plt.xlabel('CIFAR-10 Classes')
    plt.ylabel('Expert ID')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_params_vs_performance(models_data, save_path):
    """
    Scatter plot comparing Model Size (Parameters/FLOPs) vs Accuracy.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    for m in models_data:
        plt.scatter(m['params'], m['accuracy'], s=100, label=m['name'])
        plt.text(m['params'], m['accuracy'] + 0.5, m['name'], fontsize=9, ha='center')

    plt.title('Accuracy vs. Parameter Count')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_expert_class_distribution(model, dataloader, device):
    """
    Computes which expert handles which class on the given dataloader.
    """
    model.eval()
    heatmap = torch.zeros(model.num_experts, model.num_classes, device=device)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            if len(out) == 3:
                _, router_probs, _ = out
            else:
                _, router_probs = out
            
            _, top1_indices = torch.max(router_probs, dim=1)
            
            for i in range(len(targets)):
                expert_idx = top1_indices[i]
                class_idx = targets[i]
                heatmap[expert_idx, class_idx] += 1
                
    return heatmap.cpu().tolist()