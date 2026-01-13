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

def plot_multimodel_learning_curves(histories_dict, save_dir):
    """
    Plots comparison of multiple models (Validation Accuracy & Loss)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    markers = ['o', '^', 'x', 's', 'd', 'v', '*', 'p']
    linestyles = ['-', '--', '-.', ':']
    
    # Determine max epochs
    max_epochs = 0
    for h in histories_dict.values():
        if hasattr(h, 'history'): h = h.history
        if len(h['val_acc']) > max_epochs:
            max_epochs = len(h['val_acc'])

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    for i, (name, history) in enumerate(histories_dict.items()):
        if hasattr(history, 'history'): history = history.history
        epochs = range(1, len(history['val_acc']) + 1)
        
        style_idx = i % len(linestyles)
        marker_idx = i % len(markers)
        
        plt.plot(epochs, history['val_acc'], label=name, 
                 linestyle=linestyles[style_idx], marker=markers[marker_idx], linewidth=1.5, alpha=0.8)

    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    if max_epochs > 0:
        plt.xlim(1, max_epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multimodel_comparison_accuracy.png'))
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    for i, (name, history) in enumerate(histories_dict.items()):
        if hasattr(history, 'history'): history = history.history
        epochs = range(1, len(history['val_loss']) + 1)
        
        style_idx = i % len(linestyles)
        marker_idx = i % len(markers)
        
        plt.plot(epochs, history['val_loss'], label=name, 
                 linestyle=linestyles[style_idx], marker=markers[marker_idx], linewidth=1.5, alpha=0.8)

    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if max_epochs > 0:
        plt.xlim(1, max_epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multimodel_comparison_loss.png'))
    plt.close()

def plot_expert_loss_history(history, save_dir, title_suffix=""):
    """
    Plots individual loss curves for each expert
    """
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(history, 'history'): history = history.history
    
    if 'expert_loss' not in history or not history['expert_loss']:
        print(f"No expert loss data found for {title_suffix}.")
        return

    expert_losses = np.array(history['expert_loss']) # (Epochs, Num_Experts)
    epochs = range(1, expert_losses.shape[0] + 1)
    num_experts = expert_losses.shape[1]
    
    plt.figure(figsize=(10, 6))
    for i in range(num_experts):
        plt.plot(epochs, expert_losses[:, i], label=f'Expert {i}')
        
    plt.title(f'Expert Loss Evolution {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f'expert_loss_evolution{title_suffix.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_expert_heatmap_from_history(history, save_dir, title_suffix=""):
    """
    Plots Expert Specialization Heatmap from the last epoch of training history.
    """
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(history, 'history'): history = history.history

    if 'expert_class_distribution' not in history or not history['expert_class_distribution']:
        print(f"No expert class distribution data found for {title_suffix}.")
        return
        
    # last epoch distribution
    final_dist = np.array(history['expert_class_distribution'][-1])  # (Num_Experts, Num_Classes)
    
    col_sums = final_dist.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    heatmap_norm = final_dist / col_sums
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_norm, cmap="viridis", cbar_kws={'label': 'Probability'})
    plt.title(f'Expert Specialization by Class (Last Epoch) {title_suffix}')
    plt.xlabel('CIFAR-10 Classes')
    plt.ylabel('Expert ID')
    plt.tight_layout()
    
    filename = f'expert_specialization_heatmap{title_suffix.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def count_total_params(model):
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_active_params_moe(model):
    """
    Counts active parameters for a forward pass in MoE.
    Active = Router + Experts * k
    """
    router_params = sum(p.numel() for p in model.router.parameters())
    expert_params = sum(p.numel() for p in model.experts[0].parameters())
    return router_params + (expert_params * model.top_k)

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

def plot_expert_utilization_histogram(history, save_dir, title_suffix=""):
    """
    Plots a bar chart showing the % of data processed by each expert in the last epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(history, 'history'): history = history.history

    if not history.get('expert_usage'):
        print(f"No expert usage data found for {title_suffix}.")
        return

    # last epoch usage
    last_epoch_counts = np.array(history['expert_usage'][-1])
    total_counts = last_epoch_counts.sum()
    if total_counts == 0: total_counts = 1
    percentages = (last_epoch_counts / total_counts) * 100
    
    num_experts = len(last_epoch_counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f'E{i}' for i in range(num_experts)], y=percentages, hue=[f'E{i}' for i in range(num_experts)], legend=False, palette="viridis")
    
    plt.title(f'Expert Utilization (Last Epoch) {title_suffix}')
    plt.ylabel('Data Processed (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(percentages):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = f'expert_utilization_histogram{title_suffix.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_expert_counts_evolution(history, save_dir, title_suffix=""):
    """
    Plots the raw number of samples processed by each expert over epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(history, 'history'): history = history.history

    if not history.get('expert_usage'):
        print(f"No expert usage data found for {title_suffix}.")
        return

    usage_data = np.array(history['expert_usage']) # (Epochs, Num_Experts)
    epochs = range(1, usage_data.shape[0] + 1)
    num_experts = usage_data.shape[1]
    
    plt.figure(figsize=(10, 6))
    for i in range(num_experts):
        plt.plot(epochs, usage_data[:, i], label=f'Expert {i}', marker='.', markersize=8)
        
    plt.title(f'Expert Sample Counts Evolution {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f'expert_counts_evolution{title_suffix.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(save_dir, filename))
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