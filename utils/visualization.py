# File: utils/visualization.py
"""
Visualization utilities for training curves and predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_maps: list,
    val_maps: list,
    learning_rates: list,
    save_path: str
):
    """
    Plot training curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_maps: Training mAPs (not used during training)
        val_maps: Validation mAPs (not used during training)
        learning_rates: Learning rates
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Learning rate
    axes[1].plot(epochs, learning_rates, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {save_path}")