# File: utils/checkpoint.py
"""
Checkpoint saving and loading utilities.
"""

import torch
import torch.nn as nn
import shutil
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth',
    ema: Optional = None,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Metrics dictionary
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        ema: EMA model (optional)
        is_best: Whether this is the best model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    # Save as best if needed
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        shutil.copyfile(checkpoint_path, best_path)
        print(f"[OK] Best model saved to {best_path}")
    
    print(f"[OK] Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        ema: EMA model (optional)
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint info
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load EMA
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    
    print("[OK] Checkpoint loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint