# File: utils/ema.py
"""
Exponential Moving Average for model weights.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Model Exponential Moving Average.
    Keeps a moving average of model parameters.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = 'cuda'):
        self.decay = decay
        self.device = device
        
        # Create EMA model
        self.ema = deepcopy(model).to(device).eval()
        
        # Disable gradients for EMA model
        for param in self.ema.parameters():
            param.requires_grad = False
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        """Get EMA state dict."""
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.ema.load_state_dict(state_dict)