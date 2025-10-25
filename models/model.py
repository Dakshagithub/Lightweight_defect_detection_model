# File: models/model.py
"""
Complete GhostCAM-BWFPN model.
"""

import torch
import torch.nn as nn
from .ghostnet import GhostNet
from .bwfpn import BWFPN
from .detection_head import DetectionHead


class GhostCAMBWFPN(nn.Module):
    """
    Complete GhostCAM-BWFPN detection model.
    
    Architecture:
    GhostNet (backbone) -> BWFPN (neck) -> DetectionHead (head)
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        anchors: list = None
    ):
        super(GhostCAMBWFPN, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = GhostNet(width_mult=width_mult, dropout=dropout)
        
        # Neck
        self.neck = BWFPN(in_channels_list=[40, 112, 160], out_channels=128)
        
        # Head
        self.head = DetectionHead(num_classes=num_classes, anchors=anchors)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass."""
        # Backbone
        c3, c4, c5 = self.backbone(x)
        
        # Neck
        p3, p4, p5 = self.neck(c3, c4, c5)
        
        # Head
        pred_p3, pred_p4, pred_p5 = self.head(p3, p4, p5)
        
        return pred_p3, pred_p4, pred_p5


if __name__ == '__main__':
    model = GhostCAMBWFPN(num_classes=6)
    x = torch.randn(2, 3, 200, 200)
    pred_p3, pred_p4, pred_p5 = model(x)
    
    print("Model output shapes:")
    print(f"  P3: {pred_p3.shape}")
    print(f"  P4: {pred_p4.shape}")
    print(f"  P5: {pred_p5.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")