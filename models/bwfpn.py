# File: models/bwfpn.py
"""
Bi-directional Weighted Feature Pyramid Network (BWFPN) - FIXED.
Enhanced FPN with learnable weights for feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BWFPN(nn.Module):
    """
    Bi-directional Weighted Feature Pyramid Network.
    
    Features:
    - Top-down pathway for semantic information flow
    - Bottom-up pathway for localization information flow
    - Learnable fusion weights (fast normalized fusion)
    - Consistent 128-channel output for all scales
    - Handles dimension mismatches automatically
    """
    
    def __init__(self, in_channels_list: list = [40, 112, 160], out_channels: int = 128):
        super(BWFPN, self).__init__()
        
        # Lateral connections (reduce channels)
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        
        # Top-down pathway
        self.td_conv_c4 = ConvBlock(out_channels, out_channels)
        self.td_conv_c3 = ConvBlock(out_channels, out_channels)
        
        # Bottom-up pathway
        self.bu_conv_c4 = ConvBlock(out_channels, out_channels)
        self.bu_conv_c5 = ConvBlock(out_channels, out_channels)
        
        # Learnable fusion weights (Fast Normalized Fusion)
        # Top-down weights
        self.w_td_c4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_td_c3 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
        # Bottom-up weights
        self.w_bu_c4 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_bu_c5 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
        # Output convolutions
        self.out_conv_p3 = ConvBlock(out_channels, out_channels)
        self.out_conv_p4 = ConvBlock(out_channels, out_channels)
        self.out_conv_p5 = ConvBlock(out_channels, out_channels)
        
        self.epsilon = 1e-4
    
    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Fast normalized fusion weights."""
        weights = F.relu(weights)
        return weights / (weights.sum() + self.epsilon)
    
    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> tuple:
        """
        Forward pass of BWFPN with dimension mismatch handling.
        
        Args:
            c3: C3 features (B, C3, H/8, W/8)
            c4: C4 features (B, C4, H/16, W/16)
            c5: C5 features (B, C5, H/32, W/32)
            
        Returns:
            Tuple of (P3, P4, P5) - all with same channels (128)
        """
        # Lateral connections
        lat_c3 = self.lateral_c3(c3)
        lat_c4 = self.lateral_c4(c4)
        lat_c5 = self.lateral_c5(c5)
        
        # === Top-Down Pathway ===
        td_p5 = lat_c5
        
        # P4
        w_td_c4 = self._normalize_weights(self.w_td_c4)
        td_p5_up = F.interpolate(td_p5, size=lat_c4.shape[2:], mode='nearest')
        td_p4 = self.td_conv_c4(w_td_c4[0] * lat_c4 + w_td_c4[1] * td_p5_up)
        
        # P3
        w_td_c3 = self._normalize_weights(self.w_td_c3)
        td_p4_up = F.interpolate(td_p4, size=lat_c3.shape[2:], mode='nearest')
        td_p3 = self.td_conv_c3(w_td_c3[0] * lat_c3 + w_td_c3[1] * td_p4_up)
        
        # === Bottom-Up Pathway ===
        bu_p3 = td_p3
        
        # P4 with size matching
        w_bu_c4 = self._normalize_weights(self.w_bu_c4)
        bu_p3_down = F.max_pool2d(bu_p3, kernel_size=2, stride=2)
        
        # Fix dimension mismatch
        if bu_p3_down.shape[2:] != lat_c4.shape[2:]:
            bu_p3_down = F.interpolate(bu_p3_down, size=lat_c4.shape[2:], mode='nearest')
        
        bu_p4 = self.bu_conv_c4(
            w_bu_c4[0] * lat_c4 + 
            w_bu_c4[1] * td_p4 + 
            w_bu_c4[2] * bu_p3_down
        )
        
        # P5 with size matching
        w_bu_c5 = self._normalize_weights(self.w_bu_c5)
        bu_p4_down = F.max_pool2d(bu_p4, kernel_size=2, stride=2)
        
        # Fix dimension mismatch
        if bu_p4_down.shape[2:] != td_p5.shape[2:]:
            bu_p4_down = F.interpolate(bu_p4_down, size=td_p5.shape[2:], mode='nearest')
        
        bu_p5 = self.bu_conv_c5(w_bu_c5[0] * td_p5 + w_bu_c5[1] * bu_p4_down)
        
        # === Output convolutions ===
        p3 = self.out_conv_p3(bu_p3)
        p4 = self.out_conv_p4(bu_p4)
        p5 = self.out_conv_p5(bu_p5)
        
        return p3, p4, p5


if __name__ == '__main__':
    # Test BWFPN
    bwfpn = BWFPN()
    
    c3 = torch.randn(2, 40, 52, 52)
    c4 = torch.randn(2, 112, 26, 26)
    c5 = torch.randn(2, 160, 13, 13)
    
    p3, p4, p5 = bwfpn(c3, c4, c5)
    
    print("BWFPN output shapes:")
    print(f"  P3: {p3.shape}")
    print(f"  P4: {p4.shape}")
    print(f"  P5: {p5.shape}")