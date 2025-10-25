# File: models/ghostnet.py
"""
GhostNet Backbone - Lightweight CNN with Ghost modules.
Optimized for mobile/edge deployment.
"""

import torch
import torch.nn as nn
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GhostModule(nn.Module):
    """
    Ghost Module - generates feature maps with fewer parameters.
    Primary conv + cheap operations to generate ghost features.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True
    ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                     kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, 
                     groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck block with optional SE and stride."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False
    ):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        
        # First ghost module
        self.ghost1 = GhostModule(in_channels, mid_channels, kernel_size=1, relu=True)
        
        # Depthwise conv if stride > 1
        if stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_channels, mid_channels, kernel_size, stride,
                (kernel_size - 1) // 2, groups=mid_channels, bias=False
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)
        
        # SE block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(mid_channels)
        
        # Second ghost module
        self.ghost2 = GhostModule(mid_channels, out_channels, kernel_size=1, relu=False)
        
        # Shortcut
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First ghost module
        x = self.ghost1(x)
        
        # Depthwise conv
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        
        # SE
        if self.use_se:
            x = self.se(x)
        
        # Second ghost module
        x = self.ghost2(x)
        
        return x + self.shortcut(residual)


class GhostNet(nn.Module):
    """
    GhostNet backbone for feature extraction.
    Returns multi-scale features: C3, C4, C5
    """
    
    def __init__(self, width_mult: float = 1.0, dropout: float = 0.2):
        super(GhostNet, self).__init__()
        
        # Building blocks configuration
        # [in_channels, mid_channels, out_channels, kernel_size, stride, use_se]
        self.cfgs = [
            # Stage 1
            [16, 16, 16, 3, 1, False],
            # Stage 2
            [16, 48, 24, 3, 2, False],
            [24, 72, 24, 3, 1, False],
            # Stage 3
            [24, 72, 40, 5, 2, True],
            [40, 120, 40, 5, 1, True],
            # Stage 4
            [40, 240, 80, 3, 2, False],
            [80, 200, 80, 3, 1, False],
            [80, 184, 80, 3, 1, False],
            [80, 184, 80, 3, 1, False],
            [80, 480, 112, 3, 1, True],
            [112, 672, 112, 3, 1, True],
            # Stage 5
            [112, 672, 160, 5, 2, True],
            [160, 960, 160, 5, 1, True],
            [160, 960, 160, 5, 1, True],
            [160, 960, 160, 5, 1, True],
        ]
        
        # Apply width multiplier
        self.cfgs = [[int(c * width_mult) if i < 3 else c for i, c in enumerate(cfg)] 
                     for cfg in self.cfgs]
        
        # Stem
        out_channels = int(16 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_channels = out_channels
        
        for cfg in self.cfgs:
            layers = []
            layers.append(GhostBottleneck(*cfg))
            in_channels = cfg[2]
            self.stages.append(nn.Sequential(*layers))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Tuple of (C3, C4, C5) features
        """
        x = self.stem(x)  # /2
        
        # Stage 1-2: C2 (not used)
        for i in range(3):
            x = self.stages[i](x)  # /4
        
        # Stage 3: C3
        c3 = x
        for i in range(3, 5):
            x = self.stages[i](x)
        c3 = x  # /8
        
        # Stage 4: C4
        for i in range(5, 11):
            x = self.stages[i](x)
        c4 = x  # /16
        
        # Stage 5: C5
        for i in range(11, 15):
            x = self.stages[i](x)
        c5 = x  # /32
        
        if self.dropout is not None:
            c3 = self.dropout(c3)
            c4 = self.dropout(c4)
            c5 = self.dropout(c5)
        
        return c3, c4, c5


if __name__ == '__main__':
    # Test GhostNet
    model = GhostNet()
    x = torch.randn(2, 3, 200, 200)
    c3, c4, c5 = model(x)
    
    print("GhostNet output shapes:")
    print(f"  C3: {c3.shape}")
    print(f"  C4: {c4.shape}")
    print(f"  C5: {c5.shape}")