"""
Model architectures for jewelry refinement.

Includes:
- RefinementCNN: Lightweight residual U-Net style architecture
- RefinementGAN: Generator for GAN-based refinement
- PatchGANDiscriminator: 70x70 PatchGAN discriminator
"""

import torch
import torch.nn as nn


class RefinementCNN(nn.Module):
    """
    Lightweight residual CNN for jewelry image refinement.
    
    Architecture:
        - 3 encoder blocks (conv + pool)
        - Bottleneck with residual connections
        - 3 decoder blocks (upsample + conv)
        - Skip connections from encoder to decoder
        - Residual output prediction
    
    Args:
        in_channels (int): Input channels (default: 3 for RGB)
        out_channels (int): Output channels (default: 3 for RGB)
        features (int): Base number of features (default: 64)
    
    Input Shape:  (B, 3, H, W)
    Output Shape: (B, 3, H, W)
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, features, pool=True)
        self.enc2 = self._conv_block(features, features*2, pool=True)
        self.enc3 = self._conv_block(features*2, features*4, pool=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*4, features*4, 3, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*4, features*4, 3, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = self._upconv_block(features*4, features*2)
        self.dec2 = self._upconv_block(features*2, features)
        self.dec1 = self._upconv_block(features, features)
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(features, out_channels, 1),
            nn.Tanh()
        )
        
        # Residual scaling factor
        self.residual_scale = 0.1
    
    def _conv_block(self, in_c, out_c, pool=False):
        """Create convolutional block."""
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_c, out_c):
        """Create upsampling block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass with residual learning.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Refined image (B, C, H, W)
        """
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decode
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        # Predict residual
        residual = self.out(d1)
        
        # Add scaled residual to input
        output = x + residual * self.residual_scale
        
        return output


class RefinementGAN(nn.Module):
    """
    GAN-based generator for jewelry refinement.
    
    Similar to RefinementCNN but optimized for adversarial training.
    Uses instance normalization instead of batch normalization.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        features (int): Base number of features
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, features)
        self.enc2 = self._conv_block(features, features*2)
        self.enc3 = self._conv_block(features*2, features*4)
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4)
        )
        
        # Decoder
        self.dec3 = self._upconv_block(features*4, features*2)
        self.dec2 = self._upconv_block(features*2, features)
        self.dec1 = self._upconv_block(features, features)
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(features, out_channels, 7, padding=3),
            nn.Tanh()
        )
        
        self.residual_scale = 0.2
    
    def _conv_block(self, in_c, out_c):
        """Encoder block with instance norm."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _upconv_block(self, in_c, out_c):
        """Decoder block with instance norm."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass."""
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decode
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        # Output
        residual = self.out(d1)
        output = x + residual * self.residual_scale
        
        return output


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.
    
    Classifies whether 70x70 patches are real or fake.
    More stable than full-image discriminator.
    
    Args:
        in_channels (int): Input channels (default: 3)
        features (int): Base number of features (default: 64)
    
    Input Shape:  (B, 3, H, W)
    Output Shape: (B, 1, H/16, W/16) - patch predictions
    """
    
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Layer 1: (B, 3, H, W) -> (B, 64, H/2, W/2)
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: -> (B, 128, H/4, W/4)
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.InstanceNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: -> (B, 256, H/8, W/8)
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.InstanceNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: -> (B, 512, H/16, W/16)
            nn.Conv2d(features*4, features*8, 4, 1, 1),
            nn.InstanceNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: -> (B, 1, H/16, W/16)
            nn.Conv2d(features*8, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Patch predictions (B, 1, H/16, W/16)
        """
        return self.model(x)
