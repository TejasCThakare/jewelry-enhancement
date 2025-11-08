"""
Loss functions for refinement training.

Includes:
- RefinementLoss: Combined L1 + MSE loss for CNN mode
- GANLoss: BCE-based adversarial loss
- PerceptualLoss: VGG-based perceptual loss (LPIPS-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RefinementLoss(nn.Module):
    """
    Combined loss for CNN-based refinement.
    
    Loss = λ1 * L1 + λ2 * MSE
    
    Args:
        l1_weight (float): Weight for L1 loss (default: 1.0)
        mse_weight (float): Weight for MSE loss (default: 0.5)
    """
    
    def __init__(self, l1_weight=1.0, mse_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Calculate combined loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.mse_weight * mse
        
        return total_loss, {'l1': l1.item(), 'mse': mse.item()}


class GANLoss(nn.Module):
    """
    BCE-based adversarial loss for GAN training.
    
    Supports both LSGAN (MSE) and vanilla GAN (BCE) losses.
    
    Args:
        gan_mode (str): 'vanilla' or 'lsgan' (default: 'vanilla')
    """
    
    def __init__(self, gan_mode='vanilla'):
        super().__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def forward(self, prediction, target_is_real):
        """
        Calculate GAN loss.
        
        Args:
            prediction (torch.Tensor): Discriminator output
            target_is_real (bool): Whether target should be real
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        
        loss = self.loss(prediction, target)
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Compares high-level features instead of pixel values.
    More perceptually aligned than L1/MSE.
    
    Args:
        layer_weights (dict): Weights for different VGG layers
    """
    
    def __init__(self, layer_weights=None):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Default layer weights
        if layer_weights is None:
            self.layer_weights = {
                'relu1_2': 1.0,
                'relu2_2': 1.0,
                'relu3_4': 1.0,
                'relu4_4': 1.0
            }
        else:
            self.layer_weights = layer_weights
        
        # Layer indices in VGG19
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26
        }
    
    def forward(self, pred, target):
        """
        Calculate perceptual loss.
        
        Args:
            pred (torch.Tensor): Predicted image (B, 3, H, W)
            target (torch.Tensor): Target image (B, 3, H, W)
            
        Returns:
            torch.Tensor: Scalar perceptual loss
        """
        # Normalize to ImageNet stats
        pred = self._normalize(pred)
        target = self._normalize(target)
        
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            # Check if this layer should contribute to loss
            for name, idx in self.layer_indices.items():
                if i == idx and name in self.layer_weights:
                    weight = self.layer_weights[name]
                    loss += weight * F.l1_loss(x_pred, x_target)
        
        return loss
    
    def _normalize(self, x):
        """Normalize image to ImageNet stats."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Assume input is in [-1, 1], convert to [0, 1]
        x = (x + 1) / 2
        
        # Normalize
        return (x - mean) / std


class CombinedGANLoss(nn.Module):
    """
    Combined loss for GAN-based training.
    
    For Generator:
        Loss_G = λ_adv * L_adv + λ_L1 * L_L1 + λ_perc * L_perc
    
    Args:
        adv_weight (float): Weight for adversarial loss
        l1_weight (float): Weight for L1 loss
        perc_weight (float): Weight for perceptual loss
    """
    
    def __init__(self, adv_weight=1.0, l1_weight=100.0, perc_weight=10.0):
        super().__init__()
        
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight
        
        self.gan_loss = GANLoss('vanilla')
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, fake, real, discriminator_output):
        """
        Calculate combined GAN loss for generator.
        
        Args:
            fake (torch.Tensor): Generated image
            real (torch.Tensor): Target image
            discriminator_output (torch.Tensor): D(fake)
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Adversarial loss
        adv_loss = self.gan_loss(discriminator_output, True)
        
        # L1 loss
        l1 = self.l1_loss(fake, real)
        
        # Perceptual loss
        perc = self.perceptual_loss(fake, real)
        
        # Combined
        total_loss = (self.adv_weight * adv_loss + 
                     self.l1_weight * l1 + 
                     self.perc_weight * perc)
        
        loss_dict = {
            'adv': adv_loss.item(),
            'l1': l1.item(),
            'perceptual': perc.item()
        }
        
        return total_loss, loss_dict
