"""
Training module for jewelry refinement models.

This module provides:
- RefinementCNN: Lightweight residual CNN
- RefinementGAN: GAN-based refinement with PatchGAN discriminator
- Training pipeline with CNN and GAN modes
- Loss functions (L1, MSE, Adversarial, Perceptual)
- Dataset splitting utilities
"""

from .models import RefinementCNN, RefinementGAN, PatchGANDiscriminator
from .losses import RefinementLoss, GANLoss, PerceptualLoss
from .train_refinement_gan import RefinementTrainer
from .dataset_split import create_dataset_splits

__version__ = '1.0.0'

__all__ = [
    'RefinementCNN',
    'RefinementGAN',
    'PatchGANDiscriminator',
    'RefinementLoss',
    'GANLoss',
    'PerceptualLoss',
    'RefinementTrainer',
    'create_dataset_splits'
]
