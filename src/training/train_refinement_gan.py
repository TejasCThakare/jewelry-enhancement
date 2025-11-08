"""
Main training script for jewelry refinement models.

Supports two modes:
- CNN mode: Train RefinementCNN with L1+MSE loss
- GAN mode: Train RefinementGAN with adversarial+L1+perceptual loss

Usage:
    python src/training/train_refinement_gan.py --mode cnn --epochs 100
    python src/training/train_refinement_gan.py --mode gan --epochs 200
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml
import json
from datetime import datetime

from src.training.models import RefinementCNN, RefinementGAN, PatchGANDiscriminator
from src.training.losses import RefinementLoss, GANLoss, CombinedGANLoss
from src.utils.image_io import load_image


class JewelryDataset(torch.utils.data.Dataset):
    """Dataset for loading training pairs - FIXED for variable sizes."""
    
    def __init__(self, data_dir, split='train', target_size=512):
        import numpy as np
        
        self.data_dir = Path(data_dir) / split
        self.target_size = target_size  # Resize all images to this size
        self.files = sorted(list(self.data_dir.glob('*_input.npy')))
        
        if len(self.files) == 0:
            raise ValueError(f"No data found in {self.data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        import numpy as np
        import cv2
        
        input_path = self.files[idx]
        target_path = str(input_path).replace('_input.npy', '_target.npy')
        
        # Load
        input_img = np.load(input_path)
        target_img = np.load(target_path)
        
        # CRITICAL FIX: Resize to fixed size
        if input_img.shape[:2] != (self.target_size, self.target_size):
            input_img = cv2.resize(input_img, (self.target_size, self.target_size), 
                                  interpolation=cv2.INTER_CUBIC)
        
        if target_img.shape[:2] != (self.target_size, self.target_size):
            target_img = cv2.resize(target_img, (self.target_size, self.target_size), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [-1, 1]
        input_img = (input_img / 127.5) - 1.0
        target_img = (target_img / 127.5) - 1.0
        
        # To tensor (C, H, W)
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float()
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float()
        
        return input_tensor, target_tensor



class RefinementTrainer:
    """
    Unified trainer for CNN and GAN modes.
    
    Args:
        config (dict): Training configuration
        mode (str): 'cnn' or 'gan'
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, config, mode='cnn', device='cuda'):
        self.config = config
        self.mode = mode
        self.device = device
        self.use_mixed_precision = config.get('mixed_precision', False)
        
        # Create models
        if mode == 'cnn':
            self.generator = RefinementCNN(
                features=config['model']['features']
            ).to(device)
            self.discriminator = None
        else:  # gan
            self.generator = RefinementGAN(
                features=config['model']['features']
            ).to(device)
            self.discriminator = PatchGANDiscriminator().to(device)
        
        # Create optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['training']['lr_g'],
            betas=(0.5, 0.999)
        )

        
        
        if mode == 'gan':
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=config['training']['lr_d'],
                betas=(0.5, 0.999)
            )
        
        # Create losses
        if mode == 'cnn':
            self.criterion = RefinementLoss(
                l1_weight=config['loss']['l1_weight'],
                mse_weight=config['loss']['mse_weight']
            )
        else:  # gan
            self.criterion_G = CombinedGANLoss(
                adv_weight=config['loss']['adv_weight'],
                l1_weight=config['loss']['l1_weight'],
                perc_weight=config['loss']['perc_weight'],
                device=device
            )
            self.criterion_D = GANLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Tracking
        self.best_psnr = 0
        self.current_epoch = 0
        
        # TensorBoard
        log_dir = Path('results/logs') / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        print(f"✓ Initialized {mode.upper()} trainer")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print(f"  Logs: {log_dir}")
    
    def train_epoch_cnn(self, train_loader, epoch):
        """Train one epoch in CNN mode."""
        self.generator.train()
        epoch_loss = 0
        loss_dict_sum = {'l1': 0, 'mse': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.generator(inputs)
                    loss, loss_dict = self.criterion(outputs, targets)
                
                # Backward
                self.optimizer_G.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                outputs = self.generator(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()
            
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_dict = {k: v/len(train_loader) for k, v in loss_dict_sum.items()}
        
        return avg_loss, avg_loss_dict
    
    def train_epoch_gan(self, train_loader, epoch):
        """Train one epoch in GAN mode."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_loss_G = 0
        epoch_loss_D = 0
        loss_dict_sum = {'adv': 0, 'l1': 0, 'perceptual': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            # ==================
            # Train Discriminator
            # ==================
            self.optimizer_D.zero_grad()
            
            # Real images
            real_output = self.discriminator(targets)
            loss_D_real = self.criterion_D(real_output, True)
            
            # Fake images
            with torch.no_grad():
                fake = self.generator(inputs)
            fake_output = self.discriminator(fake.detach())
            loss_D_fake = self.criterion_D(fake_output, False)
            
            # Combined D loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer_D.step()
            
            # ==================
            # Train Generator
            # ==================
            self.optimizer_G.zero_grad()
            
            fake = self.generator(inputs)
            fake_output = self.discriminator(fake)
            
            # Combined G loss
            loss_G, loss_dict = self.criterion_G(fake, targets, fake_output)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer_G.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v
            
            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}'
            })
        
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        avg_loss_dict = {k: v/len(train_loader) for k, v in loss_dict_sum.items()}
        
        return avg_loss_G, avg_loss_D, avg_loss_dict
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.generator.eval()
        
        from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        
        psnr_metric = PeakSignalNoiseRatio().to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.generator(inputs)
                
                # Update metrics
                psnr_metric.update(outputs, targets)
                ssim_metric.update(outputs, targets)
        
        psnr = psnr_metric.compute().item()
        ssim = ssim_metric.compute().item()
        
        return psnr, ssim
    
    def train(self, train_loader, val_loader, epochs, save_dir='models'):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"TRAINING {self.mode.upper()} MODE")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print("="*70)
        print()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            if self.mode == 'cnn':
                avg_loss, loss_dict = self.train_epoch_cnn(train_loader, epoch)
                
                # Log
                self.writer.add_scalar('Loss/train', avg_loss, epoch)
                self.writer.add_scalar('Loss/L1', loss_dict['l1'], epoch)
                self.writer.add_scalar('Loss/MSE', loss_dict['mse'], epoch)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} "
                      f"(L1: {loss_dict['l1']:.4f}, MSE: {loss_dict['mse']:.4f})")
            
            else:  # gan
                avg_loss_G, avg_loss_D, loss_dict = self.train_epoch_gan(train_loader, epoch)
                
                # Log
                self.writer.add_scalar('Loss/G', avg_loss_G, epoch)
                self.writer.add_scalar('Loss/D', avg_loss_D, epoch)
                self.writer.add_scalar('Loss/Adversarial', loss_dict['adv'], epoch)
                self.writer.add_scalar('Loss/L1', loss_dict['l1'], epoch)
                self.writer.add_scalar('Loss/Perceptual', loss_dict['perceptual'], epoch)
                
                print(f"Epoch {epoch+1}/{epochs} - G: {avg_loss_G:.4f}, D: {avg_loss_D:.4f}")
            
            # Validate
            if (epoch + 1) % self.config['training']['val_every'] == 0:
                psnr, ssim = self.validate(val_loader)
                
                self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
                self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
                
                print(f"  Validation - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
                # Save best model
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    self.save_checkpoint(save_path / f'{self.mode}_best.pth', epoch, psnr)
                    print(f"  ✓ Saved best model (PSNR: {psnr:.2f} dB)")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(save_path / f'{self.mode}_epoch_{epoch+1}.pth', epoch)
        
        self.writer.close()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Models saved to: {save_path}")
        print("="*70)
    
    def save_checkpoint(self, path, epoch, psnr=None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        if self.mode == 'gan':
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
        
        if psnr is not None:
            checkpoint['psnr'] = psnr
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.best_psnr = checkpoint['best_psnr']
        self.current_epoch = checkpoint['epoch']
        
        if self.mode == 'gan':
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description='Train jewelry refinement model')
    parser.add_argument('--mode', type=str, required=True, choices=['cnn', 'gan'],
                       help='Training mode: cnn or gan')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Training config file')
    parser.add_argument('--data', type=str, default='data/training',
                       help='Training data directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override config with args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.mixed_precision:
        config['mixed_precision'] = True
    
    # Device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = JewelryDataset(args.data, split='train')
    val_dataset = JewelryDataset(args.data, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Trainer
    trainer = RefinementTrainer(config, mode=args.mode, device=device)
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=config['training']['epochs'],
        save_dir='models'
    )


if __name__ == '__main__':
    main()
