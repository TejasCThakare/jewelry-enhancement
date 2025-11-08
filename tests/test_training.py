"""
Unit tests for training module.

Tests:
- Model initialization
- Forward pass
- Loss calculation
- Checkpoint saving/loading
- Training step
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import torch
import tempfile
import yaml

from src.training.models import RefinementCNN, RefinementGAN, PatchGANDiscriminator
from src.training.losses import RefinementLoss, GANLoss, PerceptualLoss
from src.training.train_refinement_gan import RefinementTrainer


class TestModels(unittest.TestCase):
    """Test model architectures."""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 2
        self.img_size = 256
    
    def test_refinement_cnn(self):
        """Test RefinementCNN forward pass."""
        model = RefinementCNN().to(self.device)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        output = model(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_refinement_gan(self):
        """Test RefinementGAN forward pass."""
        model = RefinementGAN().to(self.device)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        output = model(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_discriminator(self):
        """Test PatchGANDiscriminator."""
        model = PatchGANDiscriminator().to(self.device)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        output = model(x)
        
        # Output should be smaller than input (patch predictions)
        self.assertEqual(len(output.shape), 4)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 1)


class TestLosses(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pred = torch.randn(2, 3, 64, 64).to(self.device)
        self.target = torch.randn(2, 3, 64, 64).to(self.device)
    
    def test_refinement_loss(self):
        """Test combined L1+MSE loss."""
        loss_fn = RefinementLoss()
        loss, loss_dict = loss_fn(self.pred, self.target)
        
        self.assertIsInstance(loss.item(), float)
        self.assertIn('l1', loss_dict)
        self.assertIn('mse', loss_dict)
    
    def test_gan_loss(self):
        """Test adversarial loss."""
        loss_fn = GANLoss('vanilla')
        pred = torch.randn(2, 1, 16, 16).to(self.device)
        
        loss_real = loss_fn(pred, True)
        loss_fake = loss_fn(pred, False)
        
        self.assertIsInstance(loss_real.item(), float)
        self.assertIsInstance(loss_fake.item(), float)
    
    def test_perceptual_loss(self):
        """Test VGG-based perceptual loss."""
        loss_fn = PerceptualLoss().to(self.device)
        loss = loss_fn(self.pred, self.target)
        
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss))


class TestTraining(unittest.TestCase):
    """Test training pipeline."""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Minimal config
        self.config = {
            'model': {'features': 32},
            'training': {
                'lr_g': 0.0001,
                'lr_d': 0.0004,
                'val_every': 1,
                'save_every': 1
            },
            'loss': {
                'l1_weight': 1.0,
                'mse_weight': 0.5,
                'adv_weight': 1.0,
                'perc_weight': 10.0
            },
            'mixed_precision': False
        }
    
    def test_trainer_initialization_cnn(self):
        """Test CNN trainer initialization."""
        trainer = RefinementTrainer(self.config, mode='cnn', device=self.device)
        
        self.assertIsNotNone(trainer.generator)
        self.assertIsNone(trainer.discriminator)
        self.assertIsNotNone(trainer.optimizer_G)
    
    def test_trainer_initialization_gan(self):
        """Test GAN trainer initialization."""
        trainer = RefinementTrainer(self.config, mode='gan', device=self.device)
        
        self.assertIsNotNone(trainer.generator)
        self.assertIsNotNone(trainer.discriminator)
        self.assertIsNotNone(trainer.optimizer_G)
        self.assertIsNotNone(trainer.optimizer_D)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        trainer = RefinementTrainer(self.config, mode='cnn', device=self.device)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pth'
            
            # Save
            trainer.save_checkpoint(checkpoint_path, epoch=0, psnr=25.0)
            self.assertTrue(checkpoint_path.exists())
            
            # Load
            epoch = trainer.load_checkpoint(checkpoint_path)
            self.assertEqual(epoch, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestLosses))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
