import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import ImageQualityMetrics


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create a slightly modified version
    noise = np.random.randint(-10, 10, original.shape, dtype=np.int16)
    modified = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return original, modified


class TestImageQualityMetrics:
    """Test image quality metrics calculations."""
    
    def test_calculate_psnr(self, sample_images):
        """Test PSNR calculation."""
        original, modified = sample_images
        metrics = ImageQualityMetrics()
        
        psnr = metrics.calculate_psnr(original, modified)
        
        assert isinstance(psnr, float)
        assert psnr > 0
        assert psnr < 100  # Reasonable range for PSNR
    
    def test_calculate_ssim(self, sample_images):
        """Test SSIM calculation."""
        original, modified = sample_images
        metrics = ImageQualityMetrics()
        
        ssim = metrics.calculate_ssim(original, modified)
        
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1  # SSIM is between 0 and 1
    
    def test_calculate_mse(self, sample_images):
        """Test MSE calculation."""
        original, modified = sample_images
        metrics = ImageQualityMetrics()
        
        mse = metrics.calculate_mse(original, modified)
        
        assert isinstance(mse, float)
        assert mse >= 0
    
    def test_calculate_mae(self, sample_images):
        """Test MAE calculation."""
        original, modified = sample_images
        metrics = ImageQualityMetrics()
        
        mae = metrics.calculate_mae(original, modified)
        
        assert isinstance(mae, float)
        assert mae >= 0
    
    def test_calculate_sharpness(self, sample_images):
        """Test sharpness calculation."""
        original, _ = sample_images
        metrics = ImageQualityMetrics()
        
        sharpness = metrics.calculate_sharpness(original)
        
        assert isinstance(sharpness, float)
        assert sharpness >= 0
    
    def test_calculate_all_metrics(self, sample_images):
        """Test calculating all metrics at once."""
        original, modified = sample_images
        metrics = ImageQualityMetrics()
        
        results = metrics.calculate_all_metrics(original, modified)
        
        assert isinstance(results, dict)
        assert 'psnr' in results
        assert 'ssim' in results
        assert 'mse' in results
        assert 'mae' in results
        assert 'sharpness_original' in results
        assert 'sharpness_enhanced' in results
    
    def test_compare_images(self, sample_images):
        """Test comparing original, degraded, and enhanced images."""
        original, modified = sample_images
        enhanced = original.copy()  # Use original as enhanced for testing
        
        metrics = ImageQualityMetrics()
        comparison = metrics.compare_images(original, modified, enhanced)
        
        assert isinstance(comparison, dict)
        assert 'original_vs_degraded' in comparison
        assert 'original_vs_enhanced' in comparison
        assert 'degraded_vs_enhanced' in comparison
    
    def test_identical_images(self, sample_images):
        """Test metrics for identical images."""
        original, _ = sample_images
        metrics = ImageQualityMetrics()
        
        psnr = metrics.calculate_psnr(original, original)
        ssim = metrics.calculate_ssim(original, original)
        mse = metrics.calculate_mse(original, original)
        
        assert psnr > 80  # Very high PSNR for identical images
        assert ssim > 0.99  # Near perfect SSIM
        assert mse < 1  # Very low MSE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
