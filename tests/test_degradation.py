import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.degradation.blur import BlurDegradation
from src.degradation.noise import NoiseDegradation
from src.degradation.compression import CompressionDegradation
from src.degradation.color_shift import ColorShiftDegradation
from src.degradation.pipeline import DegradationPipeline


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


class TestBlurDegradation:
    """Test blur degradation operations."""
    
    def test_gaussian_blur(self, sample_image):
        """Test Gaussian blur application."""
        blur = BlurDegradation(sigma_range=(0.5, 2.0), motion_range=(3, 10))
        blurred = blur.apply_gaussian_blur(sample_image, sigma=1.5)
        
        assert blurred.shape == sample_image.shape
        assert blurred.dtype == sample_image.dtype
        assert not np.array_equal(blurred, sample_image)
    
    def test_motion_blur(self, sample_image):
        """Test motion blur application."""
        blur = BlurDegradation(sigma_range=(0.5, 2.0), motion_range=(3, 10))
        blurred = blur.apply_motion_blur(sample_image, length=5, angle=45)
        
        assert blurred.shape == sample_image.shape
        assert blurred.dtype == sample_image.dtype
    
    def test_random_blur(self, sample_image):
        """Test random blur application."""
        blur = BlurDegradation(sigma_range=(0.5, 2.0), motion_range=(3, 10))
        blurred = blur.apply_random_blur(sample_image)
        
        assert blurred.shape == sample_image.shape
        assert blurred.dtype == sample_image.dtype


class TestNoiseDegradation:
    """Test noise degradation operations."""
    
    def test_gaussian_noise(self, sample_image):
        """Test Gaussian noise application."""
        noise = NoiseDegradation(gaussian_range=(10, 30), sp_range=(0.001, 0.01))
        noisy = noise.apply_gaussian_noise(sample_image, sigma=20)
        
        assert noisy.shape == sample_image.shape
        assert noisy.dtype == sample_image.dtype
        assert not np.array_equal(noisy, sample_image)
    
    def test_salt_pepper_noise(self, sample_image):
        """Test salt-and-pepper noise application."""
        noise = NoiseDegradation(gaussian_range=(10, 30), sp_range=(0.001, 0.01))
        noisy = noise.apply_salt_pepper_noise(sample_image, amount=0.005)
        
        assert noisy.shape == sample_image.shape
        assert noisy.dtype == sample_image.dtype
    
    def test_poisson_noise(self, sample_image):
        """Test Poisson noise application."""
        noise = NoiseDegradation(gaussian_range=(10, 30), sp_range=(0.001, 0.01))
        noisy = noise.apply_poisson_noise(sample_image)
        
        assert noisy.shape == sample_image.shape
        assert noisy.dtype == sample_image.dtype


class TestCompressionDegradation:
    """Test compression degradation operations."""
    
    def test_jpeg_compression(self, sample_image):
        """Test JPEG compression application."""
        compression = CompressionDegradation(quality_range=(20, 80))
        compressed = compression.apply_jpeg_compression(sample_image, quality=50)
        
        assert compressed.shape == sample_image.shape
        assert compressed.dtype == sample_image.dtype
    
    def test_multiple_compressions(self, sample_image):
        """Test multiple JPEG compression cycles."""
        compression = CompressionDegradation(quality_range=(20, 80))
        compressed = compression.apply_multiple_compressions(sample_image, num_iterations=3)
        
        assert compressed.shape == sample_image.shape
        assert compressed.dtype == sample_image.dtype


class TestColorShiftDegradation:
    """Test color shift degradation operations."""
    
    def test_temperature_shift(self, sample_image):
        """Test color temperature shift."""
        color_shift = ColorShiftDegradation(
            temperature_range=(-20, 20),
            saturation_range=(-10, 10)
        )
        shifted = color_shift.apply_color_temperature_shift(sample_image, delta=10)
        
        assert shifted.shape == sample_image.shape
        assert shifted.dtype == sample_image.dtype
    
    def test_saturation_shift(self, sample_image):
        """Test saturation shift."""
        color_shift = ColorShiftDegradation(
            temperature_range=(-20, 20),
            saturation_range=(-10, 10)
        )
        shifted = color_shift.apply_saturation_shift(sample_image, delta=5)
        
        assert shifted.shape == sample_image.shape
        assert shifted.dtype == sample_image.dtype
    
    def test_random_color_shift(self, sample_image):
        """Test random color shift."""
        color_shift = ColorShiftDegradation(
            temperature_range=(-20, 20),
            saturation_range=(-10, 10)
        )
        shifted = color_shift.apply_random_color_shift(sample_image)
        
        assert shifted.shape == sample_image.shape
        assert shifted.dtype == sample_image.dtype


class TestDegradationPipeline:
    """Test complete degradation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with config."""
        pipeline = DegradationPipeline("config/degradation_config.yaml")
        assert pipeline.config is not None
    
    def test_mild_degradation(self, sample_image):
        """Test mild degradation level."""
        pipeline = DegradationPipeline("config/degradation_config.yaml")
        degraded = pipeline.apply_degradation(sample_image, "level1_mild")
        
        assert degraded.shape == sample_image.shape
        assert degraded.dtype == sample_image.dtype
    
    def test_moderate_degradation(self, sample_image):
        """Test moderate degradation level."""
        pipeline = DegradationPipeline("config/degradation_config.yaml")
        degraded = pipeline.apply_degradation(sample_image, "level2_moderate")
        
        assert degraded.shape == sample_image.shape
        assert degraded.dtype == sample_image.dtype
    
    def test_severe_degradation(self, sample_image):
        """Test severe degradation level."""
        pipeline = DegradationPipeline("config/degradation_config.yaml")
        degraded = pipeline.apply_degradation(sample_image, "level3_severe")
        
        assert degraded.shape == sample_image.shape
        assert degraded.dtype == sample_image.dtype
    
    def test_invalid_level(self, sample_image):
        """Test error handling for invalid degradation level."""
        pipeline = DegradationPipeline("config/degradation_config.yaml")
        
        with pytest.raises(ValueError):
            pipeline.apply_degradation(sample_image, "invalid_level")
    
    def test_get_level_description(self):
        """Test getting degradation level description."""
        pipeline = DegradationPipeline("config/degradation_config.yaml")
        description = pipeline.get_level_description("level1_mild")
        
        assert isinstance(description, str)
        assert len(description) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
