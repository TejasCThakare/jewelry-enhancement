import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.enhancement.preprocessor import JewelryPreprocessor
from src.enhancement.postprocessor import JewelryPostprocessor


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


class TestJewelryPreprocessor:
    """Test preprocessing operations."""
    
    def test_denoise(self, sample_image):
        """Test denoising operation."""
        preprocessor = JewelryPreprocessor()
        denoised = preprocessor.denoise(sample_image)
        
        assert denoised.shape == sample_image.shape
        assert denoised.dtype == sample_image.dtype
    
    def test_enhance_contrast(self, sample_image):
        """Test contrast enhancement."""
        preprocessor = JewelryPreprocessor()
        enhanced = preprocessor.enhance_contrast(sample_image)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
    
    def test_sharpen(self, sample_image):
        """Test sharpening operation."""
        config = {'sharpen': {'enabled': True, 'kernel_size': 3, 'sigma': 1.0}}
        preprocessor = JewelryPreprocessor(config)
        sharpened = preprocessor.sharpen(sample_image)
        
        assert sharpened.shape == sample_image.shape
        assert sharpened.dtype == sample_image.dtype
    
    def test_full_preprocessing(self, sample_image):
        """Test complete preprocessing pipeline."""
        preprocessor = JewelryPreprocessor()
        processed = preprocessor.preprocess(
            sample_image,
            denoise=True,
            enhance_contrast=True,
            sharpen=False
        )
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == sample_image.dtype


class TestJewelryPostprocessor:
    """Test postprocessing operations."""
    
    def test_enhance_metallic_tones(self, sample_image):
        """Test metallic tone enhancement."""
        postprocessor = JewelryPostprocessor()
        enhanced = postprocessor.enhance_metallic_tones(sample_image)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
    
    def test_remove_background_noise(self, sample_image):
        """Test background noise removal."""
        postprocessor = JewelryPostprocessor()
        filtered = postprocessor.remove_background_noise(sample_image)
        
        assert filtered.shape == sample_image.shape
        assert filtered.dtype == sample_image.dtype
    
    def test_apply_final_sharpen(self, sample_image):
        """Test final sharpening."""
        postprocessor = JewelryPostprocessor()
        sharpened = postprocessor.apply_final_sharpen(sample_image)
        
        assert sharpened.shape == sample_image.shape
        assert sharpened.dtype == sample_image.dtype
    
    def test_full_postprocessing(self, sample_image):
        """Test complete postprocessing pipeline."""
        postprocessor = JewelryPostprocessor()
        processed = postprocessor.postprocess(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == sample_image.dtype


class TestEnhancerInitialization:
    """Test enhancer initialization (without loading full model)."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        import yaml
        
        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'model' in config
        assert 'inference' in config
        assert config['model']['scale'] in [2, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
