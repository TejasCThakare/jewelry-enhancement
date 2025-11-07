import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional

from .blur import BlurDegradation
from .noise import NoiseDegradation
from .compression import CompressionDegradation
from .color_shift import ColorShiftDegradation


class DegradationPipeline:
    """Complete degradation pipeline for simulating real-world image quality loss."""
    
    def __init__(self, config_path: str):
        """
        Initialize degradation pipeline.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed if specified
        if 'pipeline' in self.config and 'seed' in self.config['pipeline']:
            np.random.seed(self.config['pipeline']['seed'])
    
    def _should_apply(self, probability: float) -> bool:
        """Determine if degradation should be applied based on probability."""
        return np.random.rand() < probability
    
    def _downscale_image(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Downscale image if specified in config.
        
        Args:
            image: Input image
            config: Degradation level config
            
        Returns:
            Downscaled image
        """
        if 'downscale' not in config:
            return image
        
        factor_range = config['downscale']['factor']
        if factor_range[0] >= 1.0:
            return image
        
        factor = np.random.uniform(*factor_range)
        method = config['downscale'].get('method', 'bicubic')
        
        new_width = int(image.shape[1] * factor)
        new_height = int(image.shape[0] * factor)
        
        interpolation = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }.get(method, cv2.INTER_CUBIC)
        
        downscaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Upscale back to original size
        return cv2.resize(downscaled, (image.shape[1], image.shape[0]), interpolation=interpolation)
    
    def apply_degradation(self, image: np.ndarray, level: str) -> np.ndarray:
        """
        Apply complete degradation pipeline.
        
        Args:
            image: Input image in BGR format
            level: Degradation level key (e.g., 'level1_mild', 'level2_moderate', 'level3_severe')
            
        Returns:
            Degraded image
        """
        if level not in self.config['degradation_levels']:
            raise ValueError(f"Unknown degradation level: {level}")
        
        level_config = self.config['degradation_levels'][level]
        degraded = image.copy()
        
        # Step 1: Apply blur
        if 'blur' in level_config and self._should_apply(level_config['blur'].get('apply_probability', 1.0)):
            blur_degrader = BlurDegradation(
                sigma_range=tuple(level_config['blur']['gaussian_sigma']),
                motion_range=tuple(level_config['blur']['motion_length'])
            )
            degraded = blur_degrader.apply_random_blur(degraded)
        
        # Step 2: Downscale (simulates low resolution)
        degraded = self._downscale_image(degraded, level_config)
        
        # Step 3: Apply noise
        if 'noise' in level_config and self._should_apply(level_config['noise'].get('apply_probability', 1.0)):
            noise_degrader = NoiseDegradation(
                gaussian_range=tuple(level_config['noise']['gaussian_sigma']),
                sp_range=tuple(level_config['noise']['salt_pepper_amount'])
            )
            degraded = noise_degrader.apply_random_noise(degraded)
        
        # Step 4: Apply compression
        if 'compression' in level_config and self._should_apply(level_config['compression'].get('apply_probability', 1.0)):
            compression_degrader = CompressionDegradation(
                quality_range=tuple(level_config['compression']['jpeg_quality'])
            )
            degraded = compression_degrader.apply_jpeg_compression(degraded)
        
        # Step 5: Apply color shift
        if 'color_shift' in level_config and self._should_apply(level_config['color_shift'].get('apply_probability', 1.0)):
            color_degrader = ColorShiftDegradation(
                temperature_range=tuple(level_config['color_shift']['temperature_delta']),
                saturation_range=tuple(level_config['color_shift'].get('saturation_delta', [-5, 5]))
            )
            degraded = color_degrader.apply_random_color_shift(degraded)
        
        return degraded
    
    def get_level_description(self, level: str) -> str:
        """Get description of degradation level."""
        if level not in self.config['degradation_levels']:
            return "Unknown level"
        return self.config['degradation_levels'][level].get('description', 'No description available')
