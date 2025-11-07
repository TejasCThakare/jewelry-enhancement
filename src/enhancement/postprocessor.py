"""
Image postprocessing module for jewelry enhancement.
UPDATED: Minimal postprocessing - Real-ESRGAN output is already optimal
"""

import cv2
import numpy as np
from typing import Optional


class JewelryPostprocessor:
    """Handles jewelry-specific image postprocessing - MINIMAL VERSION."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize postprocessor with disabled settings."""
        self.config = config or {}
        
        # ALL DISABLED - Real-ESRGAN output is already good
        self.metallic_params = {
            'enabled': False,
            'saturation_boost': 15,
            'brightness_boost': 5
        }
        
        self.background_params = {
            'enabled': False,
            'method': 'bilateral',
            'd': 9,
            'sigmaColor': 75,
            'sigmaSpace': 75
        }
        
        self.sharpen_params = {
            'enabled': False,
            'amount': 0.3
        }
    
    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply minimal postprocessing - just return image as-is.
        Real-ESRGAN output is already optimal.
        
        Args:
            image: Enhanced image from Real-ESRGAN in BGR format
            
        Returns:
            Original enhanced image unchanged
        """
        return image
    
    def enhance_metallic_tones(self, image: np.ndarray) -> np.ndarray:
        """No enhancement - return image as-is."""
        return image
    
    def remove_background_noise(self, image: np.ndarray) -> np.ndarray:
        """No noise removal - return image as-is."""
        return image
    
    def sharpen(self, image: np.ndarray, amount: float = 0.3) -> np.ndarray:
        """No sharpening - return image as-is."""
        return image
