"""
Image preprocessing module for jewelry enhancement.
UPDATED: Minimal preprocessing - let Real-ESRGAN handle it
"""

import cv2
import numpy as np
from typing import Optional


class JewelryPreprocessor:
    """Handles jewelry-specific image preprocessing - MINIMAL VERSION."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize preprocessor with minimal settings."""
        self.config = config or {}
        
        # Minimal denoise settings (disabled by default)
        self.denoise_params = self.config.get('denoise', {
            'enabled': False,
            'h': 10,
            'templateWindowSize': 7,
            'searchWindowSize': 21
        })
        
        # Minimal contrast enhancement (disabled)
        self.clahe_params = self.config.get('clahe', {
            'enabled': False,
            'clipLimit': 2.0,
            'tileGridSize': (8, 8)
        })
    
    def preprocess(self, image: np.ndarray, denoise: bool = False, 
                   enhance_contrast: bool = False, sharpen: bool = False) -> np.ndarray:
        """
        Apply minimal preprocessing - just return image as-is.
        Real-ESRGAN does its own preprocessing internally.
        
        Args:
            image: Input image in BGR format
            denoise: Ignored - no denoising applied
            enhance_contrast: Ignored - no contrast enhancement
            sharpen: Ignored - no sharpening
            
        Returns:
            Original image unchanged
        """
        return image
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """No denoising - return image as-is."""
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """No contrast enhancement - return image as-is."""
        return image
    
    def sharpen(self, image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """No sharpening - return image as-is."""
        return image
