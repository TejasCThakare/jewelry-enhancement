import cv2
import numpy as np
from typing import Optional


class JewelryPreprocessor:
    """Preprocessing operations before enhancement."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        
        # Default denoise parameters
        self.denoise_params = self.config.get('denoise', {
            'h': 10,
            'hColor': 10,
            'templateWindowSize': 7,
            'searchWindowSize': 21
        })
        
        # Default CLAHE parameters
        self.clahe_params = self.config.get('contrast', {
            'clipLimit': 3.0,
            'tileGridSize': [8, 8]
        })
        
        # Sharpen parameters
        self.sharpen_params = self.config.get('sharpen', {
            'enabled': False,
            'kernel_size': 3,
            'sigma': 1.0
        })
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.denoise_params['h'],
            hColor=self.denoise_params['hColor'],
            templateWindowSize=self.denoise_params['templateWindowSize'],
            searchWindowSize=self.denoise_params['searchWindowSize']
        )
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_params['clipLimit'],
            tileGridSize=tuple(self.clahe_params['tileGridSize'])
        )
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Sharpened image
        """
        if not self.sharpen_params['enabled']:
            return image
        
        # Unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), self.sharpen_params['sigma'])
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    def preprocess(self, image: np.ndarray, denoise: bool = True, 
                   enhance_contrast: bool = True, sharpen: bool = False) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image: Input image in BGR format
            denoise: Whether to apply denoising
            enhance_contrast: Whether to enhance contrast
            sharpen: Whether to apply sharpening
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        if denoise:
            result = self.denoise(result)
        
        if enhance_contrast:
            result = self.enhance_contrast(result)
        
        if sharpen:
            result = self.sharpen(result)
        
        return result
