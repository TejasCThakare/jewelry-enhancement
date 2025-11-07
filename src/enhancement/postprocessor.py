import cv2
import numpy as np
from typing import Optional


class JewelryPostprocessor:
    """Post-processing operations after enhancement."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize postprocessor.
        
        Args:
            config: Configuration dictionary with postprocessing parameters
        """
        self.config = config or {}
        
        # Metallic enhancement parameters
        self.metallic_params = self.config.get('metallic_enhancement', {
            'enabled': True,
            'saturation_boost': 15,
            'brightness_boost': 5
        })
        
        # Background cleanup parameters
        self.background_params = self.config.get('background_cleanup', {
            'enabled': True,
            'method': 'bilateral',
            'd': 9,
            'sigmaColor': 75,
            'sigmaSpace': 75
        })
        
        # Final sharpen parameters
        self.sharpen_params = self.config.get('final_sharpen', {
            'enabled': True,
            'amount': 0.3
        })
    
    def enhance_metallic_tones(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance gold, silver, and gemstone colors typical in jewelry.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Enhanced image
        """
        if not self.metallic_params['enabled']:
            return image
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation for more vibrant jewelry colors
        saturation_boost = self.metallic_params['saturation_boost']
        s = s + saturation_boost
        s = np.clip(s, 0, 255)
        
        # Slight brightness boost
        brightness_boost = self.metallic_params['brightness_boost']
        v = v + brightness_boost
        v = np.clip(v, 0, 255)
        
        # Merge and convert back
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def remove_background_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to reduce background noise while preserving edges.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Filtered image
        """
        if not self.background_params['enabled']:
            return image
        
        method = self.background_params['method']
        
        if method == 'bilateral':
            return cv2.bilateralFilter(
                image,
                d=self.background_params['d'],
                sigmaColor=self.background_params['sigmaColor'],
                sigmaSpace=self.background_params['sigmaSpace']
            )
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            return image
    
    def apply_final_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply subtle sharpening to enhance jewelry details.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Sharpened image
        """
        if not self.sharpen_params['enabled']:
            return image
        
        amount = self.sharpen_params['amount']
        
        # Unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)
        
        return sharpened
    
    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete postprocessing pipeline.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Postprocessed image
        """
        result = image.copy()
        
        # Step 1: Enhance metallic tones
        result = self.enhance_metallic_tones(result)
        
        # Step 2: Remove background noise
        result = self.remove_background_noise(result)
        
        # Step 3: Apply final sharpening
        result = self.apply_final_sharpen(result)
        
        return result
