import cv2
import numpy as np
from typing import Tuple, Optional


class NoiseDegradation:
    """Apply various types of noise degradation to images."""
    
    def __init__(self, gaussian_range: Tuple[float, float], sp_range: Tuple[float, float]):
        """
        Initialize noise degradation.
        
        Args:
            gaussian_range: Range for Gaussian noise sigma (min, max)
            sp_range: Range for salt-and-pepper noise amount (min, max)
        """
        self.gaussian_range = gaussian_range
        self.sp_range = sp_range
    
    def apply_gaussian_noise(self, image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply Gaussian noise to simulate sensor noise.
        
        Args:
            image: Input image in BGR format
            sigma: Noise strength, randomly sampled if None
            
        Returns:
            Noisy image
        """
        if sigma is None:
            sigma = np.random.uniform(*self.gaussian_range)
        
        if sigma < 1.0:
            return image
        
        noise = np.random.randn(*image.shape) * sigma
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def apply_salt_pepper_noise(self, image: np.ndarray, amount: Optional[float] = None) -> np.ndarray:
        """
        Apply salt-and-pepper noise.
        
        Args:
            image: Input image in BGR format
            amount: Noise density, randomly sampled if None
            
        Returns:
            Noisy image
        """
        if amount is None:
            amount = np.random.uniform(*self.sp_range)
        
        if amount < 0.0001:
            return image
        
        noisy = image.copy()
        
        # Salt noise (white pixels)
        num_salt = int(np.ceil(amount * image.size * 0.5))
        coords = tuple(np.random.randint(0, i - 1, num_salt) for i in image.shape[:2])
        noisy[coords] = 255
        
        # Pepper noise (black pixels)
        num_pepper = int(np.ceil(amount * image.size * 0.5))
        coords = tuple(np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2])
        noisy[coords] = 0
        
        return noisy
    
    def apply_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Poisson noise to simulate photon counting statistics.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Noisy image
        """
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply Poisson noise
        noisy = np.random.poisson(normalized * 255.0) / 255.0
        
        # Convert back to uint8
        return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    
    def apply_random_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random noise type.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Noisy image
        """
        noise_type = np.random.choice(['gaussian', 'salt_pepper', 'both'])
        
        if noise_type == 'gaussian':
            return self.apply_gaussian_noise(image)
        elif noise_type == 'salt_pepper':
            return self.apply_salt_pepper_noise(image)
        else:
            # Apply both
            image = self.apply_gaussian_noise(image)
            return self.apply_salt_pepper_noise(image)
