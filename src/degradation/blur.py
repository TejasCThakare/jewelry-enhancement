import cv2
import numpy as np
from typing import Tuple, Optional


class BlurDegradation:
    """Apply various types of blur degradation to images."""
    
    def __init__(self, sigma_range: Tuple[float, float], motion_range: Tuple[int, int]):
        """
        Initialize blur degradation.
        
        Args:
            sigma_range: Range for Gaussian blur sigma (min, max)
            motion_range: Range for motion blur length (min, max)
        """
        self.sigma_range = sigma_range
        self.motion_range = motion_range
    
    def apply_gaussian_blur(self, image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply Gaussian blur to simulate defocus.
        
        Args:
            image: Input image in BGR format
            sigma: Blur strength, randomly sampled if None
            
        Returns:
            Blurred image
        """
        if sigma is None:
            sigma = np.random.uniform(*self.sigma_range)
        
        if sigma < 0.1:
            return image
        
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_motion_blur(self, 
                         image: np.ndarray, 
                         length: Optional[int] = None, 
                         angle: Optional[float] = None) -> np.ndarray:
        """
        Apply motion blur to simulate camera shake.
        
        Args:
            image: Input image in BGR format
            length: Motion blur length in pixels, randomly sampled if None
            angle: Motion blur angle in degrees, randomly sampled if None
            
        Returns:
            Motion blurred image
        """
        if length is None:
            length = np.random.randint(*self.motion_range)
        
        if angle is None:
            angle = np.random.uniform(0, 180)
        
        if length < 3:
            return image
        
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        kernel[int((length - 1) / 2), :] = np.ones(length)
        kernel = kernel / length
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        
        # Apply kernel
        return cv2.filter2D(image, -1, kernel)
    
    def apply_random_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random blur type (Gaussian or motion).
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Blurred image
        """
        blur_type = np.random.choice(['gaussian', 'motion', 'both'])
        
        if blur_type == 'gaussian':
            return self.apply_gaussian_blur(image)
        elif blur_type == 'motion':
            return self.apply_motion_blur(image)
        else:
            # Apply both
            image = self.apply_gaussian_blur(image)
            return self.apply_motion_blur(image)
