import cv2
import numpy as np
from typing import Tuple, Optional


class ColorShiftDegradation:
    """Apply color temperature and saturation shifts."""
    
    def __init__(self, temperature_range: Tuple[int, int], saturation_range: Tuple[int, int]):
        """
        Initialize color shift degradation.
        
        Args:
            temperature_range: Range for temperature shift (min, max)
            saturation_range: Range for saturation shift (min, max)
        """
        self.temperature_range = temperature_range
        self.saturation_range = saturation_range
    
    def apply_color_temperature_shift(self, image: np.ndarray, delta: Optional[int] = None) -> np.ndarray:
        """
        Apply color temperature shift to simulate incorrect white balance.
        
        Args:
            image: Input image in BGR format
            delta: Temperature shift amount, randomly sampled if None
            
        Returns:
            Color-shifted image
        """
        if delta is None:
            delta = np.random.randint(*self.temperature_range)
        
        if abs(delta) < 1:
            return image
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Apply temperature shift to b channel (blue-yellow)
        b = b + delta
        b = np.clip(b, 0, 255)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b]).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def apply_saturation_shift(self, image: np.ndarray, delta: Optional[int] = None) -> np.ndarray:
        """
        Apply saturation shift.
        
        Args:
            image: Input image in BGR format
            delta: Saturation shift amount, randomly sampled if None
            
        Returns:
            Saturation-shifted image
        """
        if delta is None:
            delta = np.random.randint(*self.saturation_range)
        
        if abs(delta) < 1:
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Apply saturation shift
        s = s + delta
        s = np.clip(s, 0, 255)
        
        # Merge and convert back
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def apply_random_color_shift(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random color shifts.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Color-shifted image
        """
        # Apply temperature shift
        image = self.apply_color_temperature_shift(image)
        
        # Apply saturation shift
        image = self.apply_saturation_shift(image)
        
        return image
