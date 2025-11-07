import cv2
import numpy as np
from typing import Tuple, Optional
from io import BytesIO


class CompressionDegradation:
    """Apply JPEG compression artifacts."""
    
    def __init__(self, quality_range: Tuple[int, int]):
        """
        Initialize compression degradation.
        
        Args:
            quality_range: Range for JPEG quality (min, max), 0-100
        """
        self.quality_range = quality_range
    
    def apply_jpeg_compression(self, image: np.ndarray, quality: Optional[int] = None) -> np.ndarray:
        """
        Apply JPEG compression to simulate repeated encoding.
        
        Args:
            image: Input image in BGR format
            quality: JPEG quality factor (0-100), randomly sampled if None
            
        Returns:
            Compressed and decompressed image
        """
        if quality is None:
            quality = np.random.randint(*self.quality_range)
        
        # Ensure quality is in valid range
        quality = max(1, min(100, quality))
        
        # Encode with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_image = cv2.imencode('.jpg', image, encode_param)
        
        # Decode back to image
        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        
        return decoded_image
    
    def apply_multiple_compressions(self, image: np.ndarray, num_iterations: int = 2) -> np.ndarray:
        """
        Apply JPEG compression multiple times to simulate repeated editing.
        
        Args:
            image: Input image in BGR format
            num_iterations: Number of compression cycles
            
        Returns:
            Multi-compressed image
        """
        result = image.copy()
        
        for _ in range(num_iterations):
            # Use progressively lower quality
            quality = np.random.randint(*self.quality_range)
            result = self.apply_jpeg_compression(result, quality)
        
        return result
