import cv2
import numpy as np
from typing import Dict, List, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ImageQualityMetrics:
    """Calculate various image quality metrics."""
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            original: Original (reference) image
            enhanced: Enhanced (test) image
            
        Returns:
            PSNR value in dB
        """
        try:
            return psnr(original, enhanced, data_range=255)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index.
        
        Args:
            original: Original (reference) image
            enhanced: Enhanced (test) image
            
        Returns:
            SSIM value between 0 and 1
        """
        try:
            # Ensure images are same size
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # Calculate SSIM
            score = ssim(original, enhanced, multichannel=True, channel_axis=2, data_range=255)
            return score
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0.0
    
    @staticmethod
    def calculate_mse(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            original: Original (reference) image
            enhanced: Enhanced (test) image
            
        Returns:
            MSE value
        """
        try:
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
            return float(mse)
        except Exception as e:
            print(f"Error calculating MSE: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_mae(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            original: Original (reference) image
            enhanced: Enhanced (test) image
            
        Returns:
            MAE value
        """
        try:
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            mae = np.mean(np.abs(original.astype(float) - enhanced.astype(float)))
            return float(mae)
        except Exception as e:
            print(f"Error calculating MAE: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Input image
            
        Returns:
            Sharpness score (higher is sharper)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except Exception as e:
            print(f"Error calculating sharpness: {e}")
            return 0.0
    
    def calculate_all_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            original: Original (reference) image
            enhanced: Enhanced (test) image
            
        Returns:
            Dictionary with all metric scores
        """
        return {
            'psnr': self.calculate_psnr(original, enhanced),
            'ssim': self.calculate_ssim(original, enhanced),
            'mse': self.calculate_mse(original, enhanced),
            'mae': self.calculate_mae(original, enhanced),
            'sharpness_original': self.calculate_sharpness(original),
            'sharpness_enhanced': self.calculate_sharpness(enhanced),
        }
    
    def compare_images(self, original: np.ndarray, degraded: np.ndarray, 
                      enhanced: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare original, degraded, and enhanced images.
        
        Args:
            original: Original high-quality image
            degraded: Degraded low-quality image
            enhanced: Enhanced image
            
        Returns:
            Dictionary with comparison metrics
        """
        return {
            'original_vs_degraded': self.calculate_all_metrics(original, degraded),
            'original_vs_enhanced': self.calculate_all_metrics(original, enhanced),
            'degraded_vs_enhanced': self.calculate_all_metrics(degraded, enhanced),
        }
