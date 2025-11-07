import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import urllib.request
import os


class RealESRGANEnhancer:
    """Real-ESRGAN based image enhancement."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Real-ESRGAN enhancer.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        
        # Model configuration
        model_config = self.config.get('model', {})
        self.scale = model_config.get('scale', 4)
        self.weights_path = model_config.get('weights', {}).get('path', 'weights/RealESRGAN_x4plus.pth')
        self.weights_url = model_config.get('weights', {}).get('url', 
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth')
        self.auto_download = model_config.get('weights', {}).get('auto_download', True)
        
        # Inference configuration
        inference_config = self.config.get('inference', {})
        self.tile_size = inference_config.get('tile_size', 400)
        self.tile_pad = inference_config.get('tile_pad', 10)
        self.pre_pad = inference_config.get('pre_pad', 0)
        self.half_precision = inference_config.get('half_precision', True)
        
        # Device configuration
        device_config = inference_config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Download weights if needed
        if self.auto_download and not os.path.exists(self.weights_path):
            self._download_weights()
        
        # Initialize model
        self.upsampler = None
        self._initialize_model()
    
    def _download_weights(self):
        """Download pre-trained weights if not present."""
        weights_dir = Path(self.weights_path).parent
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading weights from {self.weights_url}...")
        urllib.request.urlretrieve(self.weights_url, self.weights_path)
        print(f"Weights downloaded to {self.weights_path}")
    
    def _initialize_model(self):
        """Initialize Real-ESRGAN model."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Define model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.scale
            )
            
            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=self.weights_path,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.half_precision and torch.cuda.is_available(),
                device=self.device
            )
            
            print(f"Real-ESRGAN model initialized on {self.device}")
            
        except ImportError as e:
            print(f"Error importing Real-ESRGAN: {e}")
            print("Please install required packages: pip install realesrgan basicsr facexlib gfpgan")
            raise
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def enhance(self, image: np.ndarray, outscale: Optional[int] = None) -> np.ndarray:
        """
        Enhance image using Real-ESRGAN.
        
        Args:
            image: Input image in BGR format
            outscale: Output scale factor, defaults to model scale
            
        Returns:
            Enhanced image in BGR format
        """
        if self.upsampler is None:
            raise RuntimeError("Model not initialized")
        
        if outscale is None:
            outscale = self.scale
        
        try:
            # Real-ESRGAN expects BGR format
            output, _ = self.upsampler.enhance(image, outscale=outscale)
            return output
        except Exception as e:
            print(f"Enhancement failed: {e}")
            # Return upscaled image using bicubic interpolation as fallback
            height, width = image.shape[:2]
            return cv2.resize(image, (width * outscale, height * outscale), 
                            interpolation=cv2.INTER_CUBIC)
    
    def enhance_batch(self, images: list, outscale: Optional[int] = None) -> list:
        """
        Enhance multiple images.
        
        Args:
            images: List of input images in BGR format
            outscale: Output scale factor
            
        Returns:
            List of enhanced images
        """
        return [self.enhance(img, outscale) for img in images]
