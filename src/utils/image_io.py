import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image in BGR format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image in BGR format
        output_path: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format
        ext = Path(output_path).suffix.lower()
        
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)])
        else:
            cv2.imwrite(output_path, image)
        
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def load_images_from_directory(directory: str, 
                               extensions: Optional[List[str]] = None,
                               max_images: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Load all images from a directory.
    
    Args:
        directory: Directory path
        extensions: List of file extensions to load (default: ['.jpg', '.jpeg', '.png'])
        max_images: Maximum number of images to load
        
    Returns:
        List of tuples (filename, image)
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return []
    
    images = []
    
    for ext in extensions:
        pattern = f"*{ext}"
        for image_path in directory_path.rglob(pattern):
            try:
                image = load_image(str(image_path))
                images.append((image_path.name, image))
                
                if max_images and len(images) >= max_images:
                    return images
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                continue
    
    return images


def resize_image(image: np.ndarray, max_size: int = 1024, 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to fit within max_size while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if maintain_aspect:
        if height > max_size or width > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(image, (max_size, max_size), interpolation=cv2.INTER_AREA)
    
    return image


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    
    Args:
        image: Image in BGR format
        
    Returns:
        Image in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR.
    
    Args:
        image: Image in RGB format
        
    Returns:
        Image in BGR format
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
