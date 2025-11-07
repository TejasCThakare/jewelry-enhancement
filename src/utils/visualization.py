import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def create_comparison_grid(images: List[np.ndarray], 
                          labels: List[str],
                          grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images in BGR format
        labels: List of labels for each image
        grid_size: Tuple of (rows, cols), auto-calculated if None
        
    Returns:
        Combined grid image
    """
    if not images:
        raise ValueError("No images provided")
    
    num_images = len(images)
    
    # Auto-calculate grid size
    if grid_size is None:
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Resize all images to same size
    target_height = 512
    target_width = 512
    resized_images = []
    
    for img in images:
        resized = cv2.resize(img, (target_width, target_height))
        resized_images.append(resized)
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_images = []
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                img = resized_images[idx].copy()
                
                # Add label
                label = labels[idx] if idx < len(labels) else ""
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 0), 1, cv2.LINE_AA)
                
                row_images.append(img)
            else:
                # Add blank image
                row_images.append(np.zeros((target_height, target_width, 3), dtype=np.uint8))
        
        grid_rows.append(np.hstack(row_images))
    
    return np.vstack(grid_rows)


def save_comparison(original: np.ndarray, 
                   degraded: np.ndarray, 
                   enhanced: np.ndarray,
                   output_path: str,
                   metrics: Optional[Dict[str, float]] = None):
    """
    Save side-by-side comparison of images.
    
    Args:
        original: Original image
        degraded: Degraded image
        enhanced: Enhanced image
        output_path: Output file path
        metrics: Optional metrics to display
    """
    images = [original, degraded, enhanced]
    labels = ['Original', 'Degraded', 'Enhanced']
    
    grid = create_comparison_grid(images, labels, grid_size=(1, 3))
    
    # Add metrics text if provided
    if metrics:
        text_y = 60
        for key, value in metrics.items():
            text = f"{key}: {value:.4f}"
            cv2.putText(grid, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(grid, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 0), 1, cv2.LINE_AA)
            text_y += 25
    
    cv2.imwrite(output_path, grid)


def plot_metrics(metrics_dict: Dict[str, List[float]], 
                output_path: str,
                title: str = "Image Quality Metrics"):
    """
    Plot metrics as bar chart.
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
        output_path: Output file path
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(list(metrics_dict.values())[0]))
    width = 0.8 / len(metrics_dict)
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        offset = width * i - (width * len(metrics_dict) / 2)
        plt.bar(x + offset, values, width, label=metric_name)
    
    plt.xlabel('Image Index')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_distribution(metrics: List[float], 
                            metric_name: str,
                            output_path: str):
    """
    Plot distribution of a metric across images.
    
    Args:
        metrics: List of metric values
        metric_name: Name of the metric
        output_path: Output file path
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(metrics, kde=True, bins=30)
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric_name}')
    plt.axvline(np.mean(metrics), color='r', linestyle='--', label=f'Mean: {np.mean(metrics):.4f}')
    plt.axvline(np.median(metrics), color='g', linestyle='--', label=f'Median: {np.median(metrics):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_before_after_slider(original: np.ndarray, 
                               enhanced: np.ndarray,
                               output_path: str):
    """
    Create a before/after comparison image with vertical split.
    
    Args:
        original: Original/degraded image
        enhanced: Enhanced image
        output_path: Output file path
    """
    # Ensure same size
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    height, width = original.shape[:2]
    
    # Create split image
    split_image = np.zeros_like(original)
    split_point = width // 2
    
    split_image[:, :split_point] = original[:, :split_point]
    split_image[:, split_point:] = enhanced[:, split_point:]
    
    # Draw dividing line
    cv2.line(split_image, (split_point, 0), (split_point, height), (255, 255, 255), 3)
    
    # Add labels
    cv2.putText(split_image, "Before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(split_image, "Before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(split_image, "After", (split_point + 20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(split_image, "After", (split_point + 20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite(output_path, split_image)
