"""
Script to evaluate enhancement results.
UPDATED: Fixed PSNR calculation with proper resizing
"""

import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.image_io import load_image


def calculate_psnr(img1, img2):
    """Calculate PSNR - handles different sizes correctly."""
    # Ensure same type
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Resize if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100.0
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1, img2):
    """Calculate SSIM."""
    from skimage.metrics import structural_similarity as ssim
    
    # Resize if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.astype(np.uint8)
    
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        gray2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.astype(np.uint8)
    
    score = ssim(gray1, gray2, data_range=255)
    return float(score)


def evaluate_results(original_dir="data/raw",
                    degraded_dir="data/degraded",
                    enhanced_dir="data/enhanced",
                    output_dir="results",
                    max_images=50):
    """Evaluate enhancement results."""
    print("=" * 70)
    print("EVALUATING ENHANCEMENT RESULTS")
    print("=" * 70)
    print()
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Get original images
    original_path = Path(original_dir)
    original_files = sorted(list(original_path.glob("*.jpg")) + 
                          list(original_path.glob("*.png")))[:max_images]
    
    print(f"Found {len(original_files)} original images")
    print()
    
    # Results storage
    all_results = []
    
    # Process each degradation level
    degraded_path = Path(degraded_dir)
    enhanced_path = Path(enhanced_dir)
    
    for level_dir in sorted(degraded_path.iterdir()):
        if not level_dir.is_dir():
            continue
        
        level_name = level_dir.name
        print(f"Processing: {level_name}")
        
        results = {
            'level': level_name,
            'psnr_degraded': [],
            'psnr_enhanced': [],
            'ssim_degraded': [],
            'ssim_enhanced': [],
        }
        
        # Process each image
        for orig_file in tqdm(original_files, desc=f"  {level_name}"):
            try:
                # Load original
                original = load_image(str(orig_file))
                
                # Load degraded
                deg_file = level_dir / orig_file.name
                if not deg_file.exists():
                    continue
                degraded = load_image(str(deg_file))
                
                # Load enhanced
                enh_file = enhanced_path / level_name / orig_file.name
                if not enh_file.exists():
                    continue
                enhanced = load_image(str(enh_file))
                
                # Calculate metrics
                psnr_deg = calculate_psnr(original, degraded)
                psnr_enh = calculate_psnr(original, enhanced)
                ssim_deg = calculate_ssim(original, degraded)
                ssim_enh = calculate_ssim(original, enhanced)
                
                # Store
                results['psnr_degraded'].append(psnr_deg)
                results['psnr_enhanced'].append(psnr_enh)
                results['ssim_degraded'].append(ssim_deg)
                results['ssim_enhanced'].append(ssim_enh)
                
            except Exception as e:
                pass
        
        # Print results
        if results['psnr_enhanced']:
            psnr_d = np.mean(results['psnr_degraded'])
            psnr_e = np.mean(results['psnr_enhanced'])
            ssim_d = np.mean(results['ssim_degraded'])
            ssim_e = np.mean(results['ssim_enhanced'])
            
            print(f"  PSNR: {psnr_d:.2f} → {psnr_e:.2f} dB (+{psnr_e-psnr_d:.2f})")
            print(f"  SSIM: {ssim_d:.4f} → {ssim_e:.4f} (+{ssim_e-ssim_d:.4f})")
            print()
            
            all_results.append(results)
    
    # Save CSV
    csv_path = metrics_dir / "detailed_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Level', 'Metric', 'Degraded_Mean', 'Degraded_Std', 
                        'Enhanced_Mean', 'Enhanced_Std', 'Improvement'])
        
        for res in all_results:
            level = res['level']
            
            # PSNR
            d_mean = np.mean(res['psnr_degraded'])
            d_std = np.std(res['psnr_degraded'])
            e_mean = np.mean(res['psnr_enhanced'])
            e_std = np.std(res['psnr_enhanced'])
            imp = e_mean - d_mean
            writer.writerow([level, 'PSNR', f"{d_mean:.2f}", f"{d_std:.2f}",
                           f"{e_mean:.2f}", f"{e_std:.2f}", f"+{imp:.2f} dB"])
            
            # SSIM
            d_mean = np.mean(res['ssim_degraded'])
            d_std = np.std(res['ssim_degraded'])
            e_mean = np.mean(res['ssim_enhanced'])
            e_std = np.std(res['ssim_enhanced'])
            imp = (e_mean - d_mean) * 100
            writer.writerow([level, 'SSIM', f"{d_mean:.4f}", f"{d_std:.4f}",
                           f"{e_mean:.4f}", f"{e_std:.4f}", f"+{imp:.2f}%"])
    
    print("=" * 70)
    print(f"Results saved to: {csv_path}")
    print("=" * 70)
    
    # Display
    df = pd.read_csv(csv_path)
    print("\n" + df.to_string(index=False))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="data/raw")
    parser.add_argument("--degraded", default="data/degraded")
    parser.add_argument("--enhanced", default="data/enhanced")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    
    evaluate_results(args.original, args.degraded, args.enhanced, args.output)


if __name__ == "__main__":
    main()
