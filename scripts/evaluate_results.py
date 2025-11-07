import sys
import csv
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import ImageQualityMetrics
from src.utils.image_io import load_image
from src.utils.visualization import (
    save_comparison, 
    plot_metric_distribution,
    create_before_after_slider
)


def evaluate_results(original_dir: str = "data/raw",
                    degraded_dir: str = "data/degraded",
                    enhanced_dir: str = "data/enhanced",
                    output_dir: str = "results"):
    """
    Evaluate enhancement results and generate metrics.
    
    Args:
        original_dir: Directory with original images
        degraded_dir: Directory with degraded images
        enhanced_dir: Directory with enhanced images
        output_dir: Output directory for results
    """
    print("=" * 60)
    print("Evaluating Enhancement Results")
    print("=" * 60)
    print()
    
    # Initialize metrics calculator
    metrics_calc = ImageQualityMetrics()
    
    # Create output directories
    output_path = Path(output_dir)
    (output_path / "comparisons").mkdir(parents=True, exist_ok=True)
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    (output_path / "visualizations").mkdir(parents=True, exist_ok=True)
    
    # Get original images
    original_path = Path(original_dir)
    original_files = list(original_path.glob("*.jpg")) + \
                    list(original_path.glob("*.jpeg")) + \
                    list(original_path.glob("*.png"))
    
    print(f"Found {len(original_files)} original images")
    print()
    
    # Process each degradation level
    degraded_path = Path(degraded_dir)
    enhanced_path = Path(enhanced_dir)
    
    all_results = []
    
    for degradation_level_dir in sorted(degraded_path.iterdir()):
        if not degradation_level_dir.is_dir():
            continue
        
        level_name = degradation_level_dir.name
        print(f"Evaluating degradation level: {level_name}")
        
        level_results = {
            'level': level_name,
            'psnr_degraded': [],
            'psnr_enhanced': [],
            'ssim_degraded': [],
            'ssim_enhanced': [],
            'mse_degraded': [],
            'mse_enhanced': [],
        }
        
        # Process each image
        for original_file in tqdm(original_files[:20], desc=f"  Evaluating ({level_name})"):  # First 20 for demo
            try:
                # Load images
                original = load_image(str(original_file))
                degraded_file = degradation_level_dir / original_file.name
                enhanced_file = enhanced_path / level_name / original_file.name
                
                if not degraded_file.exists() or not enhanced_file.exists():
                    continue
                
                degraded = load_image(str(degraded_file))
                enhanced = load_image(str(enhanced_file))
                
                # Calculate metrics
                metrics_degraded = metrics_calc.calculate_all_metrics(original, degraded)
                metrics_enhanced = metrics_calc.calculate_all_metrics(original, enhanced)
                
                # Store metrics
                level_results['psnr_degraded'].append(metrics_degraded['psnr'])
                level_results['psnr_enhanced'].append(metrics_enhanced['psnr'])
                level_results['ssim_degraded'].append(metrics_degraded['ssim'])
                level_results['ssim_enhanced'].append(metrics_enhanced['ssim'])
                level_results['mse_degraded'].append(metrics_degraded['mse'])
                level_results['mse_enhanced'].append(metrics_enhanced['mse'])
                
                # Save comparison for first few images
                if len(level_results['psnr_enhanced']) <= 5:
                    comparison_path = output_path / "comparisons" / f"{level_name}_{original_file.stem}.jpg"
                    save_comparison(
                        original, degraded, enhanced, 
                        str(comparison_path),
                        {
                            'PSNR_Enh': metrics_enhanced['psnr'],
                            'SSIM_Enh': metrics_enhanced['ssim']
                        }
                    )
                    
                    # Create before/after slider
                    slider_path = output_path / "comparisons" / f"{level_name}_{original_file.stem}_slider.jpg"
                    create_before_after_slider(degraded, enhanced, str(slider_path))
                
            except Exception as e:
                print(f"    Error evaluating {original_file.name}: {e}")
        
        # Calculate statistics
        if level_results['psnr_enhanced']:
            print(f"  Results for {level_name}:")
            print(f"    PSNR: {np.mean(level_results['psnr_degraded']):.2f} dB (degraded) -> "
                  f"{np.mean(level_results['psnr_enhanced']):.2f} dB (enhanced)")
            print(f"    SSIM: {np.mean(level_results['ssim_degraded']):.4f} (degraded) -> "
                  f"{np.mean(level_results['ssim_enhanced']):.4f} (enhanced)")
            print(f"    Images evaluated: {len(level_results['psnr_enhanced'])}")
            print()
            
            all_results.append(level_results)
            
            # Plot distributions
            dist_path = output_path / "visualizations" / f"{level_name}_psnr_distribution.png"
            plot_metric_distribution(
                level_results['psnr_enhanced'],
                f"PSNR ({level_name})",
                str(dist_path)
            )
    
    # Save detailed metrics to CSV
    csv_path = output_path / "metrics" / "detailed_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Level', 'Metric', 'Degraded_Mean', 'Degraded_Std', 
                        'Enhanced_Mean', 'Enhanced_Std', 'Improvement'])
        
        for result in all_results:
            level = result['level']
            
            for metric in ['psnr', 'ssim', 'mse']:
                deg_mean = np.mean(result[f'{metric}_degraded'])
                deg_std = np.std(result[f'{metric}_degraded'])
                enh_mean = np.mean(result[f'{metric}_enhanced'])
                enh_std = np.std(result[f'{metric}_enhanced'])
                
                if metric == 'mse':
                    improvement = ((deg_mean - enh_mean) / deg_mean) * 100
                else:
                    improvement = ((enh_mean - deg_mean) / deg_mean) * 100
                
                writer.writerow([
                    level, metric.upper(),
                    f"{deg_mean:.4f}", f"{deg_std:.4f}",
                    f"{enh_mean:.4f}", f"{enh_std:.4f}",
                    f"{improvement:.2f}%"
                ])
    
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Comparisons: {output_dir}/comparisons/")
    print(f"  - Metrics: {output_dir}/metrics/")
    print(f"  - Visualizations: {output_dir}/visualizations/")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate enhancement results")
    parser.add_argument("--original", default="data/raw", help="Original images directory")
    parser.add_argument("--degraded", default="data/degraded", help="Degraded images directory")
    parser.add_argument("--enhanced", default="data/enhanced", help="Enhanced images directory")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    evaluate_results(args.original, args.degraded, args.enhanced, args.output)


if __name__ == "__main__":
    main()
