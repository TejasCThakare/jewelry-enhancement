"""
Generate training data for refinement model.

This script:
1. Loads raw images
2. Generates I_HQ (Real-ESRGAN on raw)
3. Loads degraded images (level1)
4. Generates I_R (Real-ESRGAN on degraded)
5. Saves NORMALIZED (I_R, I_HQ) pairs for training

Usage:
    python scripts/generate_training_data.py --max-images 490
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import yaml
from tqdm import tqdm

from src.enhancement.enhancer import RealESRGANEnhancer
from src.utils.image_io import load_image


def generate_training_data(
    raw_dir='data/raw',
    degraded_dir='data/degraded/level1_mild',
    output_dir='data/training',
    config_path='config/model_config.yaml',
    max_images=None
):
    """
    Generate training pairs (I_R, I_HQ) for refinement model.
    
    Args:
        raw_dir (str): Directory containing raw images
        degraded_dir (str): Directory containing degraded images
        output_dir (str): Output directory for training data
        config_path (str): Path to model config
        max_images (int): Maximum images to process (None = all)
    """
    
    print("=" * 70)
    print("GENERATING TRAINING DATA FOR REFINEMENT MODEL")
    print("=" * 70)
    print()
    
    # Load config
    print("Loading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize Real-ESRGAN
    print("Initializing Real-ESRGAN...")
    enhancer = RealESRGANEnhancer(config)
    print("✓ Real-ESRGAN initialized")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get raw image files
    raw_path = Path(raw_dir)
    raw_files = sorted(list(raw_path.glob('*.jpg')) + list(raw_path.glob('*.png')))
    
    if max_images:
        raw_files = raw_files[:max_images]
    
    print(f"Processing {len(raw_files)} images...")
    print(f"Raw directory:      {raw_dir}")
    print(f"Degraded directory: {degraded_dir}")
    print(f"Output directory:   {output_dir}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for idx, raw_file in enumerate(tqdm(raw_files, desc="Generating pairs")):
        try:
            # 1. Load raw image
            raw = load_image(str(raw_file))
            
            # 2. Generate I_HQ (target - best quality)
            I_HQ = enhancer.enhance(raw)
            
            # 3. Load degraded image
            deg_file = Path(degraded_dir) / raw_file.name
            if not deg_file.exists():
                tqdm.write(f"  Skipping {raw_file.name} - no degraded version")
                failed_count += 1
                continue
            
            I_deg = load_image(str(deg_file))
            
            # 4. Generate I_R (input - restored from degraded)
            I_R = enhancer.enhance(I_deg)
            
            # ===== CRITICAL FIX: NORMALIZE TO [-1, 1] =====
            # Convert from [0, 255] to [-1, 1]
            I_R_normalized = (I_R.astype(np.float32) / 127.5) - 1.0
            I_HQ_normalized = (I_HQ.astype(np.float32) / 127.5) - 1.0
            
            # 5. Save NORMALIZED data as .npy files
            np.save(output_path / f'{idx:04d}_input.npy', I_R_normalized)
            np.save(output_path / f'{idx:04d}_target.npy', I_HQ_normalized)
            
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"  Error on {raw_file.name}: {e}")
            failed_count += 1
            continue
    
    print()
    print("=" * 70)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Successfully processed: {success_count} pairs")
    print(f"Failed:                 {failed_count} pairs")
    print(f"Output directory:       {output_dir}")
    print()
    
    # Verify normalization
    if success_count > 0:
        sample_input = np.load(output_path / '0000_input.npy')
        sample_target = np.load(output_path / '0000_target.npy')
        print("Data verification:")
        print(f"  Input range:  [{sample_input.min():.2f}, {sample_input.max():.2f}]")
        print(f"  Target range: [{sample_target.min():.2f}, {sample_target.max():.2f}]")
        
        if sample_input.min() >= -1.1 and sample_input.max() <= 1.1:
            print("  ✅ Normalization correct!")
        else:
            print("  ❌ WARNING: Data not properly normalized!")
    print()
    
    print("Next steps:")
    print("  1. Split data: python src/training/dataset_split.py")
    print("  2. Train CNN:  python src/training/train_refinement_gan.py --mode cnn")
    print("  3. Train GAN:  python src/training/train_refinement_gan.py --mode gan")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data for refinement model'
    )
    parser.add_argument('--raw', type=str, default='data/raw',
                       help='Raw images directory')
    parser.add_argument('--degraded', type=str, default='data/degraded/level1_mild',
                       help='Degraded images directory')
    parser.add_argument('--output', type=str, default='data/training',
                       help='Output directory for training data')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Model configuration file')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    generate_training_data(
        args.raw,
        args.degraded,
        args.output,
        args.config,
        args.max_images
    )


if __name__ == '__main__':
    main()
