"""
Script to run enhancement pipeline on degraded images.
UPDATED: Minimal pre/post processing for best quality
"""

import sys
from pathlib import Path
from tqdm import tqdm
import yaml
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.enhancement.enhancer import RealESRGANEnhancer
from src.utils.image_io import load_image, save_image


def run_enhancement(input_dir: str = "data/degraded",
                   output_dir: str = "data/enhanced",
                   config_path: str = "config/model_config.yaml"):
    """
    Run enhancement on all degraded images.
    """
    print("=" * 60)
    print("Running Enhancement Pipeline")
    print("=" * 60)
    print()
    
    # Load config
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize enhancer
    print("Initializing enhancement pipeline...")
    enhancer = RealESRGANEnhancer(config)
    print()
    
    # Process each degradation level
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    total_images = 0
    failed_images = 0
    start_time = time.time()
    
    for degradation_level_dir in sorted(input_path.iterdir()):
        if not degradation_level_dir.is_dir():
            continue
        
        level_name = degradation_level_dir.name
        print(f"Processing degradation level: {level_name}")
        
        # Create output directory
        level_output_dir = output_path / level_name
        level_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(degradation_level_dir.glob("*.jpg")) + \
                     list(degradation_level_dir.glob("*.png"))
        
        # Process each image
        for image_file in tqdm(image_files, desc=f"  Enhancing ({level_name})"):
            try:
                # Load image
                image = load_image(str(image_file))
                
                # SIMPLIFIED: Just enhance - no pre/post processing
                enhanced = enhancer.enhance(image)
                
                # Save result
                output_file = level_output_dir / image_file.name
                save_image(enhanced, str(output_file))
                
                total_images += 1
                
            except Exception as e:
                print(f"    Error processing {image_file.name}: {e}")
                failed_images += 1
        
        print(f"  Completed: {len(image_files)}/{len(image_files)} images")
        print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per image: {elapsed/total_images:.2f} seconds")
    print(f"Failed images: {failed_images}")
    print()
    print(f"Enhanced images saved to: {output_dir}")
    print()
    print("Enhancement complete!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhancement pipeline")
    parser.add_argument("--input", default="data/degraded", help="Input directory")
    parser.add_argument("--output", default="data/enhanced", help="Output directory")
    parser.add_argument("--config", default="config/model_config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    run_enhancement(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
