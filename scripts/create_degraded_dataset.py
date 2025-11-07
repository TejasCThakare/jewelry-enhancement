import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from src.degradation.pipeline import DegradationPipeline
from src.utils.image_io import load_image, save_image


def create_degraded_dataset(input_dir: str = "data/raw",
                           output_base_dir: str = "data/degraded",
                           config_path: str = "config/degradation_config.yaml"):
    """
    Create degraded versions of all images in the dataset.
    
    Args:
        input_dir: Directory containing original images
        output_base_dir: Base directory for degraded images
        config_path: Path to degradation configuration file
    """
    print("=" * 60)
    print("Creating Degraded Dataset")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    print(f"Loading configuration from {config_path}...")
    pipeline = DegradationPipeline(config_path)
    
    # Get input images
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    image_files = list(input_path.glob("*.jpg")) + \
                 list(input_path.glob("*.jpeg")) + \
                 list(input_path.glob("*.png"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print()
    
    # Get degradation levels from config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    degradation_levels = list(config['degradation_levels'].keys())
    
    # Process each degradation level
    total_processed = 0
    
    for level in degradation_levels:
        print(f"Processing degradation level: {level}")
        print(f"  Description: {pipeline.get_level_description(level)}")
        
        # Create output directory
        output_dir = Path(output_base_dir) / level
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        failed_images = []
        
        for image_path in tqdm(image_files, desc=f"  Degrading images ({level})"):
            try:
                # Load image
                image = load_image(str(image_path))
                
                # Apply degradation
                degraded = pipeline.apply_degradation(image, level)
                
                # Save degraded image
                output_path = output_dir / image_path.name
                save_image(degraded, str(output_path))
                
                total_processed += 1
                
            except Exception as e:
                print(f"    Error processing {image_path.name}: {e}")
                failed_images.append(image_path.name)
        
        print(f"  Completed: {len(image_files) - len(failed_images)}/{len(image_files)} images")
        
        if failed_images:
            print(f"  Failed images: {failed_images[:5]}...")
        
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images processed: {total_processed}")
    print(f"Degradation levels: {len(degradation_levels)}")
    print(f"Output directory: {output_base_dir}")
    print()
    
    # Verify output
    for level in degradation_levels:
        level_dir = Path(output_base_dir) / level
        level_count = len(list(level_dir.glob("*.jpg"))) + \
                     len(list(level_dir.glob("*.jpeg"))) + \
                     len(list(level_dir.glob("*.png")))
        print(f"  {level}: {level_count} images")
    
    print("\nDegraded dataset creation complete!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create degraded dataset")
    parser.add_argument("--input", default="data/raw", help="Input directory with original images")
    parser.add_argument("--output", default="data/degraded", help="Output base directory")
    parser.add_argument("--config", default="config/degradation_config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    create_degraded_dataset(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
