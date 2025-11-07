import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.enhancement.preprocessor import JewelryPreprocessor
from src.enhancement.enhancer import RealESRGANEnhancer
from src.enhancement.postprocessor import JewelryPostprocessor
from src.utils.image_io import load_image, save_image


def run_enhancement(input_dir: str = "data/degraded",
                   output_dir: str = "data/enhanced",
                   config_path: str = "config/model_config.yaml"):
    """
    Run enhancement pipeline on all degraded images.
    
    Args:
        input_dir: Directory containing degraded images
        output_dir: Output directory for enhanced images
        config_path: Path to model configuration file
    """
    print("=" * 60)
    print("Running Enhancement Pipeline")
    print("=" * 60)
    print()
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline components
    print("Initializing enhancement pipeline...")
    preprocessor = JewelryPreprocessor(config.get('preprocessing', {}))
    enhancer = RealESRGANEnhancer(config)
    postprocessor = JewelryPostprocessor(config.get('postprocessing', {}))
    print("Pipeline initialized successfully")
    print()
    
    # Get input images
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Process all subdirectories (degradation levels)
    total_processed = 0
    total_time = 0
    failed_images = []
    
    for degradation_dir in sorted(input_path.iterdir()):
        if not degradation_dir.is_dir():
            continue
        
        level_name = degradation_dir.name
        print(f"Processing degradation level: {level_name}")
        
        # Get images
        image_files = list(degradation_dir.glob("*.jpg")) + \
                     list(degradation_dir.glob("*.jpeg")) + \
                     list(degradation_dir.glob("*.png"))
        
        print(f"  Found {len(image_files)} images")
        
        # Create output directory
        output_level_dir = Path(output_dir) / level_name
        output_level_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        for image_path in tqdm(image_files, desc=f"  Enhancing ({level_name})"):
            try:
                start_time = time.time()
                
                # Load image
                image = load_image(str(image_path))
                
                # Step 1: Preprocessing
                preprocessed = preprocessor.preprocess(image, denoise=True, enhance_contrast=True)
                
                # Step 2: Enhancement (Real-ESRGAN)
                enhanced = enhancer.enhance(preprocessed)
                
                # Step 3: Postprocessing
                final = postprocessor.postprocess(enhanced)
                
                # Save result
                output_path = output_level_dir / image_path.name
                save_image(final, str(output_path))
                
                # Track time
                elapsed = time.time() - start_time
                total_time += elapsed
                total_processed += 1
                
            except Exception as e:
                print(f"    Error processing {image_path.name}: {e}")
                failed_images.append((level_name, image_path.name))
        
        print(f"  Completed: {len(image_files) - len([f for l, f in failed_images if l == level_name])}/{len(image_files)} images")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images processed: {total_processed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / total_processed:.2f} seconds" if total_processed > 0 else "N/A")
    print(f"Failed images: {len(failed_images)}")
    
    if failed_images:
        print("\nFailed images:")
        for level, filename in failed_images[:10]:
            print(f"  {level}/{filename}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print(f"\nEnhanced images saved to: {output_dir}")
    print("\nEnhancement complete!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhancement pipeline")
    parser.add_argument("--input", default="data/degraded", help="Input directory with degraded images")
    parser.add_argument("--output", default="data/enhanced", help="Output directory")
    parser.add_argument("--config", default="config/model_config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    run_enhancement(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
