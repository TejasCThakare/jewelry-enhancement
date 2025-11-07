import os
import sys
import zipfile
from pathlib import Path
import shutil


def download_kaggle_dataset():
    """Download Tanishq jewelry dataset from Kaggle."""
    try:
        import kaggle
        
        print("Downloading Tanishq Jewellery Dataset from Kaggle...")
        
        # Set dataset path
        dataset_name = "sapnilpatel/tanishq-jewellery-dataset"
        download_path = "data/temp"
        
        # Create temp directory
        Path(download_path).mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        print("Dataset downloaded successfully")
        return download_path
        
    except ImportError:
        print("Kaggle API not installed. Please install: pip install kaggle")
        print("Also configure Kaggle API token: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def organize_dataset(download_path: str, output_path: str = "data/raw"):
    """
    Organize downloaded dataset into proper structure.
    
    Args:
        download_path: Path where dataset was downloaded
        output_path: Target path for organized data
    """
    print(f"Organizing dataset from {download_path} to {output_path}...")
    
    download_dir = Path(download_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_count = 0
    
    for ext in image_extensions:
        for image_path in download_dir.rglob(f"*{ext}"):
            # Copy to output directory with organized naming
            output_file = output_dir / image_path.name
            shutil.copy2(image_path, output_file)
            image_count += 1
    
    print(f"Organized {image_count} images into {output_path}")
    
    # Clean up temp directory
    if download_dir.exists() and download_dir != output_dir:
        shutil.rmtree(download_dir)
        print("Cleaned up temporary files")
    
    return image_count


def verify_dataset(data_path: str = "data/raw"):
    """
    Verify dataset integrity.
    
    Args:
        data_path: Path to dataset
    """
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"Error: Dataset directory not found at {data_path}")
        return False
    
    # Count images
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png"))
    
    print(f"\nDataset verification:")
    print(f"  Location: {data_path}")
    print(f"  Total images: {len(image_files)}")
    
    if len(image_files) == 0:
        print("  Warning: No images found!")
        return False
    
    # Check image properties
    import cv2
    sample_sizes = []
    
    for img_path in image_files[:10]:  # Check first 10 images
        img = cv2.imread(str(img_path))
        if img is not None:
            sample_sizes.append(img.shape[:2])
    
    if sample_sizes:
        print(f"  Sample image sizes: {sample_sizes[:3]}")
    
    print("  Status: Dataset ready!")
    return True


def main():
    """Main function to download and organize dataset."""
    print("=" * 60)
    print("Jewelry Enhancement - Dataset Download Script")
    print("=" * 60)
    print()
    
    # Check if dataset already exists
    raw_data_path = "data/raw"
    if Path(raw_data_path).exists():
        image_count = len(list(Path(raw_data_path).glob("*.jpg"))) + \
                     len(list(Path(raw_data_path).glob("*.jpeg"))) + \
                     len(list(Path(raw_data_path).glob("*.png")))
        
        if image_count > 0:
            print(f"Dataset already exists with {image_count} images at {raw_data_path}")
            response = input("Do you want to re-download? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Using existing dataset")
                verify_dataset(raw_data_path)
                return
    
    # Download dataset
    temp_path = download_kaggle_dataset()
    
    # Organize dataset
    organize_dataset(temp_path, raw_data_path)
    
    # Verify dataset
    verify_dataset(raw_data_path)
    
    print("\nDataset download complete!")
    print(f"Images are available at: {raw_data_path}")


if __name__ == "__main__":
    main()
