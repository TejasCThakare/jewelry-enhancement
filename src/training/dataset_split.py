"""
Dataset splitting utility for train/val/test splits.

Creates 80/10/10 split from generated training data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import shutil
from sklearn.model_selection import train_test_split


def create_dataset_splits(
    source_dir='data/training',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """
    Split training data into train/val/test sets.
    
    Args:
        source_dir (str): Directory containing *_input.npy and *_target.npy files
        train_ratio (float): Training set ratio (default: 0.8)
        val_ratio (float): Validation set ratio (default: 0.1)
        test_ratio (float): Test set ratio (default: 0.1)
        seed (int): Random seed for reproducibility
    
    Directory structure after splitting:
        data/training/
        ├── train/
        │   ├── 0000_input.npy
        │   ├── 0000_target.npy
        │   └── ...
        ├── val/
        └── test/
    """
    
    print("="*70)
    print("CREATING DATASET SPLITS")
    print("="*70)
    print()
    
    source_path = Path(source_dir)
    
    # Get all input files
    input_files = sorted(list(source_path.glob('*_input.npy')))
    
    if len(input_files) == 0:
        raise ValueError(f"No training data found in {source_dir}")
    
    # Get corresponding target files
    target_files = [
        str(f).replace('_input.npy', '_target.npy')
        for f in input_files
    ]
    
    # Verify all target files exist
    for target_file in target_files:
        if not Path(target_file).exists():
            raise ValueError(f"Target file not found: {target_file}")
    
    print(f"Found {len(input_files)} training pairs")
    print(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print()
    
    # Create splits
    indices = list(range(len(input_files)))
    
    # Train + (Val + Test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )
    
    # Val + Test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio_adjusted),
        random_state=seed
    )
    
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_idx)} pairs")
    print(f"  Val:   {len(val_idx)} pairs")
    print(f"  Test:  {len(test_idx)} pairs")
    print()
    
    # Create split directories and copy files
    for split_name, indices in splits.items():
        split_dir = source_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"Creating {split_name} split...")
        
        for new_idx, old_idx in enumerate(indices):
            # Copy input file
            src_input = input_files[old_idx]
            dst_input = split_dir / f'{new_idx:04d}_input.npy'
            shutil.copy2(src_input, dst_input)
            
            # Copy target file
            src_target = Path(target_files[old_idx])
            dst_target = split_dir / f'{new_idx:04d}_target.npy'
            shutil.copy2(src_target, dst_target)
        
        print(f"  ✓ Copied {len(indices)} pairs to {split_dir}")
    
    print()
    print("="*70)
    print("DATASET SPLITS CREATED SUCCESSFULLY")
    print("="*70)
    print()
    print("Directory structure:")
    print(f"  {source_dir}/")
    print(f"  ├── train/  ({len(train_idx)} pairs)")
    print(f"  ├── val/    ({len(val_idx)} pairs)")
    print(f"  └── test/   ({len(test_idx)} pairs)")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--source', type=str, default='data/training',
                       help='Source directory containing training data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0 (got {total})")
    
    create_dataset_splits(
        args.source,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
