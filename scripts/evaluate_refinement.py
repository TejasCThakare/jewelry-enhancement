"""
Evaluate refinement model on test set.

Computes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)

Saves:
- Metrics to results/metrics/eval_results.json
- Visual comparisons to results/comparisons/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance
)
import cv2

from src.training.models import RefinementCNN, RefinementGAN
from src.utils.image_io import save_image


class RefinementEvaluator:
    """
    Evaluator for refinement models.
    
    Args:
        model_path (str): Path to trained model checkpoint
        mode (str): 'cnn' or 'gan'
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, model_path, mode='cnn', device='cuda'):
        self.device = device
        self.mode = mode
        
        # Load model
        print(f"Loading {mode.upper()} model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        if mode == 'cnn':
            self.model = RefinementCNN().to(device)
        else:
            self.model = RefinementGAN().to(device)
        
        self.model.load_state_dict(checkpoint['generator_state_dict'])
        self.model.eval()
        print("✓ Model loaded")
        
        # Initialize metrics
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity().to(device)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
    
    @torch.no_grad()
    def evaluate(self, test_loader, output_dir='results'):
        """
        Evaluate model on test set.
        
        Args:
            test_loader (DataLoader): Test data loader
            output_dir (str): Output directory for results
            
        Returns:
            dict: Evaluation metrics
        """
        
        print("\n" + "=" * 70)
        print("EVALUATING REFINEMENT MODEL")
        print("=" * 70)
        print(f"Test samples: {len(test_loader.dataset)}")
        print()
        
        output_path = Path(output_dir)
        comparison_dir = output_path / 'comparisons'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        for idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Generate predictions
            outputs = self.model(inputs)
            
            # Update metrics (per-batch)
            psnr_val = self.psnr(outputs, targets).item()
            ssim_val = self.ssim(outputs, targets).item()
            lpips_val = self.lpips(outputs, targets).item()
            
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_lpips.append(lpips_val)
            
            # Update FID
            # Denormalize from [-1, 1] to [0, 1]
            outputs_01 = (outputs + 1) / 2
            targets_01 = (targets + 1) / 2
            outputs_255 = (outputs_01 * 255).byte()
            targets_255 = (targets_01 * 255).byte()
            
            self.fid.update(targets_255, real=True)
            self.fid.update(outputs_255, real=False)
            
            # Save first 10 comparisons
            if idx < 10:
                self._save_comparison(
                    inputs[0], outputs[0], targets[0],
                    comparison_dir / f'test_{idx:04d}.jpg'
                )
        
        # Compute final metrics
        metrics = {
            'psnr_mean': np.mean(all_psnr),
            'psnr_std': np.std(all_psnr),
            'ssim_mean': np.mean(all_ssim),
            'ssim_std': np.std(all_ssim),
            'lpips_mean': np.mean(all_lpips),
            'lpips_std': np.std(all_lpips),
            'fid': self.fid.compute().item()
        }
        
        # Save metrics
        metrics_path = output_path / 'metrics' / 'eval_results.json'
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
        print(f"FID:   {metrics['fid']:.2f}")
        print("=" * 70)
        print(f"\n✓ Results saved to {metrics_path}")
        print(f"✓ Comparisons saved to {comparison_dir}")
        
        return metrics
    
    def _save_comparison(self, input_img, output_img, target_img, path):
        """Save side-by-side comparison."""
        
        # Convert to numpy (C, H, W) -> (H, W, C)
        def to_numpy(tensor):
            img = tensor.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            # Denormalize from [-1, 1] to [0, 255]
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            return img
        
        input_np = to_numpy(input_img)
        output_np = to_numpy(output_img)
        target_np = to_numpy(target_img)
        
        # Resize for display
        h, w = 400, 400
        input_disp = cv2.resize(input_np, (w, h))
        output_disp = cv2.resize(output_np, (w, h))
        target_disp = cv2.resize(target_np, (w, h))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(input_disp, 'Input', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(output_disp, 'Refined', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(target_disp, 'Target', (10, 30), font, 1, (255, 255, 255), 2)
        
        # Concatenate
        gap = np.zeros((h, 10, 3), dtype=np.uint8)
        comparison = np.hstack([input_disp, gap, output_disp, gap, target_disp])
        
        # Save
        save_image(comparison, str(path))


def main():
    parser = argparse.ArgumentParser(description='Evaluate refinement model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, required=True, choices=['cnn', 'gan'],
                       help='Model mode: cnn or gan')
    parser.add_argument('--data', type=str, default='data/training',
                       help='Training data directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Test dataset
    from src.training.train_refinement_gan import JewelryDataset
    test_dataset = JewelryDataset(args.data, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluator
    evaluator = RefinementEvaluator(args.model, args.mode, device)
    
    # Evaluate
    evaluator.evaluate(test_loader, args.output)


if __name__ == '__main__':
    main()
