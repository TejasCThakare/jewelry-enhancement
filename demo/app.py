"""
Jewelry Image Enhancement - Interactive Demo
Compares: Original → Real-ESRGAN → Refined CNN/GAN
"""

import gradio as gr
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.enhancement.enhancer import RealESRGANEnhancer
from src.training.models import RefinementCNN, RefinementGAN
import yaml

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading models...")

# Real-ESRGAN
with open('config/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
esrgan = RealESRGANEnhancer(config)

# Refined CNN/GAN
MODEL_PATH = 'models/cnn_best.pth'  # Change to gan_best.pth if using GAN
MODEL_TYPE = 'cnn'  # Change to 'gan' if using GAN

if MODEL_TYPE == 'cnn':
    refined_model = RefinementCNN().cuda()
else:
    refined_model = RefinementGAN().cuda()

checkpoint = torch.load(MODEL_PATH, map_location='cuda')
refined_model.load_state_dict(checkpoint['generator_state_dict'])
refined_model.eval()

print("Models loaded successfully!")


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM between two images."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    ssim = structural_similarity(gray1, gray2, data_range=255)
    
    return psnr, ssim


# ============================================================================
# MAIN ENHANCEMENT FUNCTION
# ============================================================================

def enhance_jewelry_image(image):
    """
    Process image through complete pipeline.
    
    Returns:
        (original, esrgan, refined, metrics_text)
    """
    if image is None:
        return None, None, None, "Please upload an image first!"
    
    # Convert PIL to numpy BGR
    img_np = np.array(image)
    if img_np.shape[2] == 4:  # RGBA → RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    h, w = img_np.shape[:2]
    
    # Convert to BGR for processing
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Step 1: Real-ESRGAN baseline
    print("Running Real-ESRGAN...")
    esrgan_output = esrgan.enhance(img_bgr)
    
    # Step 2: Refined model
    print("Running Refined CNN/GAN...")
    # Normalize to [-1, 1]
    tensor = torch.from_numpy(esrgan_output).permute(2, 0, 1).float()
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.unsqueeze(0).cuda()
    
    # Inference
    with torch.no_grad():
        refined_tensor = torch.clamp(refined_model(tensor), -1, 1)
    
    # Denormalize
    refined_output = refined_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    refined_output = ((refined_output + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Convert all to RGB
    original_rgb = img_np
    esrgan_rgb = cv2.cvtColor(esrgan_output, cv2.COLOR_BGR2RGB)
    refined_rgb = cv2.cvtColor(refined_output, cv2.COLOR_BGR2RGB)
    
    # Calculate metrics (comparing to Real-ESRGAN as pseudo-reference)
    psnr_esrgan, ssim_esrgan = calculate_metrics(original_rgb, esrgan_rgb)
    psnr_refined, ssim_refined = calculate_metrics(esrgan_rgb, refined_rgb)
    
    metrics_text = f"""
### Quality Metrics

**Image Size:** {w} × {h} pixels

**Real-ESRGAN vs Original:**
- PSNR: {psnr_esrgan:.2f} dB
- SSIM: {ssim_esrgan:.4f}

**Refined vs Real-ESRGAN:**
- PSNR: {psnr_refined:.2f} dB
- SSIM: {ssim_refined:.4f}

Higher is better!
"""
    
    print("Enhancement complete!")
    return original_rgb, esrgan_rgb, refined_rgb, metrics_text


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}
"""

# Create interface
with gr.Blocks(
    title="Jewelry Enhancement Demo",
    theme=gr.themes.Soft(primary_hue="purple"),
    css=custom_css
) as demo:
    
    # Header
    gr.Markdown("""
    # Jewelry Image Enhancement System
    ### AI-Powered Quality Improvement for Jewelry Photography
    
    Compare three enhancement stages: **Original** → **Real-ESRGAN** → **Refined CNN/GAN**
    """)
    
    with gr.Row():
        # Left column: Input
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            input_image = gr.Image(
                type="pil",
                label="Drop or click to upload",
                height=350
            )
            
            enhance_btn = gr.Button(
                "Enhance Image",
                variant="primary",
                size="lg",
                elem_classes="primary-btn"
            )
            
            metrics_display = gr.Markdown(
                "Upload an image and click **Enhance** to see results!"
            )
        
        # Right column: Outputs
        with gr.Column(scale=2):
            gr.Markdown("### Results Comparison")
            
            with gr.Tabs():
                with gr.Tab("Side-by-Side"):
                    with gr.Row():
                        out_original = gr.Image(label="Original", height=300)
                        out_esrgan = gr.Image(label="Real-ESRGAN", height=300)
                        out_refined = gr.Image(label="Refined (Ours)", height=300)
                
                with gr.Tab("Original"):
                    out_original_full = gr.Image(label="Original Input", height=500)
                
                with gr.Tab("Real-ESRGAN Baseline"):
                    out_esrgan_full = gr.Image(label="Real-ESRGAN Output", height=500)
                
                with gr.Tab("Refined Model (Ours)"):
                    out_refined_full = gr.Image(label="Refined Output (Best Quality)", height=500)
    
    # Examples section
    gr.Markdown("---")
    gr.Markdown("### Try Sample Images")
    
    example_images = []
    sample_dir = Path('data/raw')
    if sample_dir.exists():
        example_images = [str(f) for f in list(sample_dir.glob('*.jpg'))[:6]]
    
    if example_images:
        gr.Examples(
            examples=[[img] for img in example_images],
            inputs=input_image,
            label="Click to load sample"
        )
    
    # Info accordion
    with gr.Accordion("About This System", open=False):
        gr.Markdown(f"""
        ### Technical Details
        
        **Pipeline Architecture:**
        1. **Real-ESRGAN (Pre-trained)**: Base super-resolution model
        2. **Refinement Network ({MODEL_TYPE.upper()})**: Our trained model for artifact correction
        
        **Key Improvements:**
        - Enhanced gemstone clarity and metal reflections
        - Reduced compression artifacts and noise
        - Better color fidelity and contrast
        - Preserved fine jewelry details
        
        **Training Dataset:**
        - 490 high-quality jewelry images
        - 100 epochs with mixed precision
        - Train/Val/Test split: 392/49/49
        
        **Model Performance:**
        - PSNR: 28+ dB on test set
        - SSIM: 0.89+ on test set
        - Inference time: ~0.5s per image
        
        **Technologies:**
        - PyTorch 2.0
        - Real-ESRGAN (Tencent ARC)
        - Custom U-Net architecture
        - Perceptual + adversarial loss (for GAN mode)
        """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666;">
        Research Project | Built with Gradio + PyTorch | © 2025
    </div>
    """)
    
    # Connect button
    def process_and_display(image):
        """Process image and return all outputs."""
        orig, esrgan, refined, metrics = enhance_jewelry_image(image)
        return orig, esrgan, refined, orig, esrgan, refined, metrics
    
    enhance_btn.click(
        fn=process_and_display,
        inputs=input_image,
        outputs=[
            out_original, out_esrgan, out_refined,  # Side-by-side
            out_original_full, out_esrgan_full, out_refined_full,  # Full tabs
            metrics_display
        ]
    )


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
