"""
Jewellery Image Enhancement - Interactive Demo
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
# MAIN ENHANCEMENT FUNCTION
# ============================================================================

def enhance_jewelry_image(image):
    """
    Process image through complete pipeline with memory-efficient refinement.
    
    Returns:
        (original, real-esrgan, refined)
    """
    if image is None:
        return None, None, None
    
    # Convert PIL to numpy BGR
    img_np = np.array(image)
    if img_np.shape[2] == 4:  # RGBA → RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    h, w = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Step 1: Real-ESRGAN baseline
    print("Running Real-ESRGAN...")
    esrgan_output = esrgan.enhance(img_bgr)

    # Step 2: Refined model (memory-efficient)
    print("Running Refined CNN/GAN...")
    max_dim = 512
    eh, ew = esrgan_output.shape[:2]
    scale = min(max_dim / max(eh, ew), 1.0)
    if scale < 1:
        new_h, new_w = int(eh * scale), int(ew * scale)
        esrgan_resized = cv2.resize(esrgan_output, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        esrgan_resized = esrgan_output

    tensor = torch.from_numpy(esrgan_resized).permute(2, 0, 1).float()
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.unsqueeze(0).cuda()
    
    with torch.no_grad():
        refined_tensor = torch.clamp(refined_model(tensor), -1, 1)
    
    refined_output_resized = refined_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    refined_output_resized = ((refined_output_resized + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Upscale refined output to match ESRGAN output
    refined_output = cv2.resize(refined_output_resized, (esrgan_output.shape[1], esrgan_output.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
    
    # Resize ESRGAN and refined to original size
    esrgan_rgb = cv2.cvtColor(cv2.resize(esrgan_output, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    refined_rgb = cv2.cvtColor(cv2.resize(refined_output, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    
    original_rgb = img_np
    
    print("Enhancement complete!")
    return original_rgb, esrgan_rgb, refined_rgb

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}
"""

with gr.Blocks(
    title="Jewelry Enhancement Demo",
    theme=gr.themes.Soft(primary_hue="purple"),
    css=custom_css
) as demo:
    
    gr.Markdown("""
    # Jewellery Image Enhancement System
    
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            input_image = gr.Image(type="pil", label="Drop or click to upload", height=350)
            
            enhance_btn = gr.Button("Enhance Image", variant="primary", size="lg", elem_classes="primary-btn")
        
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
    
    # with gr.Accordion("About This System", open=False):
    #     gr.Markdown(f"""
    #     ### Technical Details
        
    #     **Pipeline Architecture:**
    #     1. **Real-ESRGAN (Pre-trained)**: Base super-resolution model
    #     2. **Refinement Network ({MODEL_TYPE.upper()})**: Our trained model for artifact correction
        
    #     **Key Improvements:**
    #     - Enhanced gemstone clarity and metal reflections
    #     - Reduced compression artifacts and noise
    #     - Better color fidelity and contrast
    #     - Preserved fine jewelry details
    #     """)
    
    # gr.Markdown("""
    # ---
    # <div style="text-align: center; color: #666;">
    #     Research Project | Built with Gradio + PyTorch | © 2025
    # </div>
    # """)
    
    def process_and_display(image):
        """Process image and return all outputs."""
        orig, esrgan, refined = enhance_jewelry_image(image)
        return orig, esrgan, refined, orig, esrgan, refined
    
    enhance_btn.click(
        fn=process_and_display,
        inputs=input_image,
        outputs=[
            out_original, out_esrgan, out_refined,
            out_original_full, out_esrgan_full, out_refined_full
        ]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
