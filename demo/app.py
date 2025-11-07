import sys
from pathlib import Path
import gradio as gr
import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.degradation.pipeline import DegradationPipeline
from src.enhancement.preprocessor import JewelryPreprocessor
from src.enhancement.enhancer import RealESRGANEnhancer
from src.enhancement.postprocessor import JewelryPostprocessor
from src.evaluation.metrics import ImageQualityMetrics


# Load configurations
with open("config/model_config.yaml", 'r') as f:
    model_config = yaml.safe_load(f)

with open("config/degradation_config.yaml", 'r') as f:
    degradation_config = yaml.safe_load(f)


# Initialize pipeline components (lazy loading for faster startup)
_preprocessor = None
_enhancer = None
_postprocessor = None
_degradation_pipeline = None
_metrics_calc = None


def get_preprocessor():
    """Lazy load preprocessor."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = JewelryPreprocessor(model_config.get('preprocessing', {}))
    return _preprocessor


def get_enhancer():
    """Lazy load enhancer."""
    global _enhancer
    if _enhancer is None:
        _enhancer = RealESRGANEnhancer(model_config)
    return _enhancer


def get_postprocessor():
    """Lazy load postprocessor."""
    global _postprocessor
    if _postprocessor is None:
        _postprocessor = JewelryPostprocessor(model_config.get('postprocessing', {}))
    return _postprocessor


def get_degradation_pipeline():
    """Lazy load degradation pipeline."""
    global _degradation_pipeline
    if _degradation_pipeline is None:
        _degradation_pipeline = DegradationPipeline("config/degradation_config.yaml")
    return _degradation_pipeline


def get_metrics_calc():
    """Lazy load metrics calculator."""
    global _metrics_calc
    if _metrics_calc is None:
        _metrics_calc = ImageQualityMetrics()
    return _metrics_calc


def enhance_image(input_image, apply_preprocessing, apply_postprocessing):
    """
    Enhance uploaded image.
    
    Args:
        input_image: Input image (numpy array in RGB format from Gradio)
        apply_preprocessing: Whether to apply preprocessing
        apply_postprocessing: Whether to apply postprocessing
        
    Returns:
        Enhanced image in RGB format
    """
    if input_image is None:
        return None
    
    try:
        # Convert RGB to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        # Preprocessing
        if apply_preprocessing:
            preprocessor = get_preprocessor()
            image_bgr = preprocessor.preprocess(image_bgr)
        
        # Enhancement
        enhancer = get_enhancer()
        enhanced_bgr = enhancer.enhance(image_bgr)
        
        # Postprocessing
        if apply_postprocessing:
            postprocessor = get_postprocessor()
            enhanced_bgr = postprocessor.postprocess(enhanced_bgr)
        
        # Convert BGR back to RGB for Gradio
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        return enhanced_rgb
        
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return input_image


def degrade_and_enhance(input_image, degradation_level):
    """
    Apply degradation then enhance image.
    
    Args:
        input_image: Input image (numpy array in RGB format)
        degradation_level: Degradation level to apply
        
    Returns:
        Tuple of (degraded image, enhanced image, metrics text)
    """
    if input_image is None:
        return None, None, "No image provided"
    
    try:
        # Convert RGB to BGR
        image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        # Apply degradation
        pipeline = get_degradation_pipeline()
        level_map = {
            "Mild": "level1_mild",
            "Moderate": "level2_moderate",
            "Severe": "level3_severe"
        }
        degraded_bgr = pipeline.apply_degradation(image_bgr, level_map[degradation_level])
        
        # Enhance degraded image
        preprocessor = get_preprocessor()
        enhancer = get_enhancer()
        postprocessor = get_postprocessor()
        
        preprocessed = preprocessor.preprocess(degraded_bgr)
        enhanced_bgr = enhancer.enhance(preprocessed)
        final_bgr = postprocessor.postprocess(enhanced_bgr)
        
        # Calculate metrics
        metrics_calc = get_metrics_calc()
        metrics = metrics_calc.calculate_all_metrics(image_bgr, final_bgr)
        
        metrics_text = f"""
Enhancement Metrics:
  PSNR: {metrics['psnr']:.2f} dB
  SSIM: {metrics['ssim']:.4f}
  Sharpness Gain: {metrics['sharpness_enhanced'] - metrics['sharpness_original']:.2f}
        """
        
        # Convert BGR to RGB
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        
        return degraded_rgb, enhanced_rgb, metrics_text
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return input_image, input_image, error_msg


# Create Gradio interface
with gr.Blocks(title="Jewelry Image Enhancement", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Jewelry Image Enhancement Pipeline
        
        Transform low-quality jewelry images into high-quality product photos using Real-ESRGAN and domain-specific processing.
        
        **Features:**
        - Pre-trained Real-ESRGAN 4x super-resolution
        - Jewelry-specific preprocessing (denoising, contrast enhancement)
        - Metallic tone enhancement for gold, silver, and gemstones
        - Synthetic degradation simulation for testing
        """
    )
    
    with gr.Tabs():
        # Tab 1: Direct Enhancement
        with gr.Tab("Enhance Image"):
            gr.Markdown("### Upload a low-quality jewelry image to enhance")
            
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="Upload Image", type="numpy")
                    preprocess_check = gr.Checkbox(label="Apply Preprocessing", value=True)
                    postprocess_check = gr.Checkbox(label="Apply Postprocessing", value=True)
                    enhance_btn = gr.Button("Enhance Image", variant="primary")
                
                with gr.Column():
                    output_img = gr.Image(label="Enhanced Image")
            
            enhance_btn.click(
                fn=enhance_image,
                inputs=[input_img, preprocess_check, postprocess_check],
                outputs=output_img
            )
            
            gr.Examples(
                examples=[],
                inputs=input_img,
                label="Example Images (Add your own)"
            )
        
        # Tab 2: Degradation + Enhancement Demo
        with gr.Tab("Degradation Demo"):
            gr.Markdown("### Test enhancement on synthetically degraded images")
            
            with gr.Row():
                with gr.Column():
                    demo_input_img = gr.Image(label="Upload High-Quality Image", type="numpy")
                    degradation_level = gr.Radio(
                        choices=["Mild", "Moderate", "Severe"],
                        value="Moderate",
                        label="Degradation Level"
                    )
                    demo_btn = gr.Button("Apply Degradation & Enhance", variant="primary")
                
                with gr.Column():
                    degraded_output = gr.Image(label="Degraded Image")
                    enhanced_output = gr.Image(label="Enhanced Image")
            
            metrics_output = gr.Textbox(label="Metrics", lines=5)
            
            demo_btn.click(
                fn=degrade_and_enhance,
                inputs=[demo_input_img, degradation_level],
                outputs=[degraded_output, enhanced_output, metrics_output]
            )
        
        # Tab 3: About
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## Technical Details
                
                ### Pipeline Architecture
                
                1. **Preprocessing**
                   - Non-local means denoising
                   - CLAHE contrast enhancement
                   - Optional sharpening
                
                2. **Core Enhancement**
                   - Real-ESRGAN 4x super-resolution
                   - Tile-based processing for large images
                   - GPU acceleration when available
                
                3. **Postprocessing**
                   - Metallic tone enhancement (HSV adjustment)
                   - Bilateral filtering for background cleanup
                   - Subtle sharpening for detail enhancement
                
                ### Synthetic Degradation Model
                
                The degradation pipeline simulates real-world quality loss:
                
                - **Blur**: Gaussian and motion blur
                - **Noise**: Gaussian and salt-and-pepper noise
                - **Compression**: JPEG artifacts
                - **Color**: Temperature and saturation shifts
                - **Resolution**: Downscaling and upscaling
                
                ### Performance
                
                - Processing time: ~0.8s per image on NVIDIA T4
                - Model size: ~64MB
                - Supported formats: JPG, JPEG, PNG
                
                ### References
                
                - Real-ESRGAN: [Paper](https://arxiv.org/abs/2107.10833)
                - BasicSR Framework: [GitHub](https://github.com/XPixelGroup/BasicSR)
                - Dataset: Tanishq Jewellery Dataset (Kaggle)
                
                ### Contact
                
                For questions or feedback, please open an issue on GitHub.
                """
            )
    
    gr.Markdown(
        """
        ---
        **Note:** First enhancement may take longer as the model loads. Subsequent enhancements will be faster.
        """
    )


if __name__ == "__main__":
    print("Starting Jewelry Enhancement Demo...")
    print("Loading models (this may take a moment)...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
