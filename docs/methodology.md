# Methodology

## Overview

This document describes the technical methodology used in the Jewelry Image Enhancement Pipeline, including degradation modeling, enhancement architecture, and evaluation protocols.

## Problem Formulation

Given a high-quality jewelry image $I_{HR}$, we simulate degradation to create $I_{LR}$ and then apply enhancement to produce $I_{SR}$. The goal is to maximize similarity between $I_{SR}$ and $I_{HR}$.

## Degradation Pipeline

### Mathematical Model

The degradation process follows a second-order model:

$$I_{LR} = D(I_{HR}) = C_2(N_2(S(C_1(N_1(B(I_{HR}))))))$$

Where:
- $B$: Blur operation (Gaussian or motion)
- $N_1$: First-stage noise (Gaussian + salt-and-pepper)
- $C_1$: First JPEG compression
- $S$: Downsampling operation
- $N_2$: Second-stage noise
- $C_2$: Second JPEG compression

### Blur Operations

**Gaussian Blur:**
$$I_{blur} = I \ast G_{\sigma}$$

where $G_{\sigma}$ is a Gaussian kernel with standard deviation $\sigma \in [0.5, 4.0]$ depending on degradation level.

**Motion Blur:**
$$I_{motion} = I \ast K_{length, angle}$$

where $K$ is a motion kernel with length $\in [3, 20]$ pixels and angle $\in [0, 180]$ degrees.

### Noise Operations

**Gaussian Noise:**
$$I_{noisy} = I + \mathcal{N}(0, \sigma^2)$$

with $\sigma \in [5, 40]$ depending on level.

**Salt-and-Pepper Noise:**
Random pixels set to 0 or 255 with probability $p \in [0.001, 0.015]$.

### Compression

JPEG compression with quality factor $q \in [10, 85]$:
- Level 1 (Mild): $q \in [70, 85]$
- Level 2 (Moderate): $q \in [40, 70]$
- Level 3 (Severe): $q \in [10, 40]$

### Color Shifts

Temperature shift in LAB color space:
$$b_{new} = b + \delta_T$$

where $\delta_T \in [-30, 30]$.

Saturation shift in HSV color space:
$$s_{new} = \text{clip}(s + \delta_S, 0, 255)$$

where $\delta_S \in [-15, 15]$.

## Enhancement Pipeline

### Architecture

The enhancement pipeline consists of three stages:

1. **Preprocessing**: $I_{pre} = P(I_{LR})$
2. **Core Enhancement**: $I_{enh} = E(I_{pre})$
3. **Postprocessing**: $I_{SR} = Q(I_{enh})$

### Preprocessing Operations

**Non-Local Means Denoising:**
Removes noise while preserving edges using patch-based similarity.

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**
Enhances local contrast in LAB color space with clip limit = 3.0.

### Core Enhancement: Real-ESRGAN

Real-ESRGAN uses a Residual-in-Residual Dense Block (RRDB) architecture:

**Network Structure:**
- Input: 3-channel RGB image
- Feature extraction: 64 feature channels
- RRDB blocks: 23 blocks with dense connections
- Upsampling: 4x using pixel shuffle
- Output: 3-channel RGB image (4x resolution)

**Training Details (Pre-trained Model):**
- Loss: L1 + Perceptual + GAN loss
- Training data: Diverse real-world degradations
- Optimization: Adam optimizer
- Batch size: Variable (tile-based inference)

**Tile-Based Processing:**
For memory efficiency, large images are processed in tiles:
- Tile size: 400×400 pixels
- Tile padding: 10 pixels (to avoid edge artifacts)
- Blending: Smooth blending at tile boundaries

### Postprocessing Operations

**Metallic Tone Enhancement:**
Boost saturation and brightness in HSV color space to enhance jewelry appearance.

**Bilateral Filtering:**
$$I_{filtered}(x) = \frac{1}{W_p} \sum_{y \in \Omega} G_{\sigma_s}(\|x-y\|) G_{\sigma_r}(|I(x)-I(y)|) I(y)$$

Preserves edges while smoothing background.

**Final Sharpening:**
Unsharp mask with amount = 0.3:
$$I_{sharp} = I + \alpha (I - G_{\sigma}(I))$$

## Evaluation Metrics

### Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 10 \log_{10} \left( \frac{MAX^2}{MSE} \right)$$

where $MAX = 255$ for 8-bit images and:

$$MSE = \frac{1}{HWC} \sum_{i,j,c} (I_{ref}(i,j,c) - I_{test}(i,j,c))^2$$

Higher is better (typically 20-40 dB range).

### Structural Similarity Index (SSIM)

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Range: [0, 1], higher is better (>0.8 is good).

### Sharpness Metric

Laplacian variance:
$$\text{Sharpness} = \text{Var}(\nabla^2 I)$$

Higher values indicate sharper images.

## Implementation Details

### Software Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch 2.0+
- **Image Processing**: OpenCV 4.8+
- **Metrics**: scikit-image
- **Interface**: Gradio 4.0+

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB

**Recommended:**
- GPU: NVIDIA T4 or better (4GB+ VRAM)
- RAM: 16 GB
- Storage: 20 GB

### Processing Performance

**On NVIDIA T4:**
- Single image (512×512): ~0.8s
- Batch processing: ~0.6s per image
- Memory usage: ~4GB GPU RAM

**On CPU (M4 Air):**
- Single image (512×512): ~5-8s
- Memory usage: ~2GB RAM

## Reproducibility

### Random Seeds

Set seeds for reproducibility:
