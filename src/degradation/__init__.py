"""
Image degradation module for simulating real-world quality loss.
"""

from .blur import BlurDegradation
from .noise import NoiseDegradation
from .compression import CompressionDegradation
from .color_shift import ColorShiftDegradation
from .pipeline import DegradationPipeline

__all__ = [
    'BlurDegradation',
    'NoiseDegradation',
    'CompressionDegradation',
    'ColorShiftDegradation',
    'DegradationPipeline',
]
