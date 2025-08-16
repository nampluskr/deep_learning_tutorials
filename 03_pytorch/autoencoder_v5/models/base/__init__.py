"""
Base components for anomaly detection models.

This module provides common building blocks, utilities, and feature extractors
that can be shared across different anomaly detection approaches.
"""

from .building_blocks import ConvBlock, DeconvBlock
from .feature_extractors import (
    PretrainedEncoder, 
    ResNetEncoder, 
    VGGEncoder, 
    EfficientNetEncoder,
    get_pretrained_encoder
)
from .utils import (
    calculate_conv_output_size,
    calculate_deconv_output_size,
    get_final_conv_size,
    count_parameters,
    model_summary
)

__all__ = [
    # Building blocks
    'ConvBlock',
    'DeconvBlock',
    
    # Feature extractors
    'PretrainedEncoder',
    'ResNetEncoder', 
    'VGGEncoder',
    'EfficientNetEncoder',
    'get_pretrained_encoder',
    
    # Utilities
    'calculate_conv_output_size',
    'calculate_deconv_output_size', 
    'get_final_conv_size',
    'count_parameters',
    'model_summary'
]