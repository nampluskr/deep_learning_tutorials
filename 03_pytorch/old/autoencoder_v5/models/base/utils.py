"""
Utility functions for model architectures and analysis.

This module provides common utility functions for calculating sizes,
model analysis, and other helper functions used across different models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any


def calculate_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    """Calculate output size after convolution operation"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def calculate_deconv_output_size(input_size: int, kernel_size: int, stride: int, 
                               padding: int, output_padding: int = 0) -> int:
    """Calculate output size after deconvolution operation"""
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding


def get_final_conv_size(input_size: int, num_layers: int = 5) -> int:
    """Calculate final feature map size after multiple conv layers with stride 2"""
    size = input_size
    for _ in range(num_layers):
        size = calculate_conv_output_size(size, kernel_size=4, stride=2, padding=1)
    return size


def calculate_receptive_field(layers_info: List[Dict[str, int]]) -> int:
    """
    Calculate receptive field of a CNN architecture
    
    Args:
        layers_info: List of dicts with 'kernel_size', 'stride', 'padding' for each layer
    
    Returns:
        Receptive field size
    """
    rf = 1
    for layer in layers_info:
        kernel_size = layer['kernel_size']
        stride = layer['stride']
        rf = rf + (kernel_size - 1) * stride
    return rf


def get_output_padding_for_target_size(input_size: int, target_size: int, 
                                     kernel_size: int = 4, stride: int = 2, 
                                     padding: int = 1) -> int:
    """Calculate output padding needed to reach target size in deconvolution"""
    expected_size = calculate_deconv_output_size(input_size, kernel_size, stride, padding)
    return max(0, target_size - expected_size)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes"""
    total_params, _ = count_parameters(model)
    return total_params * 4 / 1024**2  # 4 bytes per float32 parameter


def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 256, 256),
                 device: str = 'cpu') -> Dict[str, Any]:
    """
    Generate comprehensive model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        device: Device to run test on
        
    Returns:
        Dictionary with model statistics
    """
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    
    summary = {
        'model_name': model.__class__.__name__,
        'input_size': input_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'forward_pass_success': False,
        'output_shapes': {},
        'memory_usage_mb': 0
    }
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            x = torch.randn(1, *input_size).to(device)
            
            # Measure memory before forward pass
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            if hasattr(model, 'forward') and len(list(model.parameters())) > 0:
                outputs = model(x)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    for i, output in enumerate(outputs):
                        if torch.is_tensor(output):
                            summary['output_shapes'][f'output_{i}'] = tuple(output.shape)
                elif torch.is_tensor(outputs):
                    summary['output_shapes']['output'] = tuple(outputs.shape)
                
                summary['forward_pass_success'] = True
                
                # Measure memory usage
                if device == 'cuda':
                    memory_bytes = torch.cuda.max_memory_allocated()
                    summary['memory_usage_mb'] = memory_bytes / 1024**2
            
        except Exception as e:
            summary['error'] = str(e)
    
    return summary


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 256, 256),
                       device: str = 'cpu') -> None:
    """Print formatted model summary"""
    summary = model_summary(model, input_size, device)
    
    print(f"{'='*60}")
    print(f"Model: {summary['model_name']}")
    print(f"{'='*60}")
    print(f"Input size: {summary['input_size']}")
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"Model size: {summary['model_size_mb']:.1f} MB")
    
    if summary['forward_pass_success']:
        print(f"Forward pass: ✓ SUCCESS")
        print("Output shapes:")
        for name, shape in summary['output_shapes'].items():
            print(f"  {name}: {shape}")
        if summary['memory_usage_mb'] > 0:
            print(f"GPU memory usage: {summary['memory_usage_mb']:.1f} MB")
    else:
        print(f"Forward pass: ✗ FAILED")
        if 'error' in summary:
            print(f"Error: {summary['error']}")
    
    print(f"{'='*60}")


def safe_interpolate(x: torch.Tensor, target: torch.Tensor, 
                    mode: str = 'bilinear') -> torch.Tensor:
    """Safely interpolate tensor to match target spatial size"""
    if x.size(-1) == target.size(-1) and x.size(-2) == target.size(-2):
        return x
    
    target_size = (target.size(-2), target.size(-1))
    return F.interpolate(x, size=target_size, mode=mode, align_corners=False)


def safe_concatenate(x: torch.Tensor, skip: torch.Tensor, 
                    dim: int = 1, mode: str = 'bilinear') -> torch.Tensor:
    """Safely concatenate tensors with automatic size matching"""
    if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
        skip = safe_interpolate(skip, x, mode)
    return torch.cat([x, skip], dim=dim)


def get_activation_function(activation: str) -> nn.Module:
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'elu': nn.ELU(inplace=True),
        'prelu': nn.PReLU(),
        'selu': nn.SELU(inplace=True)
    }
    
    if activation not in activations:
        available = ', '.join(activations.keys())
        raise ValueError(f"Unknown activation: {activation}. Available: {available}")
    
    return activations[activation]


def calculate_conv_transpose_params(input_size: int, output_size: int, 
                                  kernel_size: int = 4, stride: int = 2, 
                                  padding: int = 1) -> Dict[str, int]:
    """Calculate parameters for conv transpose to achieve target output size"""
    # Calculate required output padding
    expected_size = calculate_deconv_output_size(input_size, kernel_size, stride, padding)
    output_padding = max(0, output_size - expected_size)
    
    return {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_padding': output_padding,
        'expected_output_size': expected_size + output_padding
    }


def validate_input_size(input_size: int, min_size: int = 32) -> None:
    """Validate that input size is sufficient for the architecture"""
    if input_size < min_size:
        raise ValueError(f"Input size {input_size} is too small. Minimum size is {min_size}.")
    
    # Check if input size is power of 2 (recommended for many architectures)
    if not (input_size & (input_size - 1)) == 0:
        print(f"Warning: Input size {input_size} is not a power of 2. "
              f"This may cause size mismatches in some architectures.")


def init_weights(module: nn.Module, init_type: str = 'normal', init_gain: float = 0.02) -> None:
    """Initialize network weights"""
    classname = module.__class__.__name__
    
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(f'Initialization method {init_type} not implemented')
        
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(module.weight.data, 1.0, init_gain)
        nn.init.constant_(module.bias.data, 0.0)


def freeze_model(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze model parameters"""
    for param in model.parameters():
        param.requires_grad = not freeze


def get_lr_scheduler(optimizer, scheduler_type: str = 'plateau', **kwargs):
    """Get learning rate scheduler by type"""
    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == 'warmup_cosine':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions:")
    
    # Test size calculations
    input_size = 256
    output_size = calculate_conv_output_size(input_size, 4, 2, 1)
    print(f"Conv output size: {input_size} -> {output_size}")
    
    deconv_size = calculate_deconv_output_size(output_size, 4, 2, 1)
    print(f"Deconv output size: {output_size} -> {deconv_size}")
    
    final_size = get_final_conv_size(256, 5)
    print(f"Final conv size after 5 layers: 256 -> {final_size}")
    
    # Test parameter calculation
    output_padding = get_output_padding_for_target_size(8, 16)
    print(f"Output padding needed: {output_padding}")
    
    # Test model analysis (create a simple model)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, 1, 1)
            self.fc = nn.Linear(64, 128)
        
        def forward(self, x):
            x = self.conv(x)
            # Global average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    print_model_summary(model, input_size=(3, 64, 64))
    
    print("\nAll utility functions working correctly!")