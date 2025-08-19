"""
Unified models package for anomaly detection.

This package provides a comprehensive collection of anomaly detection models
including reconstruction-based, memory-based, flow-based, statistical, and
distillation-based approaches.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

# Import reconstruction models
from .reconstruction import (
    VanillaAE, UnetAE, VAE, BetaVAE, WAE, 
    ResNetAE, VGGAutoEncoder, EfficientNetAE,
    get_reconstruction_model
)

# Import base components for external use
from .base import (
    ConvBlock, DeconvBlock,
    PretrainedEncoder, get_pretrained_encoder,
    count_parameters, model_summary, print_model_summary
)


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Unified model factory for all anomaly detection methods.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Instantiated model
        
    Available Models:
        Reconstruction-based:
            - vanilla_ae: Standard autoencoder
            - unet_ae: UNet-style autoencoder with skip connections
            - vae: Variational autoencoder
            - beta_vae: β-VAE with adjustable β parameter
            - wae: Wasserstein autoencoder
            - resnet_ae: ResNet encoder + custom decoder
            - vgg_ae: VGG encoder + custom decoder
            - efficientnet_ae: EfficientNet encoder + custom decoder
    """
    
    # Model registry with categories
    model_registry = {
        # Reconstruction-based models
        'vanilla_ae': ('reconstruction', 'vanilla_ae'),
        'unet_ae': ('reconstruction', 'unet_ae'), 
        'vae': ('reconstruction', 'vae'),
        'beta_vae': ('reconstruction', 'beta_vae'),
        'wae': ('reconstruction', 'wae'),
        'resnet_ae': ('reconstruction', 'resnet_ae'),
        'vgg_ae': ('reconstruction', 'vgg_ae'),
        'efficientnet_ae': ('reconstruction', 'efficientnet_ae'),
        
        # Placeholder for future models
        # Memory-based models (future)
        # 'patchcore': ('memory_based', 'patchcore'),
        # 'spade': ('memory_based', 'spade'),
        
        # Flow-based models (future)
        # 'fastflow': ('flow_based', 'fastflow'),
        # 'differnet': ('flow_based', 'differnet'),
        
        # Statistical models (future)
        # 'padim': ('statistical', 'padim'),
        # 'stfpm': ('statistical', 'stfpm'),
        
        # Distillation-based models (future)
        # 'efficientad': ('distillation', 'efficientad'),
        # 'student_teacher': ('distillation', 'student_teacher'),
    }
    
    if model_type not in model_registry:
        available = ', '.join(sorted(model_registry.keys()))
        raise ValueError(f"Unknown model type: {model_type}. Available models: {available}")
    
    category, model_name = model_registry[model_type]
    
    # Route to appropriate factory function
    if category == 'reconstruction':
        return get_reconstruction_model(model_name, **kwargs)
    # elif category == 'memory_based':
    #     return get_memory_model(model_name, **kwargs)
    # elif category == 'flow_based':
    #     return get_flow_model(model_name, **kwargs)
    # elif category == 'statistical':
    #     return get_statistical_model(model_name, **kwargs)
    # elif category == 'distillation':
    #     return get_distillation_model(model_name, **kwargs)
    else:
        raise NotImplementedError(f"Model category '{category}' not yet implemented")


def list_available_models() -> Dict[str, List[str]]:
    """List all available models by category"""
    return {
        'reconstruction': [
            'vanilla_ae', 'unet_ae', 'vae', 'beta_vae', 'wae',
            'resnet_ae', 'vgg_ae', 'efficientnet_ae'
        ],
        'memory_based': [
            # 'patchcore', 'spade'  # Future implementation
        ],
        'flow_based': [
            # 'fastflow', 'differnet'  # Future implementation
        ],
        'statistical': [
            # 'padim', 'stfpm'  # Future implementation
        ],
        'distillation': [
            # 'efficientad', 'student_teacher'  # Future implementation
        ]
    }


def list_models_by_category(category: str) -> List[str]:
    """List models in a specific category"""
    available_models = list_available_models()
    if category not in available_models:
        available_categories = ', '.join(available_models.keys())
        raise ValueError(f"Unknown category: {category}. Available: {available_categories}")
    return available_models[category]


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get detailed information about a specific model"""
    
    model_info = {
        # Reconstruction models
        'vanilla_ae': {
            'category': 'reconstruction',
            'description': 'Standard encoder-decoder autoencoder',
            'paper': 'Classic autoencoder architecture',
            'strengths': ['Simple', 'Fast training', 'Good baseline'],
            'weaknesses': ['Limited expressiveness', 'Mode collapse'],
            'typical_params': {
                'in_channels': 3, 'out_channels': 3, 
                'latent_dim': 512, 'input_size': 256
            }
        },
        'unet_ae': {
            'category': 'reconstruction',
            'description': 'UNet-style autoencoder with skip connections',
            'paper': 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
            'strengths': ['Better detail preservation', 'Skip connections', 'Good for fine details'],
            'weaknesses': ['More parameters', 'Slower training'],
            'typical_params': {
                'in_channels': 3, 'out_channels': 3,
                'latent_dim': 512, 'input_size': 256
            }
        },
        'vae': {
            'category': 'reconstruction',
            'description': 'Variational autoencoder with probabilistic latent space',
            'paper': 'Auto-Encoding Variational Bayes (Kingma & Welling, 2014)',
            'strengths': ['Principled probabilistic model', 'Good generative capability'],
            'weaknesses': ['KL collapse', 'Blurry reconstructions'],
            'typical_params': {
                'in_channels': 3, 'out_channels': 3,
                'latent_dim': 512, 'input_size': 256
            }
        },
        'beta_vae': {
            'category': 'reconstruction',
            'description': 'β-VAE with adjustable β for disentanglement',
            'paper': 'β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework',
            'strengths': ['Disentangled representations', 'Controllable generation'],
            'weaknesses': ['Reconstruction quality trade-off', 'Hyperparameter sensitive'],
            'typical_params': {
                'in_channels': 3, 'out_channels': 3,
                'latent_dim': 512, 'input_size': 256, 'beta': 4.0
            }
        },
        'wae': {
            'category': 'reconstruction',
            'description': 'Wasserstein autoencoder with MMD regularization',
            'paper': 'Wasserstein Auto-Encoders (Tolstikhin et al., 2018)',
            'strengths': ['Stable training', 'Better mode coverage'],
            'weaknesses': ['MMD computation cost', 'Hyperparameter tuning'],
            'typical_params': {
                'in_channels': 3, 'out_channels': 3,
                'latent_dim': 512, 'input_size': 256, 'lambda_reg': 10.0
            }
        },
        'resnet_ae': {
            'category': 'reconstruction',
            'description': 'ResNet encoder with custom decoder',
            'paper': 'Deep Residual Learning for Image Recognition',
            'strengths': ['Transfer learning', 'Rich features', 'Pre-trained weights'],
            'weaknesses': ['Large model size', 'ImageNet bias'],
            'typical_params': {
                'arch': 'resnet50', 'pretrained': True, 'freeze_backbone': False,
                'latent_dim': 512, 'input_size': 224
            }
        },
        'vgg_ae': {
            'category': 'reconstruction',
            'description': 'VGG encoder with custom decoder',
            'paper': 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
            'strengths': ['Simple architecture', 'Good feature extraction'],
            'weaknesses': ['Memory intensive', 'Many parameters'],
            'typical_params': {
                'arch': 'vgg16', 'pretrained': True, 'freeze_backbone': False,
                'latent_dim': 512, 'input_size': 224
            }
        },
        'efficientnet_ae': {
            'category': 'reconstruction',
            'description': 'EfficientNet encoder with custom decoder',
            'paper': 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks',
            'strengths': ['Efficient architecture', 'Good accuracy/efficiency trade-off'],
            'weaknesses': ['Complex architecture', 'Version dependencies'],
            'typical_params': {
                'arch': 'efficientnet_b0', 'pretrained': True, 'freeze_backbone': False,
                'latent_dim': 512, 'input_size': 224
            }
        }
    }
    
    if model_type not in model_info:
        available = ', '.join(sorted(model_info.keys()))
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    return model_info[model_type]


def create_model_comparison_table() -> str:
    """Create a comparison table of all available models"""
    models = list_available_models()['reconstruction']
    
    table = []
    table.append("| Model | Category | Description | Strengths | Weaknesses |")
    table.append("|-------|----------|-------------|-----------|------------|")
    
    for model_type in models:
        info = get_model_info(model_type)
        strengths = ', '.join(info['strengths'][:2])  # Show first 2 strengths
        weaknesses = ', '.join(info['weaknesses'][:2])  # Show first 2 weaknesses
        
        table.append(f"| {model_type} | {info['category']} | {info['description']} | {strengths} | {weaknesses} |")
    
    return '\n'.join(table)


def load_model_from_checkpoint(checkpoint_path: str, model_type: str = None, **model_params) -> nn.Module:
    """
    Load model from checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model (if not in checkpoint)
        **model_params: Additional model parameters
        
    Returns:
        Loaded model with state dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get model type from checkpoint
    if model_type is None:
        model_type = checkpoint.get('model_type', None)
        if model_type is None:
            raise ValueError("Model type not found in checkpoint and not provided")
    
    # Get model parameters from checkpoint if available
    if 'model_params' in checkpoint:
        saved_params = checkpoint['model_params']
        # Update with any provided parameters
        saved_params.update(model_params)
        model_params = saved_params
    
    # Create model
    model = get_model(model_type, **model_params)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the entire checkpoint is the state dict
        model.load_state_dict(checkpoint)
    
    return model


def save_model_checkpoint(model: nn.Module, checkpoint_path: str, 
                         model_type: str = None, model_params: Dict = None,
                         **additional_info) -> None:
    """
    Save model checkpoint with metadata
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        model_type: Type of model
        model_params: Model parameters used for initialization
        **additional_info: Additional information to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'model_params': model_params,
        **additional_info
    }
    
    torch.save(checkpoint, checkpoint_path)


# Model aliases for convenience
AE = VanillaAE
UNet = UnetAE


__all__ = [
    # Main factory function
    'get_model',
    
    # Model information functions
    'list_available_models',
    'list_models_by_category', 
    'get_model_info',
    'create_model_comparison_table',
    
    # Checkpoint functions
    'load_model_from_checkpoint',
    'save_model_checkpoint',
    
    # Direct model imports
    'VanillaAE', 'UnetAE', 'VAE', 'BetaVAE', 'WAE',
    'ResNetAE', 'VGGAutoEncoder', 'EfficientNetAE',
    
    # Base components
    'ConvBlock', 'DeconvBlock',
    'PretrainedEncoder', 'get_pretrained_encoder',
    
    # Utilities
    'count_parameters', 'model_summary', 'print_model_summary',
    
    # Aliases
    'AE', 'UNet'
]


if __name__ == "__main__":
    # Test the unified model factory
    print("Testing unified model factory...")
    
    # Test model creation
    model_types = ['vanilla_ae', 'unet_ae', 'vae']
    
    for model_type in model_types:
        try:
            print(f"\nTesting {model_type}:")
            model = get_model(model_type, latent_dim=256, input_size=128)
            print(f"✓ {model.__class__.__name__} created successfully")
            
            # Test forward pass
            x = torch.randn(1, 3, 128, 128)
            with torch.no_grad():
                if model_type in ['vae', 'beta_vae']:
                    output = model(x)
                    print(f"  Output length: {len(output)}")
                elif model_type == 'wae':
                    output = model(x)
                    print(f"  Output length: {len(output)}")
                else:
                    output = model(x)
                    print(f"  Output shapes: {[o.shape if torch.is_tensor(o) else type(o) for o in output]}")
                    
        except Exception as e:
            print(f"✗ Error testing {model_type}: {e}")
    
    # Test model information
    print(f"\nAvailable models: {list_available_models()}")
    
    print(f"\nReconstruction models: {list_models_by_category('reconstruction')}")
    
    # Test model info
    print(f"\nVAE info: {get_model_info('vae')}")
    
    # Test comparison table
    print(f"\nModel comparison table:")
    print(create_model_comparison_table())
    
    print("\nUnified model factory test completed!")