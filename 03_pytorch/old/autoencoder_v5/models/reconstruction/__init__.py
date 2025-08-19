"""
Reconstruction-based anomaly detection models.

This module contains autoencoder-based models that detect anomalies
by measuring reconstruction error between input and output.
"""

from .vanilla import VanillaAE, UnetAE
from .variational import VAE, BetaVAE, WAE
from .pretrained import ResNetAE, VGGAutoEncoder, EfficientNetAE

def get_reconstruction_model(model_type, **kwargs):
    """Factory function for reconstruction-based models"""
    
    model_registry = {
        # Vanilla autoencoders
        'vanilla_ae': VanillaAE,
        'unet_ae': UnetAE,
        
        # Variational autoencoders
        'vae': VAE,
        'beta_vae': BetaVAE,
        'wae': WAE,
        
        # Pretrained encoder autoencoders
        'resnet_ae': ResNetAE,
        'vgg_ae': VGGAutoEncoder,
        'efficientnet_ae': EfficientNetAE,
    }
    
    if model_type not in model_registry:
        available = ', '.join(model_registry.keys())
        raise ValueError(f"Unknown reconstruction model: {model_type}. "
                         f"Available: {available}")
    
    return model_registry[model_type](**kwargs)


__all__ = [
    # Models
    'VanillaAE',
    'UnetAE', 
    'VAE',
    'BetaVAE',
    'WAE',
    'ResNetAE',
    'VGGAutoEncoder',
    'EfficientNetAE',
    
    # Factory function
    'get_reconstruction_model'
]