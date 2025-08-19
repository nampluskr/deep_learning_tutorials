"""
Anomaly Detection Package

A comprehensive package for anomaly detection with focus on OLED display defects.
Supports various approaches including reconstruction-based, memory-based, and flow-based methods.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports - Main API
from .config import Config, get_preset_config
from .models import get_model, list_available_models
from .data import get_dataloaders, get_transforms, MVTecDataset
from .training import Trainer, Evaluator
from .experiments import ExperimentRunner, run_experiment

# Convenience imports
from .config.presets import (
    VANILLA_BASELINE, UNET_ADVANCED, VAE_DISENTANGLED,
    RESNET_PRETRAINED, OLED_OPTIMIZED, MULTI_CATEGORY_ENSEMBLE
)

# Quick start function
def quick_start(preset='oled_optimized', data_dir=None, category='bottle', **kwargs):
    """
    Quick start function for running anomaly detection experiments
    
    Args:
        preset: Preset configuration name
        data_dir: Path to dataset directory
        category: Dataset category or list of categories
        **kwargs: Additional configuration overrides
    
    Returns:
        Experiment results dictionary
    """
    from .experiments import run_experiment
    
    config = get_preset_config(preset)
    
    if data_dir:
        config.data.data_dir = data_dir
    if category:
        config.data.category = category
    
    # Apply any additional overrides
    if kwargs:
        config.update_from_dict(kwargs)
    
    return run_experiment(config)


# Package-level constants
SUPPORTED_MODELS = [
    'vanilla_ae', 'unet_ae', 'vae', 'beta_vae', 'wae',
    'resnet_ae', 'vgg_ae', 'efficientnet_ae'
]

SUPPORTED_DATASETS = ['mvtec', 'custom']

EVALUATION_METHODS = ['mse', 'l1', 'l2', 'ssim', 'ms_ssim']

# Package configuration
DEFAULT_CONFIG = {
    'device': 'cuda',
    'random_seed': 42,
    'log_level': 'INFO'
}


def setup_package(config_dict=None):
    """Setup package-level configurations"""
    import logging
    from .utils.logging import setup_logging
    
    config = DEFAULT_CONFIG.copy()
    if config_dict:
        config.update(config_dict)
    
    # Setup logging
    setup_logging(level=config['log_level'])
    
    # Set random seeds
    import torch
    import numpy as np
    import random
    
    seed = config['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_package_info():
    """Get package information"""
    return {
        'version': __version__,
        'author': __author__,
        'supported_models': SUPPORTED_MODELS,
        'supported_datasets': SUPPORTED_DATASETS,
        'evaluation_methods': EVALUATION_METHODS
    }


# Expose main API
__all__ = [
    # Core classes
    'Config', 'MVTecDataset', 'Trainer', 'Evaluator', 'ExperimentRunner',
    
    # Factory functions
    'get_model', 'get_preset_config', 'get_dataloaders', 'get_transforms',
    
    # High-level functions
    'quick_start', 'run_experiment', 'list_available_models',
    
    # Constants
    'SUPPORTED_MODELS', 'SUPPORTED_DATASETS', 'EVALUATION_METHODS',
    
    # Presets
    'VANILLA_BASELINE', 'UNET_ADVANCED', 'VAE_DISENTANGLED',
    'RESNET_PRETRAINED', 'OLED_OPTIMIZED', 'MULTI_CATEGORY_ENSEMBLE',
    
    # Utilities
    'setup_package', 'get_package_info'
]


# Package initialization
try:
    setup_package()
except Exception as e:
    import warnings
    warnings.warn(f"Package setup failed: {e}")


if __name__ == "__main__":
    # Package self-test
    print(f"Anomaly Detection Package v{__version__}")
    print("Running self-test...")
    
    try:
        # Test imports
        from . import config, models, data, training
        print("✓ All modules imported successfully")
        
        # Test model creation
        model = get_model('vanilla_ae', latent_dim=128, input_size=64)
        print("✓ Model creation successful")
        
        # Test configuration
        config = get_preset_config('debug_fast')
        print("✓ Configuration loading successful")
        
        print("✓ Self-test completed successfully")
        
    except Exception as e:
        print(f"✗ Self-test failed: {e}")