"""
Neural network models for anomaly detection
Contains autoencoder architectures including Vanilla AE and U-Net style AE
"""

import torch.nn as nn
from autoencoder import VanillaAE, UnetAE


# =============================================================================
# Model Factory Functions
# =============================================================================

def get_model(model_type, **model_params):
    """Load and return the specified autoencoder model"""
    available_models = ['vanilla_ae', 'unet_ae']
    in_channels = model_params.get('in_channels', 3)
    out_channels = model_params.get('out_channels', 3)
    latent_dim = model_params.get('latent_dim', 512)

    if model_type == 'vanilla_ae':
        model = VanillaAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim
        )
    elif model_type == 'unet_ae':
        model = UnetAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available models: {available_models}")
    return model


# =============================================================================
# Utility Functions for Models
# =============================================================================

def show_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    print(f" > Model: {model.__class__.__name__} on {device}")
    print(f" > Parameters: {total_params:,}")
    print(f" > Size: {total_params*4 / 1024**2:.1f} MB")


import os
import torch
import json
from dataclasses import asdict
from config import get_config_prefix


def load_trained_model(model_path, device='cuda'):
    """Load trained model from checkpoint"""
    from models import get_model

    checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    config = checkpoint.get('config', None)
    model_type = checkpoint.get('model_type', 'vanilla_ae')

    if config:
        # Use config from checkpoint
        model = get_model(
            model_type,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            latent_dim=config.latent_dim
        )
    else:
        # Use default parameters if config not available
        model = get_model(model_type)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    print(f"Model type: {model_type}")

    return model, config


def save_model(model, config):
    """Save model and configuration to disk"""
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    prefix = get_config_prefix(config)
    save_dir = os.path.join(results_dir, prefix)
    os.makedirs(save_dir, exist_ok=True)

    # Save model state dictionary
    model_filename = prefix + "_model.pth"
    model_path = os.path.join(save_dir, model_filename)
    config.model_path = model_path
    print(f" > Model saved to ./results/.../{model_filename}")
    torch.save(model.state_dict(), model_path)

    # Save configuration
    config_filename = prefix + "_config.json"
    config_path = os.path.join(save_dir, config_filename)
    config.config_path = config_path
    print(f" > Config saved to ./results/.../{config_filename}")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)


def load_weights(model, model_path):
    """Load the model state"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    pass
