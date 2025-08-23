"""
Advanced Anomaly Detection

- SSIM Autoencoder: 구조적 유사성 중심
- Memory-Augmented Networks: PaDiM, MemAE 등
- Feature-based Methods: PatchCore, SPADE 등
"""

from model_ae import VanillaEncoder, VanillaDecoder, VanillaAE
from model_ae import VAEEncoder, VAEDecoder, VAE
from model_ae import (
    mse_loss, bce_loss, combined_loss, vae_loss,
    psnr_metric, ssim_metric, mae_metric, binary_accuracy, lpips_metric
)
from model_fastflow import FastFlow, fastflow_loss, fastflow_log_prob_metric, fastflow_anomaly_score_metric
import torch.nn as nn


class Modeler:
    """Model composition manager that combines model, loss function, and metrics
    
    This class acts as a wrapper around the core model, providing a unified
    interface for model inference, loss computation, and metric evaluation.
    It encapsulates the model-specific configurations needed for training.
    """
    
    def __init__(self, model, loss_fn, metrics={}):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        
    def get_model_type(self):
        """Get model type from the wrapped model"""
        return getattr(self.model, 'model_type', 'unknown')
    
    def __repr__(self):
        return f"Modeler(model={self.get_model_type()}, metrics={list(self.metrics.keys())})"


def get_model(model_name, **model_params):
    """Factory function to create autoencoder and flow-based models"""
    available_models = ['ae', 'vanilla_ae', 'vae', 'fastflow']
    model_name = model_name.lower()

    params = {'in_channels': 3, 'out_channels': 3, 'latent_dim': 512}
    params.update(model_params)

    if model_name in ['ae', 'vanilla_ae']:
        encoder = VanillaEncoder(params['in_channels'], params['latent_dim'])
        decoder = VanillaDecoder(params['out_channels'], params['latent_dim'], params.get('img_size', 256))
        model = VanillaAE(encoder, decoder)

    elif model_name == 'vae':
        encoder = VAEEncoder(params['in_channels'], params['latent_dim'])
        decoder = VAEDecoder(params['out_channels'], params['latent_dim'], params.get('img_size', 256))
        model = VAE(encoder, decoder)

    elif model_name == 'fastflow':
        fastflow_params = {
            'backbone': 'resnet18',
            'layers': ['layer2', 'layer3'],
            'flow_steps': 8,
            'hidden_dim': 512,
            'weights_path': None
        }
        fastflow_params.update(model_params)

        model = FastFlow(
            backbone=fastflow_params['backbone'],
            layers=fastflow_params['layers'],
            flow_steps=fastflow_params['flow_steps'],
            hidden_dim=fastflow_params['hidden_dim'],
            weights_path=fastflow_params['weights_path']
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available_models}")

    return model


def get_modeler(model_name, loss_type=None, metric_names=[], **params):
    """Factory function to create Modeler with appropriate loss and metrics"""
    # Create model
    model = get_model(model_name, **params)
    
    # Get appropriate loss function
    if loss_type is None:
        # Auto-select loss based on model type
        if model_name.lower() == 'vae':
            loss_type = 'vae'
        elif model_name.lower() == 'fastflow':
            loss_type = 'fastflow'
        else:
            loss_type = 'combined'
    
    loss_fn = get_loss_fn(loss_type, **params)
    
    # Get appropriate metrics
    if not metric_names:
        # Auto-select metrics based on model type
        if model_name.lower() == 'vae':
            metric_names = ['mse', 'ssim', 'psnr', 'vae']
        elif model_name.lower() == 'fastflow':
            metric_names = ['fastflow_log_prob', 'fastflow_anomaly_score']
        else:
            metric_names = ['mse', 'ssim', 'psnr']
    
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = get_metric(metric_name, **params)
    
    return Modeler(model, loss_fn, metrics)


def get_loss_fn(loss_name, **loss_params):
    """Factory function to create loss functions"""
    available_losses = ['mse', 'bce', 'combined', 'vae', 'fastflow']
    loss_name = loss_name.lower()

    if loss_name == 'mse':
        params = {'reduction': 'mean'}
        params.update(loss_params)
        return lambda pred, target: mse_loss(pred, target, **params)

    elif loss_name == 'bce':
        params = {'reduction': 'mean'}
        params.update(loss_params)
        return lambda pred, target: bce_loss(pred, target, **params)

    elif loss_name == 'combined':
        params = {'mse_weight': 0.5, 'ssim_weight': 0.5, 'reduction': 'mean'}
        params.update(loss_params)
        return lambda pred, target: combined_loss(pred, target, **params)

    elif loss_name == 'vae':
        params = {'beta': 1.0, 'mse_weight': 1.0}
        params.update(loss_params)
        return lambda pred, target, mu, logvar: vae_loss(pred, target, mu, logvar, **params)

    elif loss_name == 'fastflow':
        params = {}
        params.update(loss_params)
        return lambda pred, target=None: fastflow_loss(pred, target, **params)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Available losses: {available_losses}")


def get_metric(metric_name, **metric_params):
    """Factory function to create a single metric function"""
    available_metrics = ['psnr', 'ssim', 'mae', 'binary_accuracy',
                         'lpips', 'mse', 'bce', 'combined', 'vae',
                         'fastflow_log_prob', 'fastflow_anomaly_score']
    metric_name = metric_name.lower()

    if metric_name == 'psnr':
        params = {'max_val': 1.0}
        params.update(metric_params)
        return lambda pred, target: psnr_metric(pred, target, **params)

    elif metric_name == 'ssim':
        params = {'data_range': 1.0}
        params.update(metric_params)
        return lambda pred, target: ssim_metric(pred, target, **params)

    elif metric_name == 'mae':
        params = {}
        params.update(metric_params)
        return lambda pred, target: mae_metric(pred, target, **params)

    elif metric_name == 'binary_accuracy':
        params = {'threshold': 0.5}
        params.update(metric_params)
        return lambda pred, target: binary_accuracy(pred, target, **params)

    elif metric_name == 'lpips':
        params = {'net': 'alex'}
        params.update(metric_params)
        return lambda pred, target: lpips_metric(pred, target, **params)

    elif metric_name == 'mse':
        params = {'reduction': 'mean'}
        params.update(metric_params)
        return lambda pred, target: mse_loss(pred, target, **params)

    elif metric_name == 'bce':
        params = {'reduction': 'mean'}
        params.update(metric_params)
        return lambda pred, target: bce_loss(pred, target, **params)

    elif metric_name == 'combined':
        params = {'mse_weight': 0.5, 'ssim_weight': 0.5, 'reduction': 'mean'}
        params.update(metric_params)
        return lambda pred, target: combined_loss(pred, target, **params)

    elif metric_name == 'vae':
        params = {'beta': 1.0, 'mse_weight': 1.0}
        params.update(metric_params)
        return lambda pred, target, mu, logvar: vae_loss(pred, target, mu, logvar, **params)

    elif metric_name == 'fastflow_log_prob':
        params = {}
        params.update(metric_params)
        return lambda pred, target=None: fastflow_log_prob_metric(pred, target, **params)

    elif metric_name == 'fastflow_anomaly_score':
        params = {}
        params.update(metric_params)
        return lambda pred, target=None: fastflow_anomaly_score_metric(pred, target, **params)

    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {available_metrics}")


if __name__ == "__main__":
    # Test Modeler
    import torch
    
    # Test different modelers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Modeler creation...")
    
    # Test Vanilla AE modeler
    vanilla_modeler = get_modeler(
        model_name='vanilla_ae',
        loss_type='combined',
        metric_names=['mse', 'ssim', 'psnr'],
        img_size=256
    )
    print(f"Vanilla AE modeler: {vanilla_modeler}")
    
    # Test VAE modeler
    vae_modeler = get_modeler(
        model_name='vae',
        loss_type='vae',
        metric_names=['mse', 'ssim', 'vae'],
        beta=1.0,
        img_size=256
    )
    print(f"VAE modeler: {vae_modeler}")
    
    # Test FastFlow modeler
    fastflow_modeler = get_modeler(
        model_name='fastflow',
        flow_steps=4,
        hidden_dim=256
    )
    print(f"FastFlow modeler: {fastflow_modeler}")
    
    print("Modeler testing completed!")