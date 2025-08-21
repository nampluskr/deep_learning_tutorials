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


def get_model(model_name, **model_params):
    """Factory function to create autoencoder and flow-based models"""
    available_models = ['ae', 'vanilla_ae', 'vae', 'fastflow']
    model_name = model_name.lower()

    params = {'in_channels': 3, 'out_channels': 3, 'latent_dim': 512}
    params.update(model_params)

    if model_name in ['ae', 'vanilla_ae']:
        encoder = VanillaEncoder(params['in_channels'], params['latent_dim'])
        decoder = VanillaDecoder(params['out_channels'], params['latent_dim'])
        model = VanillaAE(encoder, decoder)

    elif model_name == 'vae':
        encoder = VAEEncoder(params['in_channels'], params['latent_dim'])
        decoder = VAEDecoder(params['out_channels'], params['latent_dim'])
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
    pass