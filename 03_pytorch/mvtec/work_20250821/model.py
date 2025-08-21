from model_ae import VanillaEncoder, VanillaDecoder, VanillaAE
from model_ae import VAEEncoder, VAEDecoder, VAE
from model_ae import (
    mse_loss, bce_loss, combined_loss, vae_loss,
    psnr_metric, ssim_metric, mae_metric, binary_accuracy, lpips_metric
)
import torch.nn as nn


def get_model(model_name, **model_params):
    """Factory function to create autoencoder models"""
    available_models = ['ae', 'vanilla_ae', 'vae']
    default_params = {
        'in_channels': 3,
        'out_channels': 3,
        'latent_dim': 512
    }
    params = {**default_params, **model_params}
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    latent_dim = params['latent_dim']
    model_name = model_name.lower()

    if model_name in ['ae', 'vanilla_ae']:
        encoder = VanillaEncoder(in_channels=in_channels, latent_dim=latent_dim)
        decoder = VanillaDecoder(out_channels=out_channels, latent_dim=latent_dim)
        model = VanillaAE(encoder, decoder)
    elif model_name == 'vae':
        encoder = VAEEncoder(in_channels=in_channels, latent_dim=latent_dim)
        decoder = VAEDecoder(out_channels=out_channels, latent_dim=latent_dim)
        model = VAE(encoder, decoder)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available_models}")

    return model


def get_loss_fn(loss_name, **loss_params):
    """Factory function to create loss functions"""
    available_losses = ['mse', 'bce', 'combined', 'vae']
    loss_name = loss_name.lower()

    if loss_name == 'mse':
        return lambda pred, target: mse_loss(pred, target, **loss_params)

    elif loss_name == 'bce':
        return lambda pred, target: bce_loss(pred, target, **loss_params)

    elif loss_name == 'combined':
        return lambda pred, target: combined_loss(pred, target, **loss_params)

    elif loss_name == 'vae':
        return lambda pred, target, mu, logvar: vae_loss(pred, target, mu, logvar, **loss_params)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Available losses: {available_losses}")


def get_metric(metric_name, **metric_params):
    """Factory function to create a single metric function"""
    available_metrics = ['psnr', 'ssim', 'mae', 'binary_accuracy',
                         'lpips', 'mse', 'bce', 'combined', 'vae']   ]
    metric_name = metric_name.lower()

    if metric_name == 'psnr':
        return lambda pred, target: psnr_metric(pred, target, **metric_params)

    elif metric_name == 'ssim':
        return lambda pred, target: ssim_metric(pred, target, **metric_params)

    elif metric_name == 'mae':
        return lambda pred, target: mae_metric(pred, target, **metric_params)

    elif metric_name == 'binary_accuracy':
        return lambda pred, target: binary_accuracy(pred, target, **metric_params)

    elif metric_name == 'lpips':
        return lambda pred, target: lpips_metric(pred, target, **metric_params)

    elif metric_name == 'mse':
        return lambda pred, target: mse_loss(pred, target, **metric_params)

    elif metric_name == 'bce':
        return lambda pred, target: bce_loss(pred, target, **metric_params)

    elif metric_name == 'combined':
        return lambda pred, target: combined_loss(pred, target, **metric_params)

    elif metric_name == 'vae':
        return lambda pred, target, mu, logvar: vae_loss(pred, target, mu, logvar, **metric_params)

    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {available_metrics}")


if __name__ == "__main__":
    pass