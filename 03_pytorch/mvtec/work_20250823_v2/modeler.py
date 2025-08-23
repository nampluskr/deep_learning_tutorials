from model_ae import VanillaEncoder, VanillaDecoder, VanillaAE
from model_ae import VAEEncoder, VAEDecoder, VAE
from model_ae import (
    mse_loss, bce_loss, combined_loss, vae_loss,
    psnr_metric, ssim_metric, mae_metric, binary_accuracy, lpips_metric)
from model_fastflow import (FastFlow, fastflow_loss,
    fastflow_log_prob_metric, fastflow_anomaly_score_metric)
from model_patchcore import (PatchCore, patchcore_loss, patchcore_anomaly_score_metric)
from model_stfpm import STFPM, stfpm_loss, stfpm_anomaly_score_metric


class Modeler:
    """Wrapper for model, loss function, and metrics"""
    def __init__(self, model, loss_fn, metrics, device=None):
        self.model = model if device is None else model.to(device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = next(self.model.parameters()).device


def get_model(model_name, **model_params):
    """Factory function to create models"""
    available_models = ['ae', 'vanilla_ae', 'vae', 'fastflow']
    model_name = model_name.lower()

    params = {'in_channels': 3, 'out_channels': 3, 'latent_dim': 512}
    params.update(model_params)

    if model_name in ['ae', 'vanilla_ae']:
        encoder = VanillaEncoder(params['in_channels'], params['latent_dim'])
        decoder = VanillaDecoder(params['out_channels'], params['latent_dim'],
                                 params.get('img_size', 256))
        model = VanillaAE(encoder, decoder)

    elif model_name == 'vae':
        encoder = VAEEncoder(params['in_channels'], params['latent_dim'])
        decoder = VAEDecoder(params['out_channels'], params['latent_dim'],
                             params.get('img_size', 256))
        model = VAE(encoder, decoder)

    elif model_name == 'fastflow':
        model = FastFlow(
            backbone=model_params.get('backbone', 'resnet18'),
            layers=model_params.get('layers', ['layer2', 'layer3']),
            flow_steps=model_params.get('flow_steps', 8),
            hidden_dim=model_params.get('hidden_dim', 512),
            weights_path=model_params.get('weights_path', None),
        )

    elif model_name == 'patchcore':
        model = PatchCore(
            backbone=model_params.get('backbone', 'resnet18'),
            layers=model_params.get('layers', ['layer2','layer3']),
            memory_reduction=model_params.get('memory_reduction', 0.1),
            patch_size=model_params.get('patch_size', 32)
        )

    elif model_name == 'stfpm':
        model = STFPM(
            backbone=model_params.get('backbone','resnet18'),
            layers=model_params.get('layers',['layer1','layer2','layer3']),
            weights_path=model_params.get('weights_path', None)
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
        return lambda pred, target=None: fastflow_loss(pred)

    elif loss_name == 'patchcore':
        return lambda pred, target=None: patchcore_loss(pred)

    elif loss_name == 'stfpm':
        return lambda pred, target=None: stfpm_loss(pred, target)

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
        return lambda pred, target=None: fastflow_log_prob_metric(pred)

    elif metric_name == 'fastflow_anomaly_score':
        return lambda pred, target=None: fastflow_anomaly_score_metric(pred)

    elif metric_name == 'patchcore_anomaly_score':
        return lambda scores, target=None: patchcore_anomaly_score_metric(scores)

    elif metric_name == 'stfpm_anomaly_score':
        return lambda pred, target=None: stfpm_anomaly_score_metric(pred, target)

    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {available_metrics}")


if __name__ == "__main__":
    pass