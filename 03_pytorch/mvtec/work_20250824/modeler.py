import torch
from model_ae import VanillaEncoder, VanillaDecoder, VanillaAE
from model_ae import VAEEncoder, VAEDecoder, VAE
from model_ae import (
    mse_loss, bce_loss, combined_loss, vae_loss,
    psnr_metric, ssim_metric, mae_metric, binary_accuracy, lpips_metric)
# from model_fastflow import (FastFlow, fastflow_loss,
#     fastflow_log_prob_metric, fastflow_anomaly_score_metric)
# from model_patchcore import (PatchCore, patchcore_loss, patchcore_anomaly_score_metric)
# from model_stfpm import STFPM, stfpm_loss, stfpm_anomaly_score_metric


class Modeler:
    """Wrapper for model"""
    def __init__(self, model, device=None):
        self.model = model if device is None else model.to(device)
        self.device = next(self.model.parameters()).device
        self.metrics = self.model.get_metrics()

    def train_step(self, inputs, optimizer):
        self.model.train()
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.device != self.device:
                inputs[k] = v.to(self.device, non_blocking=True)

        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.model.compute_loss(outputs)
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        with torch.no_grad():
            results.update(self.model.compute_metrics(outputs))
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.eval()
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.device != self.device:
                inputs[k] = v.to(self.device, non_blocking=True)
                
        outputs = self.model(inputs)
        loss = self.model.compute_loss(outputs)

        results = {'loss': loss.item()}
        results.update(self.model.compute_metrics(outputs))
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.device != self.device:
                inputs[k] = v.to(self.device, non_blocking=True)
        
        outputs = self.model(inputs)
        scores = self.model.compute_anomaly_scores(outputs)
        return scores


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

    # elif model_name == 'fastflow':
    #     model = FastFlow(
    #         backbone=model_params.get('backbone', 'resnet18'),
    #         layers=model_params.get('layers', ['layer2', 'layer3']),
    #         flow_steps=model_params.get('flow_steps', 8),
    #         hidden_dim=model_params.get('hidden_dim', 512),
    #         weights_path=model_params.get('weights_path', None),
    #     )

    # elif model_name == 'patchcore':
    #     model = PatchCore(
    #         backbone=model_params.get('backbone', 'resnet18'),
    #         layers=model_params.get('layers', ['layer2','layer3']),
    #         memory_reduction=model_params.get('memory_reduction', 0.1),
    #         patch_size=model_params.get('patch_size', 32)
    #     )

    # elif model_name == 'stfpm':
    #     model = STFPM(
    #         backbone=model_params.get('backbone','resnet18'),
    #         layers=model_params.get('layers',['layer1','layer2','layer3']),
    #         weights_path=model_params.get('weights_path', None)
    #     )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available_models}")

    return model

if __name__ == "__main__":
    pass