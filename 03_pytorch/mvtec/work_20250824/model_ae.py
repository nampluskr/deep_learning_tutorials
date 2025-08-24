import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    """Basic deconvolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv_block(x)


class VanillaEncoder(nn.Module):
    """Vanilla CNN encoder for autoencoder"""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        return latent, features


class VanillaDecoder(nn.Module):
    """Vanilla CNN decoder for autoencoder with dynamic img_size"""
    def __init__(self, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.start_size = img_size // 32  # Encoder downsampling factor (5 conv blocks)

        self.fc = nn.Linear(latent_dim, 512 * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (512, self.start_size, self.start_size))

        layers = [
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
        ]
        # 최종 upsampling
        layers.append(
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.Sigmoid())

        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VanillaAE(nn.Module):
    """Vanilla autoencoder combining encoder and decoder"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = "vanilla_ae"

    def forward(self, inputs):
        x = inputs["image"]
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'features': features,
            'input': x
        }

    def compute_loss(self, outputs):
        preds = outputs['reconstructed']
        targets = outputs['input']
        return combined_loss(preds, targets)

    @torch.no_grad()
    def compute_metrics(self, outputs):
        preds = outputs['reconstructed']
        targets = outputs['input']
        return {
            'psnr': psnr_metric(preds, targets).item(),
            'ssim': ssim_metric(preds, targets).item(),
        }

    @torch.no_grad()
    def compute_anomaly_scores(self, outputs):
        preds = outputs["reconstructed"]
        targets = outputs["input"]
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])


class VAEEncoder(nn.Module):
    """VAE encoder with reparameterization trick"""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar, features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """VAE decoder identical to vanilla decoder but dynamic"""
    def __init__(self, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.start_size = img_size // 32  # Downsampling factor

        self.fc = nn.Linear(latent_dim, 512 * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (512, self.start_size, self.start_size))

        layers = [
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
        ]
        layers.append(
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.Sigmoid())

        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VAE(nn.Module):
    """Variational autoencoder with KL divergence loss"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = "vae"

    def forward(self, inputs):
        x = inputs["image"]
        mu, logvar, features = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'features': features,
            'input': x
        }

    def compute_loss(self, outputs):
        preds = outputs['reconstructed']
        targets = outputs['input']
        mu, logvar = outputs['mu'], outputs['logvar']
        return vae_loss(preds, targets, mu, logvar)

    @torch.no_grad()
    def compute_metrics(self, outputs):
        preds = outputs['reconstructed']
        targets = outputs['input']
        mu, logvar = outputs['mu'], outputs['logvar']
        return {
            'psnr': psnr_metric(preds, targets).item(),
            'ssim': ssim_metric(preds, targets).item(),
            'vae_loss': vae_loss(preds, targets, mu, logvar).item()
        }

    @torch.no_grad()
    def compute_anomaly_scores(self, outputs):
        preds = outputs['reconstructed']
        targets = outputs['input']
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])


# Loss Functions
def mse_loss(pred, target, reduction='mean'):
    """Mean Squared Error loss for reconstruction"""
    return F.mse_loss(pred, target, reduction=reduction)


def bce_loss(pred, target, reduction='mean'):
    """Binary Cross Entropy loss for reconstruction"""
    return F.binary_cross_entropy(pred, target, reduction=reduction)


def combined_loss(pred, target, mse_weight=0.5, ssim_weight=0.5, reduction='mean'):
    """Combined MSE and SSIM loss for better reconstruction quality"""
    mse = F.mse_loss(pred, target, reduction=reduction)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=(reduction == 'mean'))
    ssim_loss = 1 - ssim_val
    return mse_weight * mse + ssim_weight * ssim_loss


def vae_loss(pred, target, mu, logvar, beta=1.0, mse_weight=1.0):
    """VAE loss combining reconstruction and KL divergence"""
    recon_loss = F.mse_loss(pred, target, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / mu.size(0)
    return mse_weight * recon_loss + beta * kl_loss


# Metrics
def psnr_metric(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio metric"""
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def ssim_metric(pred, target, data_range=1.0):
    """Structural Similarity Index metric"""
    return ssim(pred, target, data_range=data_range, size_average=True).item()


def mae_metric(pred, target):
    """Mean Absolute Error metric"""
    return F.l1_loss(pred, target, reduction='mean').item()


_lpips_cache = {}

def lpips_metric(pred, target, net='alex'):
    """Learned Perceptual Image Patch Similarity (requires lpips package)"""
    if net not in _lpips_cache:
        _lpips_cache[net] = lpips.LPIPS(net=net)
    loss_fn = _lpips_cache[net].to(pred.device)
    return loss_fn(pred, target).mean().item()


def binary_accuracy(pred, target, threshold=0.5):
    """Binary accuracy for reconstruction quality assessment"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    return (pred_binary == target_binary).float().mean().item()


if __name__ == "__main__":
    pass