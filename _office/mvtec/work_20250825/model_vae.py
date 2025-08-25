import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ae import ConvBlock, DeconvBlock, psnr_metric, ssim_metric


# ===============================================================
# Encoder / Decoder
# ===============================================================

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
        pooled = self.pool(features).view(x.size(0), -1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar, features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """VAE decoder identical to Vanilla decoder"""
    def __init__(self, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        if img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        self.start_size = img_size // 32

        self.fc = nn.Linear(latent_dim, 512 * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (512, self.start_size, self.start_size))

        layers = [
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ]
        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        return self.deconv_blocks(x)


# ===============================================================
# Models
# ===============================================================

class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(out_channels, latent_dim, img_size)
        self.model_type = "vae"

    def forward(self, inputs):
        x = inputs["image"]
        mu, logvar, features = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return {"reconstructed": recon, "mu": mu, "logvar": logvar,
                "z": z, "features": features, "input": x}

    def compute_loss(self, outputs):
        recon = recon_loss(outputs["reconstructed"], outputs["input"])
        kl = kl_divergence(outputs["mu"], outputs["logvar"])
        return recon + kl

    def compute_metrics(self, outputs):
        return {"psnr": float(psnr_metric(outputs["reconstructed"], outputs["input"])),
                "ssim": float(ssim_metric(outputs["reconstructed"], outputs["input"]))}

    def get_metrics(self): return ["psnr", "ssim"]

    def compute_anomaly_scores(self, outputs):
        preds, targets = outputs["reconstructed"], outputs["input"]
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])


class BetaVAE(VAE):
    """Beta-VAE (KL divergence scaled by beta)"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256, beta=4.0):
        super().__init__(in_channels, out_channels, latent_dim, img_size)
        self.model_type = "beta_vae"
        self.beta = beta

    def compute_loss(self, outputs):
        recon = recon_loss(outputs["reconstructed"], outputs["input"])
        kl = kl_divergence(outputs["mu"], outputs["logvar"])
        return recon + self.beta * kl


class WAE(nn.Module):
    """Wasserstein Autoencoder with MMD penalty"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256, lambda_mmd=10.0):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(out_channels, latent_dim, img_size)
        self.model_type = "wae"
        self.latent_dim = latent_dim
        self.lambda_mmd = lambda_mmd

    def forward(self, inputs):
        x = inputs["image"]
        mu, logvar, features = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return {"reconstructed": recon, "mu": mu, "logvar": logvar,
                "z": z, "features": features, "input": x}

    def compute_loss(self, outputs):
        recon = recon_loss(outputs["reconstructed"], outputs["input"])
        z = outputs["z"]
        prior = torch.randn_like(z)
        mmd = mmd_penalty(z, prior)
        return recon + self.lambda_mmd * mmd

    def compute_metrics(self, outputs):
        return {"psnr": float(psnr_metric(outputs["reconstructed"], outputs["input"])),
                "ssim": float(ssim_metric(outputs["reconstructed"], outputs["input"]))}

    def get_metrics(self): return ["psnr", "ssim"]

    def compute_anomaly_scores(self, outputs):
        preds, targets = outputs["reconstructed"], outputs["input"]
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])


# ===============================================================
# Loss functions
# ===============================================================

def recon_loss(pred, target, reduction='mean'):
    return F.mse_loss(pred, target, reduction=reduction)


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def mmd_penalty(z, prior_samples):
    """MMD penalty for WAE"""
    def kernel(x, y):
        dim = x.size(1)
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        K = torch.exp(-0.5 * (rx.t() + rx - 2*xx) / dim) \
          + torch.exp(-0.5 * (ry.t() + ry - 2*yy) / dim) \
          - 2*torch.exp(-0.5 * (rx.t() + ry - 2*xy) / dim)
        return K.mean()
    return kernel(z, z) + kernel(prior_samples, prior_samples) - 2 * kernel(z, prior_samples)


def vae_loss(pred, target, mu, logvar, beta=1.0, mse_weight=1.0):
    """VAE loss combining reconstruction and KL divergence"""
    recon_loss = F.mse_loss(pred, target, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / mu.size(0)
    return mse_weight * recon_loss + beta * kl_loss
