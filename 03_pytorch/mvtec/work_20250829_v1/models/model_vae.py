# models/model_vae.py
# Variational Autoencoder models (Vanilla, UNet-style, Backbone) + anomaly_map

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import EncoderBlock, DecoderBlock, ResNetFeatureExtractor


# -------------------------
# Vanilla VAE
# -------------------------

class VanillaVAE(nn.Module):
    """Simple Variational Autoencoder"""

    def __init__(self, in_channels=3, img_size=256, latent_dim=128):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        feat_size = img_size // 16
        self.fc_mu = nn.Linear(512 * feat_size * feat_size, latent_dim)
        self.fc_logvar = nn.Linear(512 * feat_size * feat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * feat_size * feat_size)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, in_channels, final_layer=True)

        self.feat_size = feat_size

    def encode(self, x):
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, self.feat_size, self.feat_size)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        return self.dec4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# -------------------------
# UNet-style VAE
# -------------------------

class UNetVAE(nn.Module):
    """UNet-style Variational Autoencoder with skip connections"""

    def __init__(self, in_channels=3, img_size=256, latent_dim=128):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        feat_size = img_size // 16
        self.fc_mu = nn.Linear(512 * feat_size * feat_size, latent_dim)
        self.fc_logvar = nn.Linear(512 * feat_size * feat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * feat_size * feat_size)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256 + 256, 128)
        self.dec3 = DecoderBlock(128 + 128, 64)
        self.dec4 = DecoderBlock(64 + 64, in_channels, final_layer=True)

        self.feat_size = feat_size

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        h = e4.view(e4.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h), (e1, e2, e3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        e1, e2, e3 = skips
        h = self.fc_dec(z).view(z.size(0), 512, self.feat_size, self.feat_size)
        d1 = self.dec1(h)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        d4 = self.dec4(torch.cat([d3, e1], dim=1))
        return d4

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar


# -------------------------
# ResNet Backbone VAEs
# -------------------------

class BackboneVanillaVAE(nn.Module):
    """VAE using ResNet backbone"""

    def __init__(self, backbone="resnet18", latent_dim=128):
        super().__init__()
        self.encoder = ResNetFeatureExtractor(backbone, layers=["layer1", "layer2", "layer3"])
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 16 * 16)

        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 3, final_layer=True)
        )

    def encode(self, x):
        feats = self.encoder(x)
        h = feats[-1].view(feats[-1].size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class BackboneUNetVAE(nn.Module):
    """UNet-style VAE with ResNet backbone"""

    def __init__(self, backbone="resnet18", latent_dim=128):
        super().__init__()
        self.encoder = ResNetFeatureExtractor(backbone, layers=["layer1", "layer2", "layer3"])
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 16 * 16)

        self.decoder1 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder3 = DecoderBlock(64, 3, final_layer=True)

    def encode(self, x):
        feats = self.encoder(x)
        h = feats[-1].view(feats[-1].size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h), feats[:-1]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 16, 16)
        d1 = self.decoder1(h)
        d2 = self.decoder2(d1)
        d3 = self.decoder3(d2)
        return d3

    def forward(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# -------------------------
# Loss & Anomaly Map
# -------------------------

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss


def compute_anomaly_map(x: torch.Tensor, recon: torch.Tensor):
    """Compute reconstruction error map"""
    anomaly_map = torch.mean((x - recon) ** 2, dim=1, keepdim=True)
    score = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
    return anomaly_map, score
