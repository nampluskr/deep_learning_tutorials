# models/model_autoencoder.py
# Autoencoder models (Vanilla, UNet-style, Backbone) + anomaly_map

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import EncoderBlock, DecoderBlock, ResNetFeatureExtractor


# -------------------------
# Vanilla Autoencoder
# -------------------------

class VanillaAE(nn.Module):
    """Simple Autoencoder with encoder-decoder symmetric architecture"""

    def __init__(self, in_channels=3, img_size=256, latent_dim=128):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        feat_size = img_size // 16
        self.fc1 = nn.Linear(512 * feat_size * feat_size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 512 * feat_size * feat_size)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, in_channels, final_layer=True)

        self.feat_size = feat_size

    def forward(self, x):
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)

        h = h.view(h.size(0), -1)
        z = self.fc1(h)
        h = self.fc2(z)
        h = h.view(h.size(0), 512, self.feat_size, self.feat_size)

        out = self.dec1(h)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.dec4(out)
        return out


# -------------------------
# UNet-style Autoencoder
# -------------------------

class UNetAE(nn.Module):
    """UNet-style Autoencoder with skip connections"""

    def __init__(self, in_channels=3, img_size=256):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256 + 256, 128)  # skip connection
        self.dec3 = DecoderBlock(128 + 128, 64)
        self.dec4 = DecoderBlock(64 + 64, in_channels, final_layer=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        d4 = self.dec4(torch.cat([d3, e1], dim=1))
        return d4


# -------------------------
# ResNet Backbone Autoencoders
# -------------------------

class BackboneVanillaAE(nn.Module):
    """Autoencoder using ResNet backbone (no skip)"""

    def __init__(self, backbone="resnet18", latent_dim=128):
        super().__init__()
        self.encoder = ResNetFeatureExtractor(backbone, layers=["layer1", "layer2", "layer3"])
        self.fc1 = nn.Linear(256 * 16 * 16, latent_dim)  # 예시 input: 256x16x16
        self.fc2 = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 3, final_layer=True)
        )

    def forward(self, x):
        feats = self.encoder(x)
        h = feats[-1].view(feats[-1].size(0), -1)
        z = self.fc1(h)
        h = self.fc2(z).view(feats[-1].size(0), 256, 16, 16)
        out = self.decoder(h)
        return out


class BackboneUNetAE(nn.Module):
    """UNet-style Autoencoder with ResNet backbone features"""

    def __init__(self, backbone="resnet18"):
        super().__init__()
        self.encoder = ResNetFeatureExtractor(backbone, layers=["layer1", "layer2", "layer3"])
        self.decoder1 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder3 = DecoderBlock(64, 3, final_layer=True)

    def forward(self, x):
        feats = self.encoder(x)
        h = feats[-1]
        d1 = self.decoder1(h)
        d2 = self.decoder2(d1)
        d3 = self.decoder3(d2)
        return d3


# -------------------------
# Loss & Anomaly Map
# -------------------------

class AutoencoderLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon, x):
        return F.mse_loss(recon, x)


def compute_anomaly_map(x: torch.Tensor, recon: torch.Tensor):
    """Compute reconstruction error map"""
    anomaly_map = torch.mean((x - recon) ** 2, dim=1, keepdim=True)  # [B,1,H,W]
    score = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)    # image-level score
    return anomaly_map, score
