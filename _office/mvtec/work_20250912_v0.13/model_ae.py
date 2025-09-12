import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ===================================================================
# Convolutional Blocks
# ===================================================================

class ConvBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv_block(x)


# ===================================================================
# Vanilla Aueoencoder
# ===================================================================

class VanillaEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock(3, 64),               # 256 -> 128
            ConvBlock(64, 128),             # 128 -> 64
            ConvBlock(128, 256),            # 64  -> 32
            ConvBlock(256, 512),            # 32  -> 16
            ConvBlock(512, latent_dim),     # 16  -> 8   (latent)
        ])

    def forward(self, x):
        features = []   # [f0, f1, f2, f3, f4]   # f0: 64‑ch, f4: deepest (512‑ch)
        outputs = x
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if i < len(self.layers) - 1:
                features.append(outputs)
        latent = outputs
        return latent, features


class VanillaDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            DeconvBlock(latent_dim, 512),   # 8  -> 16
            DeconvBlock(512, 256),          # 16 -> 32
            DeconvBlock(256, 128),          # 32 -> 64
            DeconvBlock(128, 64),           # 64 -> 128
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
        )

    def forward(self, latent):
        return self.layers(latent)


class VanillaAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = VanillaEncoder(latent_dim=latent_dim)
        self.decoder = VanillaDecoder(latent_dim=latent_dim)

    def forward(self, images):
        latent, features = self.encoder(images)
        reconstructed = self.decoder(latent)

        if self.training:
            return reconstructed, latent, features
        else:
            maps = torch.mean((images - reconstructed)**2, dim=1)
            scores = torch.amax(maps.view(maps.size(0), -1), dim=1) # max_pooling
            return dict(anomaly_amp=maps, pred_score=scores)
