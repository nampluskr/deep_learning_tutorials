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

from dataloaders import denormalize_imagenet


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


# ============================================================================
# Model Architecture - ImageNet Compatible
# ============================================================================

class VanillaEncoder(nn.Module):
    def __init__(self, latent_dim=512):
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
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                features.append(out)
        latent = out
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
            # nn.Tanh()                       # [-1, 1] 로 정규화
        )

    def forward(self, latent):
        return self.layers(latent)


class VanillaAEV3(nn.Module):
    def __init__(self, latent_dim=512, img_size=256):
        super().__init__()
        self.img_size = img_size

        self.encoder = VanillaEncoder(latent_dim=latent_dim)
        self.decoder = VanillaDecoder(latent_dim=latent_dim)

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)

        if self.training:
            return reconstructed, latent, features
        else:
            return reconstructed


class UNetEncoder(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(3, base),               # 256 → 128
            ConvBlock(base, base*2),          # 128 → 64
            ConvBlock(base*2, base*4),        # 64  → 32
            ConvBlock(base*4, base*8),        # 32  → 16
            ConvBlock(base*8, base*8),        # 16  → 8   (deepest)
        ])

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)          # 다운샘플
            feats.append(x)     # 모든 단계 저장
        return feats            # [f0,f1,f2,f3,f4] (f4 deepest)


class UNetDecoder(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        # (deepest + skip) → upsample
        self.up1 = DeconvBlock(base*8 + base*8, base*8)   # 8 → 16
        self.up2 = DeconvBlock(base*8 + base*8, base*4)   # 16 → 32
        self.up3 = DeconvBlock(base*4 + base*4, base*2)   # 32 → 64
        self.up4 = DeconvBlock(base*2 + base*2, base)     # 64 → 128
        self.up5 = DeconvBlock(base + base, base)         # 128 → 256

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, feats):
        # 가장 깊은 feature 로 시작
        x = feats[-1]                     # (B,512,8,8)

        # 1) deepest + f3
        x = torch.cat([x, feats[-2]], dim=1)   # (B,1024,8,8)
        x = self.up1(x)                        # (B,512,16,16)

        # 2) + f2
        x = torch.cat([x, feats[-3]], dim=1)   # (B,1024,16,16)
        x = self.up2(x)                        # (B,256,32,32)

        # 3) + f1
        x = torch.cat([x, feats[-4]], dim=1)   # (B,512,32,32)
        x = self.up3(x)                        # (B,128,64,64)

        # 4) + f0
        x = torch.cat([x, feats[-5]], dim=1)   # (B,256,64,64)
        x = self.up4(x)                        # (B,64,128,128)

        # 5) 마지막 up‑sample (가장 얕은 skip을 다시 concat)
        x = torch.cat([x, feats[0]], dim=1)    # (B,128,128,128)
        x = self.up5(x)                        # (B,64,256,256)

        recon = self.final(x)                  # (B,3,256,256)
        return recon


class UNetAE(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.encoder = UNetEncoder(base)
        self.decoder = UNetDecoder(base)

    def forward(self, x):
        skips = self.encoder(x)          # list of 5 feature maps
        recon = self.decoder(skips)      # 복원 이미지
        if self.training:
            return recon, skips
        return recon


class VanillaAEV2(nn.Module):

    def __init__(self, latent_dim=512, img_size=256):
        super().__init__()
        self.img_size = img_size

        # Encoder - More gradual compression
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),               # 256 -> 128
            ConvBlock(64, 128),             # 128 -> 64
            ConvBlock(128, 256),            # 64 -> 32
            ConvBlock(256, 512),            # 32 -> 16
            ConvBlock(512, latent_dim),     # 16 -> 8
        )

        # Decoder - Mirror of encoder
        self.decoder = nn.Sequential(
            DeconvBlock(latent_dim, 512),   # 8 -> 16
            DeconvBlock(512, 256),          # 16 -> 32
            DeconvBlock(256, 128),          # 32 -> 64
            DeconvBlock(128, 64),           # 64 -> 128
            # 128 -> 256
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        if self.training:
            return reconstructed, latent
        else:
            return reconstructed


class VanillaAEV1(nn.Module):

    def __init__(self, latent_dim=512, img_size=256):
        super().__init__()

        self.img_size = img_size

        # Encoder - More gradual compression
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(512, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder - Mirror of encoder
        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 16 -> 32
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 128 -> 256
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        if self.training:
            return reconstructed, latent
        else:
            return reconstructed


# ============================================================================
# ResNet Backbone: AE / VAE
# ============================================================================

class ResNetEncoderV1(nn.Module):
    """Fixed ResNet backbone encoder"""

    def __init__(self, backbone='resnet18', pretrained_path=None):
        super().__init__()

        self.backbone_name = backbone

        # Create ResNet backbone
        import torchvision.models as models

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            self.feature_dims = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=False)
            self.feature_dims = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained weights: {e}")

        # Extract layers for feature extraction
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool

        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        # Global pooling for final features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Extract hierarchical features"""

        # Initial conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Extract features from each layer
        feat1 = self.layer1(x)    # 64x64
        feat2 = self.layer2(feat1)  # 32x32
        feat3 = self.layer3(feat2)  # 16x16
        feat4 = self.layer4(feat3)  # 8x8

        # Global pooled features for latent projection
        pooled_feat = self.global_pool(feat4)

        # Return skip features + final pooled features
        skip_features = [feat1, feat2, feat3, feat4]

        return skip_features, pooled_feat


class ResNetDecoderV1(nn.Module):
    """Fixed ResNet Decoder with proper dimension handling"""

    def __init__(self, backbone='resnet18', latent_dim=512, img_size=256, use_skip_connections=True):
        super().__init__()

        self.img_size = img_size
        self.use_skip_connections = use_skip_connections
        self.backbone = backbone

        # Define channel dimensions based on backbone
        if backbone in ['resnet18', 'resnet34']:
            self.backbone_final_dim = 512
            # Skip connection channels: [64, 128, 256, 512] -> reversed for decoder
            self.skip_dims = [512, 256, 128, 64] if use_skip_connections else [0, 0, 0, 0]
        elif backbone == 'resnet50':
            self.backbone_final_dim = 2048
            # Skip connection channels: [256, 512, 1024, 2048] -> reversed for decoder
            self.skip_dims = [2048, 1024, 512, 256] if use_skip_connections else [0, 0, 0, 0]

        # Calculate proper spatial dimensions
        # ResNet reduces by 32x: 256 -> 8
        self.start_size = img_size // 32  # 8x8

        # Latent to spatial conversion
        self.latent_to_spatial = nn.Sequential(
            nn.Linear(latent_dim, self.backbone_final_dim * self.start_size * self.start_size),
            nn.ReLU(inplace=True)
        )

        # Decoder blocks - Progressive upsampling
        self.decoder_blocks = nn.ModuleList()

        # Block 1: 8x8 -> 16x16
        in_ch = self.backbone_final_dim + self.skip_dims[0]
        out_ch = 256
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))

        # Block 2: 16x16 -> 32x32
        in_ch = out_ch + self.skip_dims[1]
        out_ch = 128
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))

        # Block 3: 32x32 -> 64x64
        in_ch = out_ch + self.skip_dims[2]
        out_ch = 64
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))

        # Block 4: 64x64 -> 128x128
        in_ch = out_ch + self.skip_dims[3]
        out_ch = 32
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))

        # Final block: 128x128 -> 256x256
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, latent, skip_features=None):
        """Decode with proper dimension handling"""

        batch_size = latent.size(0)

        # Project latent to spatial features
        spatial_feat = self.latent_to_spatial(latent)
        x = spatial_feat.view(batch_size, self.backbone_final_dim, self.start_size, self.start_size)

        # Progressive upsampling with skip connections
        if skip_features is not None and self.use_skip_connections:
            # Reverse skip features for decoder (deepest first)
            skip_features = skip_features[::-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            # Add skip connection if available
            if (skip_features is not None and
                self.use_skip_connections and
                i < len(skip_features)):

                skip_feat = skip_features[i]

                # Ensure spatial dimensions match
                if skip_feat.shape[-2:] != x.shape[-2:]:
                    skip_feat = F.interpolate(
                        skip_feat, size=x.shape[-2:],
                        mode='bilinear', align_corners=False
                    )

                # Concatenate skip connection
                x = torch.cat([x, skip_feat], dim=1)

            # Apply decoder block
            x = decoder_block(x)

        # Final output
        output = self.final_block(x)
        return output


class ResNetAEV1(nn.Module):
    """Fixed ResNet-based Autoencoder"""

    def __init__(self, backbone='resnet18', pretrained_path=None, latent_dim=512,
                 img_size=256, use_skip_connections=True):
        super().__init__()

        self.backbone = backbone
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections

        # Encoder
        self.encoder = ResNetEncoderV2(backbone=backbone, pretrained_path=pretrained_path)

        # Latent projection
        if backbone in ['resnet18', 'resnet34']:
            backbone_dim = 512
        elif backbone == 'resnet50':
            backbone_dim = 2048

        self.latent_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = ResNetDecoderV1(
            backbone=backbone,
            latent_dim=latent_dim,
            img_size=img_size,
            use_skip_connections=use_skip_connections
        )

    def encode(self, x):
        """Encode to latent space"""
        skip_features, pooled_feat = self.encoder(x)
        latent = self.latent_projection(pooled_feat)
        return latent, skip_features

    def decode(self, latent, skip_features=None):
        """Decode from latent space"""
        return self.decoder(latent, skip_features)

    def forward(self, x):
        latent, skip_features = self.encode(x)

        # Use skip connections if enabled
        skip_for_decoder = skip_features if self.use_skip_connections else None
        reconstructed = self.decode(latent, skip_for_decoder)

        if self.training:
            return reconstructed, latent, skip_features
        else:
            return reconstructed



if __name__ == "__main__":

    """ Usages:
    resnet18_ae = ResNetAE(
        backbone='resnet18',
        pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet18-f37072fd.pth',
        latent_dim=512,
        img_size=256,
        use_skip_connections=True)

    resnet34_ae = ResNetAE(
        backbone='resnet18',
        pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet34-b627a593.pth',
        latent_dim=512,
        img_size=256,
        use_skip_connections=True)

    resnet50_ae = ResNetAE(
        backbone='resnet50',
        pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet50-0676ba61.pth',
        latent_dim=512,
        img_size=256,
        use_skip_connections=True)

    valinilla_ae = VanillaAE(latent_dim=512, img_size=256)
    """
    pass
