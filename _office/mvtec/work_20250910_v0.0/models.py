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

from dataloader import IMAGENET_MEAN, IMAGENET_STD, denormalize_imagenet


# ============================================================================
# Model Architecture - ImageNet Compatible
# ============================================================================

class VanillaAE(nn.Module):
    """Autoencoder compatible with ImageNet normalization and backbone integration"""

    def __init__(self, latent_dim=512, input_size=256):
        super().__init__()

        self.input_size = input_size

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
            nn.Tanh(),  # Output in [-1, 1] range, will be converted to ImageNet range
        )

    def forward(self, x):
        latent = self.encoder(x)
        raw_recon = self.decoder(latent)

        # Convert Tanh output [-1, 1] to ImageNet normalized range
        # This ensures compatibility with ImageNet normalized input
        recon = self._tanh_to_imagenet_range(raw_recon)

        return recon

    def _tanh_to_imagenet_range(self, tanh_output):
        """Convert tanh output [-1, 1] to ImageNet normalized range"""
        # First convert [-1, 1] to [0, 1]
        normalized_01 = (tanh_output + 1.0) / 2.0

        # Then convert [0, 1] to ImageNet range
        device = tanh_output.device
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        imagenet_range = (normalized_01 - mean) / std
        return imagenet_range

class VanillaVAE(nn.Module):
    """Variational Autoencoder compatible with ImageNet normalization"""

    def __init__(self, latent_dim=512, input_size=256, beta=0.01):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # Shared encoder layers
        self.encoder_layers = nn.Sequential(
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
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Global average pooling to reduce spatial dimensions
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Latent space projections
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Projection back to spatial features for decoder
        self.fc_decode = nn.Linear(latent_dim, 1024 * 8 * 8)

        # Decoder
        self.decoder = nn.Sequential(
            # Reshape to spatial
            nn.Unflatten(1, (1024, 8, 8)),

            # 8 -> 16
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
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

    def encode(self, x):
        """Encode input to latent parameters"""
        features = self.encoder_layers(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """Decode latent variable to reconstruction"""
        spatial_features = self.fc_decode(z)
        raw_recon = self.decoder(spatial_features)

        # Convert Tanh output to ImageNet range
        recon = self._tanh_to_imagenet_range(raw_recon)
        return recon

    def _tanh_to_imagenet_range(self, tanh_output):
        """Convert tanh output [-1, 1] to ImageNet normalized range"""
        # First convert [-1, 1] to [0, 1]
        normalized_01 = (tanh_output + 1.0) / 2.0

        # Then convert [0, 1] to ImageNet range
        device = tanh_output.device
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        imagenet_range = (normalized_01 - mean) / std
        return imagenet_range

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        if self.training:
            return recon, mu, logvar
        else:
            return recon

class VAELoss(nn.Module):
    """VAE Loss with reconstruction and KL divergence terms for ImageNet normalized data"""

    def __init__(self, beta=0.01, capacity=0.0, gamma=1000.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.capacity = capacity
        self.gamma = gamma
        self.reduction = reduction
        self.step_count = 0
        self.warm_up_steps = 10000

    def forward(self, recon, target, mu, logvar):
        # Reconstruction loss (MSE in ImageNet normalized space)
        recon_loss = F.mse_loss(recon, target, reduction=self.reduction)

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / target.size(0)

        # Beta warm-up schedule
        self.step_count += 1
        if self.step_count < self.warm_up_steps:
            current_beta = self.beta * (self.step_count / self.warm_up_steps)
        else:
            current_beta = self.beta

        # Capacity regularization
        if self.capacity > 0:
            kld_loss = self.gamma * torch.abs(kld_loss - self.capacity)

        # Total loss
        total_loss = recon_loss + current_beta * kld_loss

        return total_loss, recon_loss, kld_loss



# ============================================================================
# ResNet Backbone: AE / VAE
# ============================================================================

class ResNetEncoder(nn.Module):
    """Fixed ResNet backbone encoder"""

    def __init__(self, backbone='resnet18', pretrained_path=None):
        super().__init__()

        self.backbone_name = backbone

        # Create ResNet backbone
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


class ResNetDecoder(nn.Module):
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


class ResNetAE(nn.Module):
    """Fixed ResNet-based Autoencoder"""

    def __init__(self, backbone='resnet18', pretrained_path=None, latent_dim=512,
                 img_size=256, use_skip_connections=True):
        super().__init__()

        self.backbone = backbone
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections

        # Encoder
        self.encoder = ResNetEncoder(backbone=backbone, pretrained_path=pretrained_path)

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
        self.decoder = ResNetDecoder(
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
        """Complete forward pass"""
        latent, skip_features = self.encode(x)

        # Use skip connections if enabled
        skip_for_decoder = skip_features if self.use_skip_connections else None
        recon = self.decode(latent, skip_for_decoder)

        # Convert to ImageNet range
        recon_imagenet = self._tanh_to_imagenet_range(recon)

        if self.training:
            return recon_imagenet, latent, skip_features
        else:
            return recon_imagenet

    def _tanh_to_imagenet_range(self, tanh_output):
        """Convert tanh output to ImageNet range"""
        normalized_01 = (tanh_output + 1.0) / 2.0

        device = tanh_output.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        return (normalized_01 - mean) / std


class ResNetVAE(nn.Module):
    """Fixed ResNet-based Variational Autoencoder"""

    def __init__(self, backbone='resnet18', pretrained_path=None, latent_dim=512,
                 img_size=256, use_skip_connections=True, beta=0.01):
        super().__init__()

        self.backbone = backbone
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        self.beta = beta

        # Encoder
        self.encoder = ResNetEncoder(backbone=backbone, pretrained_path=pretrained_path)

        # VAE latent projections
        if backbone in ['resnet18', 'resnet34']:
            backbone_dim = 512
        elif backbone == 'resnet50':
            backbone_dim = 2048

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(backbone_dim, latent_dim)
        self.fc_logvar = nn.Linear(backbone_dim, latent_dim)

        # Decoder
        self.decoder = ResNetDecoder(
            backbone=backbone,
            latent_dim=latent_dim,
            img_size=img_size,
            use_skip_connections=use_skip_connections
        )

    def encode(self, x):
        """Encode to latent parameters"""
        skip_features, pooled_feat = self.encoder(x)
        flattened = self.flatten(pooled_feat)

        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)

        return mu, logvar, skip_features

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, skip_features=None):
        """Decode from latent space"""
        return self.decoder(z, skip_features)

    def forward(self, x):
        """Complete forward pass"""
        mu, logvar, skip_features = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Use skip connections if enabled
        skip_for_decoder = skip_features if self.use_skip_connections else None
        recon = self.decode(z, skip_for_decoder)

        # Convert to ImageNet range
        recon_imagenet = self._tanh_to_imagenet_range(recon)

        if self.training:
            return recon_imagenet, mu, logvar, skip_features
        else:
            return recon_imagenet

    def _tanh_to_imagenet_range(self, tanh_output):
        """Convert tanh output to ImageNet range"""
        normalized_01 = (tanh_output + 1.0) / 2.0

        device = tanh_output.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        return (normalized_01 - mean) / std


def create_resnet_models(device):
    """Create fixed ResNet models"""

    models = {}

    # ResNet18 models
    try:
        resnet18_ae = ResNetAE(
            backbone='resnet18',
            pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet18-f37072fd.pth',
            latent_dim=512,
            img_size=256,
            use_skip_connections=True
        ).to(device)
        models['ResNet18_AE'] = resnet18_ae

        resnet18_vae = ResNetVAE(
            backbone='resnet18',
            pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet18-f37072fd.pth',
            latent_dim=512,
            img_size=256,
            use_skip_connections=True,
            beta=0.01
        ).to(device)
        models['ResNet18_VAE'] = resnet18_vae

    except Exception as e:
        print(f"⚠️ Failed to create ResNet18 models: {e}")

    # ResNet50 models
    try:
        resnet50_ae = ResNetAE(
            backbone='resnet50',
            pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet50-0676ba61.pth',
            latent_dim=512,
            img_size=256,
            use_skip_connections=True
        ).to(device)
        models['ResNet50_AE'] = resnet50_ae

        resnet50_vae = ResNetVAE(
            backbone='resnet50',
            pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet50-0676ba61.pth',
            latent_dim=512,
            img_size=256,
            use_skip_connections=True,
            beta=0.01
        ).to(device)
        models['ResNet50_VAE'] = resnet50_vae

    except Exception as e:
        print(f"⚠️ Failed to create ResNet50 models: {e}")

    return models
