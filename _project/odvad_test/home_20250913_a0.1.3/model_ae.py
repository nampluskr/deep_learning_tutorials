import torch
import torch.nn as nn
import torch.nn.functional as F
from ssim import ssim

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


# =============================================================================
# Vanilla AutoEncoder
# =============================================================================

class VanillaEncoder(nn.Module):
    """Vanilla encoder with consistent latent dimension."""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock(in_channels, 64),     # 256 -> 128
            ConvBlock(64, 128),             # 128 -> 64
            ConvBlock(128, 256),            # 64  -> 32
            ConvBlock(256, 512),            # 32  -> 16
            ConvBlock(512, latent_dim),     # 16  -> 8   (latent)
        ])

    def forward(self, x):
        features = []   # [f0, f1, f2, f3, f4]   # f0: 64ch, f4: deepest (latent_dim ch)
        outputs = x
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if i < len(self.layers) - 1:
                features.append(outputs)
        latent = outputs
        return latent, features


class VanillaDecoder(nn.Module):
    """Vanilla decoder with consistent latent dimension."""
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            DeconvBlock(latent_dim, 512),   # 8  -> 16
            DeconvBlock(512, 256),          # 16 -> 32
            DeconvBlock(256, 128),          # 32 -> 64
            DeconvBlock(128, 64),           # 64 -> 128
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 128 -> 256
        )

    def forward(self, latent):
        return self.layers(latent)


class VanillaAE(nn.Module):
    """Vanilla AutoEncoder with unified latent dimension."""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = VanillaEncoder(in_channels, latent_dim)
        self.decoder = VanillaDecoder(out_channels, latent_dim)
        self.model_type = "vanilla_ae"

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features
        
    def evaluate(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)

        anomaly_map = self.compute_anomaly_map(x, reconstructed)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_anomaly_map(self, original, reconstructed):
        anomaly_map = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        return anomaly_map


# =============================================================================
# UNet AutoEncoder
# =============================================================================

class UNetEncoder(nn.Module):
    """UNet encoder with skip connections."""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32)    # /2
        self.conv2 = ConvBlock(32, 64)             # /4
        self.conv3 = ConvBlock(64, 128)            # /8
        self.conv4 = ConvBlock(128, 256)           # /16
        self.conv5 = ConvBlock(256, 512)           # /32

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        # Forward with skip connections
        e1 = self.conv1(x)      # 32, H/2,  W/2
        e2 = self.conv2(e1)     # 64, H/4,  W/4
        e3 = self.conv3(e2)     # 128,H/8,  W/8
        e4 = self.conv4(e3)     # 256,H/16, W/16
        e5 = self.conv5(e4)     # 512,H/32, W/32

        pooled = self.pool(e5).view(x.size(0), -1)
        latent = self.fc(pooled)

        skip_connections = [e1, e2, e3, e4]
        return latent, e5, skip_connections


class UNetDecoder(nn.Module):
    """UNet decoder with skip connections."""
    def __init__(self, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.img_size = img_size
        if self.img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {self.img_size}")
        self.start_size = self.img_size // 32  # matches encoder downsampling

        self.fc = nn.Linear(latent_dim, 512 * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (512, self.start_size, self.start_size))

        # Decoder: upsample then concat with corresponding encoder feature
        self.deconv1 = DeconvBlock(512, 256)                           # /16
        self.deconv2 = DeconvBlock(256 + 256, 128)                     # /8
        self.deconv3 = DeconvBlock(128 + 128, 64)                      # /4
        self.deconv4 = DeconvBlock(64 + 64, 32)                        # /2
        self.deconv5 = nn.ConvTranspose2d(32 + 32, out_channels, 4, 2, 1)  # /1

    def forward(self, latent, skip_connections):
        x = self.fc(latent)
        x = self.unflatten(x)

        d1 = self.deconv1(x)                                           # 256, H/16
        d2 = self.deconv2(torch.cat([d1, skip_connections[3]], dim=1)) # 128, H/8
        d3 = self.deconv3(torch.cat([d2, skip_connections[2]], dim=1)) # 64,  H/4
        d4 = self.deconv4(torch.cat([d3, skip_connections[1]], dim=1)) # 32,  H/2
        d5 = self.deconv5(torch.cat([d4, skip_connections[0]], dim=1)) # C,   H/1
        reconstructed = d5
        return reconstructed


class UNetAE(nn.Module):
    """UNet AutoEncoder with skip connections."""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, latent_dim)
        self.decoder = UNetDecoder(out_channels, latent_dim, img_size)
        self.model_type = "unet_ae"

    def forward(self, x):
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return reconstructed, latent, features
        
    def evaluate(self, x):
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)

        anomaly_map = self.compute_anomaly_map(x, reconstructed)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_anomaly_map(self, original, reconstructed):
        anomaly_map = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        return anomaly_map


# =============================================================================
# Loss Functions
# =============================================================================

class AELoss(nn.Module):
    """Basic MSE loss for AutoEncoder."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, reconstructed, original):
        return F.mse_loss(reconstructed, original, reduction=self.reduction)


class AECombinedLoss(nn.Module):
    """Combined MSE and SSIM loss for AutoEncoder."""
    def __init__(self, mse_weight=0.7, ssim_weight=0.3, reduction='mean'):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.reduction = reduction

    def forward(self, reconstructed, original):
        # MSE loss
        mse_loss = F.mse_loss(reconstructed, original, reduction=self.reduction)

        # SSIM loss (requires ssim.py from pytorch_msssim library)
        ssim_val = ssim(reconstructed, original, data_range=1.0, size_average=(self.reduction == 'mean'))
        ssim_loss = 1 - ssim_val
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


# =============================================================================
# Utility Functions
# =============================================================================

def compute_ae_anomaly_map(original, reconstructed):
    """Compute anomaly map for AutoEncoder."""
    anomaly_map = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
    return anomaly_map


def compute_ae_anomaly_scores(original, reconstructed):
    """Compute anomaly scores for AutoEncoder."""
    scores = torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])
    return scores


# =============================================================================
# Metrics
# =============================================================================

class SSIMMetric(nn.Module):
    """Structural Similarity Index Measure metric."""
    
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        ssim_value = ssim(preds, targets, data_range=self.data_range, size_average=True)
        return ssim_value.item()


if __name__ == "__main__":
    pass