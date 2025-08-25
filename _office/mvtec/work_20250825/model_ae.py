import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips


# =============================================================================
# Building Blocks
# =============================================================================

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


# =============================================================================
# Vanilla AutoEncoder
# =============================================================================

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

        # Safety check for encoder's downsampling factor (5 ConvBlocks with stride=2 -> /32)
        if self.img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {self.img_size}")

        self.start_size = self.img_size // 32  # Encoder downsampling factor (5 conv blocks)

        self.fc = nn.Linear(latent_dim, 512 * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (512, self.start_size, self.start_size))

        layers = [
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VanillaAE(nn.Module):
    """Vanilla autoencoder combining encoder and decoder"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.encoder = VanillaEncoder(in_channels, latent_dim)
        self.decoder = VanillaDecoder(out_channels, latent_dim, img_size)
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

    def compute_metrics(self, outputs):
        preds = outputs['reconstructed']
        targets = outputs['input']
        return {
            'psnr': float(psnr_metric(preds, targets)),
            'ssim': float(ssim_metric(preds, targets)),
        }

    def get_metrics(self):
        return ['psnr', 'ssim']

    def compute_anomaly_scores(self, outputs):
        preds = outputs["reconstructed"]
        targets = outputs["input"]
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])


# =============================================================================
# UNet-style Autoencoder with Skip Connections
# =============================================================================

class UnetEncoder(nn.Module):
    """UNet-style encoder with skip connections for feature preservation"""
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


class UnetDecoder(nn.Module):
    """UNet-style decoder with skip connections for detailed reconstruction"""
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
        self.final_activation = nn.Sigmoid()

    def forward(self, latent, skip_connections):
        x = self.fc(latent)
        x = self.unflatten(x)

        d1 = self.deconv1(x)                                           # 256, H/16
        d2 = self.deconv2(torch.cat([d1, skip_connections[3]], dim=1)) # 128, H/8
        d3 = self.deconv3(torch.cat([d2, skip_connections[2]], dim=1)) # 64,  H/4
        d4 = self.deconv4(torch.cat([d3, skip_connections[1]], dim=1)) # 32,  H/2
        d5 = self.deconv5(torch.cat([d4, skip_connections[0]], dim=1)) # C,   H/1
        reconstructed = self.final_activation(d5)
        return reconstructed


class UnetAE(nn.Module):
    """UNet-style autoencoder with skip connections for enhanced detail preservation"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, latent_dim)
        self.decoder = UnetDecoder(out_channels, latent_dim, img_size)
        self.model_type = "unet_ae"

    def forward(self, inputs):
        x = inputs["image"]
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return {
            "reconstructed": reconstructed,
            "latent": latent,
            "features": features,
            "input": x
        }

    def compute_loss(self, outputs):
        preds = outputs["reconstructed"]
        targets = outputs["input"]
        return combined_loss(preds, targets)

    def compute_metrics(self, outputs):
        preds = outputs["reconstructed"]
        targets = outputs["input"]
        return {
            "psnr": float(psnr_metric(preds, targets)),
            "ssim": float(ssim_metric(preds, targets)),
        }

    def get_metrics(self):
        return ["psnr", "ssim"]

    def compute_anomaly_scores(self, outputs):
        preds = outputs["reconstructed"]
        targets = outputs["input"]
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
