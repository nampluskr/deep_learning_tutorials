import os
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

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


# =============================================================================
# UNet-style Autoencoder with Skip Connections
# =============================================================================

class UNetEncoder(nn.Module):
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


class UNetDecoder(nn.Module):
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


class UNetAE(nn.Module):
    """UNet-style autoencoder with skip connections for enhanced detail preservation"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, latent_dim)
        self.decoder = UNetDecoder(out_channels, latent_dim, img_size)
        self.model_type = "unet_ae"

    def forward(self, x):
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return reconstructed, latent, features


# =============================================================================
# Losses & Metrics
# =============================================================================

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        return F.mse_loss(preds, targets, reduction=self.reduction)


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        return F.binary_cross_entropy(preds, targets, reduction=self.reduction)


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, ssim_weight=0.5, reduction='mean'):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.reduction = reduction

    def forward(self, preds, targets):
        mse = F.mse_loss(preds, targets, reduction=self.reduction)
        ssim_val = ssim(preds, targets, data_range=1.0, size_average=(self.reduction =='mean'))
        ssim_loss = 1 - ssim_val
        return self.mse_weight * mse + self.ssim_weight * ssim_loss


class PSNRMetric(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, preds, targets):
        mse = F.mse_loss(preds, targets, reduction='mean')
        if mse == 0:
            return torch.tensor(float('inf')).to(preds.device)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class SSIMMetric(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        return ssim(preds, targets, data_range=self.data_range, size_average=True)


# class LPIPSMetric(nn.Module):
#     def __init__(self, net='alex'):
#         super().__init__()
#         self.weights_urls = {
#             'alex': 'https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth',
#             'vgg': 'https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth',
#             'squeeze': 'https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/squeeze.pth'
#         }
#         self.weights_paths = {
#             'alex': "/home/namu/myspace/NAMU/backbones/lpips_alex.pth",
#             'vgg': "/home/namu/myspace/NAMU/backbones/lpips_vgg.pth",
#             'squeeze': "/home/namu/myspace/NAMU/backbones/lpips_squeeze.pth",
#         }
#         original_home = os.environ.get('TORCH_HOME', '')
#         os.environ['TORCH_HOME'] = self.weights_paths[net]
#         self.loss_fn = lpips.LPIPS(net=net, verbose=False)
#         state_dict = torch.load(self.weights_paths[net], map_location='cpu')
#         self.loss_fn.load_state_dict(state_dict, strict=False)
#         if original_home:
#             os.environ['TORCH_HOME'] = original_home

#     def forward(self, preds, targets):
#         return self.loss_fn(preds, targets)


if __name__ == "__main__":

    # loss_fn = get_loss("mse")
    # print(loss_fn)

    metric_fn = get_metric("psnr")
    print(metric_fn)

    # model = get_model("ae")
    # print(model)
