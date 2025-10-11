import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# BACKBONE_DIR = r"D:\Non_Documents\2025\1_project\1_image_processing\modeling\mvtec_office\backbones"
BACKBONE_DIR = '/mnt/d/backbones'
BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
}


def get_backbone_path(backbone):
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


###########################################################
# Baseline AutoEncoder Model
###########################################################

class Baseline(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 32 * 32, out_features=latent_dim)            
        )
        self.from_linear = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256 * 32 * 32),
            nn.Unflatten(dim=1, unflattened_size=(256, 32, 32)),
        )

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.to_linear(latent)
        latent = self.from_linear(latent)
        recon = self.decoder(latent)
        return recon, latent

    def compute_anomaly_map(self, recon, images):
        anomaly_map = torch.mean((images - recon)**2, dim=1, keepdim=True)
        return anomaly_map

    def compute_anomaly_score(self, anomaly_map):
        # img_score = anomaly_maps.view(anomaly_map.size(0), -1).mean(dim=1)
        # pred_score = img_score.detach().cpu().numpy().tolist()
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        recon, *_ = self.forward(images)
        anomaly_map = self.compute_anomaly_map(recon, images)
        pred_score = self.compute_anomaly_score(anomaly_map)
        return {"anomaly_map": anomaly_map, "pred_score": pred_score}


###########################################################
# AutoEncoder Model with various backbones
###########################################################

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


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ssim import ssim


BACKBONE_DIR = '/mnt/d/backbones'
BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
}


def get_backbone_path(backbone):
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, backbone=None, in_channels=3, out_channels=3,
                 img_shape=(256, 256), latent_dim=256):
        super().__init__()
        self.img_h, self.img_w = img_shape
        assert self.img_h % 16 == 0 and self.img_w % 16 == 0, \
            "Image size must be divisible by 16."

        self.latent_dim = latent_dim
        self.backbone = backbone
        self.feat_dim = 512
        self.down_factor = 32 if backbone in ["resnet18", "resnet34", "resnet50"] else 16

        self.feat_h = self.img_h // self.down_factor
        self.feat_w = self.img_w // self.down_factor

        # ---------------- Encoder ---------------- #
        if backbone is None or (isinstance(backbone, str) and backbone.lower() == "none"):
            self.encoder = nn.Sequential(
                ConvBlock(in_channels, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
            )
        else:
            if backbone == "resnet18":
                encoder = models.resnet18(weights=None)
            elif backbone == "resnet34":
                encoder = models.resnet34(weights=None)
            elif backbone == "resnet50":
                encoder = models.resnet50(weights=None)
                self.feat_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone {backbone}")

            weight_path = get_backbone_path(backbone)
            if os.path.isfile(weight_path):
                state = torch.load(weight_path, map_location="cpu", weights_only=True)
                encoder.load_state_dict(state)

            if in_channels != 3:
                encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,
                encoder.layer2,
                encoder.layer3,
                encoder.layer4,
            )

        # ---------------- Latent ---------------- #
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim * self.feat_h * self.feat_w, self.latent_dim),
            nn.ReLU(inplace=True),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim * self.feat_h * self.feat_w),
            nn.ReLU(inplace=True),
        )

        # ---------------- Decoder ---------------- #
        deconv_layers = []
        in_c = self.feat_dim
        num_upsamples = int(torch.log2(torch.tensor(self.down_factor)).item())

        channels = [256, 128, 64, 32]
        for i in range(num_upsamples - 1):
            out_c = channels[i] if i < len(channels) else max(32 // (2 ** (i - len(channels) + 1)), out_channels)
            deconv_layers.append(DeconvBlock(in_c, out_c))
            in_c = out_c

        deconv_layers.append(
            nn.ConvTranspose2d(in_c, out_channels, kernel_size=4, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(*deconv_layers)

    def forward(self, x):
        features = self.encoder(x)
        latent = self.to_latent(features)
        recon = self.from_latent(latent)
        recon = recon.view(-1, self.feat_dim, self.feat_h, self.feat_w)
        recon = self.decoder(recon)
        return recon, latent, features

    def compute_anomaly_map(self, recon, images):
        return torch.mean((images - recon) ** 2, dim=1, keepdim=True)

    def compute_anomaly_score(self, anomaly_map):
        return torch.amax(anomaly_map, dim=(-2, -1))

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        recon, *_ = self.forward(images)
        anomaly_map = self.compute_anomaly_map(recon, images)
        pred_score = self.compute_anomaly_score(anomaly_map)
        return {"anomaly_map": anomaly_map, "pred_score": pred_score}


############################################################
# Loss Functions and Metrics
############################################################

class AELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon, original):
        return F.mse_loss(recon, original, reduction=self.reduction)


class AECombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, ssim_weight=0.3, reduction='mean'):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.reduction = reduction

    def forward(self, recon, original):
        mse_loss = F.mse_loss(recon, original, reduction=self.reduction)
        ssim_loss = 1 - ssim(recon, original, data_range=2.0, size_average=(self.reduction == 'mean'))
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


from ssim import ssim

class SSIMMetric(nn.Module):
    def __init__(self, data_range=2.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        return ssim(preds, targets, data_range=self.data_range, size_average=True)