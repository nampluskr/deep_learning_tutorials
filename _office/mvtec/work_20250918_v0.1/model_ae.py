import os
import torch
import torch.nn as nn
import torchvision.models as models

BACKBONE_DIR = r"D:\Non_Documents\2025\1_project\1_image_processing\modeling\mvtec_office\backbones"
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


class AutoEncoder(nn.Module):
    def __init__(self, backbone=None, in_channels=3, out_channels=3,
                 img_shape=(256, 256), latent_dim=256):
        super().__init__()
        self.img_h, self.img_w = img_shape
        assert self.img_h % 32 == 0 and self.img_w % 32 == 0
        self.feat_h = self.img_h // 32
        self.feat_w = self.img_w // 32
        self.latent_dim = latent_dim
        self.feat_dim = 512

        if backbone is None or (isinstance(backbone, str) and backbone.lower() == "none"):
            self.encoder = nn.Sequential(
                ConvBlock(in_channels, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
                ConvBlock(512, 512),
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

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, self.latent_dim),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(self.latent_dim, self.feat_dim * self.feat_h * self.feat_w)
        self.decoder = nn.Sequential(
            DeconvBlock(self.feat_dim, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        latent = self.pooling(features)
        recon = self.linear(latent)
        recon = recon.view(-1, recon.shape[1] // (self.feat_h * self.feat_w), self.feat_h, self.feat_w)
        recon = self.decoder(recon)
        return recon, latent, features

    def compute_anomaly_map(self, recon, images):
        anomaly_map = torch.mean((images - recon) ** 2, dim=1, keepdim=True)
        max_val = anomaly_map.amax(dim=(-2, -1), keepdim=True)
        return anomaly_map / (max_val + 1e-8)

    def compute_anomaly_score(self, anomaly_map):
        return torch.amax(anomaly_map, dim=(-2, -1))

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        recon, *_ = self.forward(images)
        anomaly_map = self.compute_anomaly_map(recon, images)
        pred_score = self.compute_anomaly_score(anomaly_map)
        return {"anomaly_map": anomaly_map, "pred_score": pred_score}
