import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

from model_ae import combined_loss, psnr_metric, ssim_metric


# ===============================================================
# Small building blocks
# ===============================================================

class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU block"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + concat(skip) + fuse"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        if skip_ch > 0:
            self.fuse = ConvBNReLU(out_ch + skip_ch, out_ch, 3, 1, 1)
        else:
            self.fuse = ConvBNReLU(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


# ===============================================================
# Encoder with selectable backbone
# ===============================================================

class BackboneFeatureEncoder(nn.Module):
    """Feature pyramid encoder returning skips + top feature"""
    def __init__(self, backbone='resnet34', in_channels=3, weights_dir=None):
        super().__init__()
        self.backbone = backbone.lower()
        self.in_channels = in_channels

        # 기본은 weights=None → 무작위 초기화
        if self.backbone == 'resnet34':
            net = tv.resnet34(weights=None)
            ch = [64, 64, 128, 256, 512]
        elif self.backbone == 'resnet50':
            net = tv.resnet50(weights=None)
            ch = [64, 256, 512, 1024, 2048]
        elif self.backbone == 'vgg16':
            net = tv.vgg16_bn(weights=None)
        elif self.backbone == 'vgg19':
            net = tv.vgg19_bn(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # in_channels 수정
        if hasattr(net, "conv1") and self.in_channels != 3:
            net.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Pretrained weights load (optional)
        if weights_dir is not None:
            weights_path = os.path.join(weights_dir, f"{self.backbone}.pth")
            if os.path.isfile(weights_path):
                try:
                    state_dict = torch.load(weights_path, map_location="cpu")
                    net.load_state_dict(state_dict)
                    print(f"[INFO] Loaded pretrained weights from {weights_path}")
                except Exception as e:
                    print(f"[WARN] Could not load weights {weights_path}: {e}")
            else:
                print(f"[WARN] No pretrained weights found at {weights_path}, using random init.")

        # Encoder parts
        if self.backbone.startswith('resnet'):
            self.enc0 = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2
            self.pool = net.maxpool                                  # /4
            self.layer1 = net.layer1                                 # /4
            self.layer2 = net.layer2                                 # /8
            self.layer3 = net.layer3                                 # /16
            self.layer4 = net.layer4                                 # /32

            if self.backbone == 'resnet34':
                self.skip_channels = (64, 64, 128, 256)
                self.top_channels = 512
            else:
                self.skip_channels = (64, 256, 512, 1024)
                self.top_channels = 2048

        else:  # vgg
            feats = list(net.features.children())
            blocks, acc = [], []
            for m in feats:
                acc.append(m)
                if isinstance(m, nn.MaxPool2d):
                    blocks.append(nn.Sequential(*acc))
                    acc = []
            self.block1, self.block2, self.block3, self.block4, self.block5 = blocks[:5]

            self.skip_channels = (64, 128, 256, 512)
            self.top_channels = 512

    def forward(self, x):
        if self.backbone.startswith('resnet'):
            e1 = self.enc0(x)    # /2
            e2 = self.pool(e1)   # /4
            e3 = self.layer1(e2) # /4
            e4 = self.layer2(e3) # /8
            e5 = self.layer3(e4) # /16
            e6 = self.layer4(e5) # /32
            return e1, e3, e4, e5, e6
        else:
            b1 = self.block1(x)  # /2
            b2 = self.block2(b1) # /4
            b3 = self.block3(b2) # /8
            b4 = self.block4(b3) # /16
            b5 = self.block5(b4) # /32
            return b1, b2, b3, b4, b5


# ===============================================================
# Decoders
# ===============================================================

class BackboneVanillaDecoder(nn.Module):
    """Vanilla decoder: no skip connections, just upsampling"""
    def __init__(self, out_channels=3, latent_dim=512, img_size=256,
                 top_channels=512, top_proj=512):
        super().__init__()
        if img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        self.start_size = img_size // 32

        self.fc = nn.Linear(latent_dim, top_proj * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (top_proj, self.start_size, self.start_size))

        self.up1 = UpBlock(in_ch=top_proj, skip_ch=0, out_ch=256)
        self.up2 = UpBlock(in_ch=256, skip_ch=0, out_ch=128)
        self.up3 = UpBlock(in_ch=128, skip_ch=0, out_ch=64)
        self.up4 = UpBlock(in_ch=64, skip_ch=0, out_ch=32)
        self.final = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return self.act(x)


class BackboneUNetDecoder(nn.Module):
    """UNet-style decoder with skip connections"""
    def __init__(self, out_channels=3, latent_dim=512, img_size=256,
                 skip_channels=(64, 128, 256, 512), top_channels=512,
                 target_skip=(32, 64, 128, 256), top_proj=512):
        super().__init__()
        if img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        self.start_size = img_size // 32

        self.fc = nn.Linear(latent_dim, top_proj * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (top_proj, self.start_size, self.start_size))

        # 1x1 conv projections
        self.proj_s1 = nn.Conv2d(skip_channels[0], target_skip[0], 1, bias=False)
        self.proj_s2 = nn.Conv2d(skip_channels[1], target_skip[1], 1, bias=False)
        self.proj_s3 = nn.Conv2d(skip_channels[2], target_skip[2], 1, bias=False)
        self.proj_s4 = nn.Conv2d(skip_channels[3], target_skip[3], 1, bias=False)

        self.up1 = UpBlock(in_ch=top_proj, skip_ch=target_skip[3], out_ch=256)
        self.up2 = UpBlock(in_ch=256, skip_ch=target_skip[2], out_ch=128)
        self.up3 = UpBlock(in_ch=128, skip_ch=target_skip[1], out_ch=64)
        self.up4 = UpBlock(in_ch=64, skip_ch=target_skip[0], out_ch=32)
        self.final = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, latent, skips):
        x = self.fc(latent)
        x = self.unflatten(x)
        s1, s2, s3, s4 = [self.proj_s1(skips[0]), self.proj_s2(skips[1]),
                          self.proj_s3(skips[2]), self.proj_s4(skips[3])]
        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        x = self.final(x)
        return self.act(x)


# ===============================================================
# Full Models
# ===============================================================

class BackboneVanillaAE(nn.Module):
    """Backbone-based Vanilla Autoencoder (no skip connections)"""
    def __init__(self, backbone='resnet34', in_channels=3, out_channels=3,
                 latent_dim=512, img_size=256, weights_dir=None):
        super().__init__()
        self.model_type = "backbone_vanilla_ae"
        self.backbone_name = backbone

        self.encoder = BackboneFeatureEncoder(backbone=backbone,
                                              in_channels=in_channels,
                                              weights_dir=weights_dir)
        top_ch = self.encoder.top_channels

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Linear(top_ch, latent_dim)

        self.decoder = BackboneVanillaDecoder(out_channels=out_channels,
                                              latent_dim=latent_dim,
                                              img_size=img_size,
                                              top_channels=top_ch)

    def forward(self, inputs):
        x = inputs["image"]
        *_, top = self.encoder(x)   # only top feature used
        pooled = self.gap(top).view(x.size(0), -1)
        latent = self.to_latent(pooled)
        recon = self.decoder(latent)
        return {"reconstructed": recon, "latent": latent, "features": top, "input": x}

    def compute_loss(self, outputs):
        return combined_loss(outputs["reconstructed"], outputs["input"])

    def compute_metrics(self, outputs):
        return {"psnr": float(psnr_metric(outputs["reconstructed"], outputs["input"])),
                "ssim": float(ssim_metric(outputs["reconstructed"], outputs["input"]))}

    def get_metrics(self):
        return ["psnr", "ssim"]

    def compute_anomaly_scores(self, outputs):
        preds, targets = outputs["reconstructed"], outputs["input"]
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])


class BackboneUNetAE(nn.Module):
    """Backbone-based UNet-style Autoencoder (with skip connections)"""
    def __init__(self, backbone='resnet34', in_channels=3, out_channels=3,
                 latent_dim=512, img_size=256, weights_dir=None):
        super().__init__()
        self.model_type = "backbone_unet_ae"
        self.backbone_name = backbone

        self.encoder = BackboneFeatureEncoder(backbone=backbone,
                                              in_channels=in_channels,
                                              weights_dir=weights_dir)
        top_ch = self.encoder.top_channels
        skip_ch = self.encoder.skip_channels

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Linear(top_ch, latent_dim)

        self.decoder = BackboneUNetDecoder(out_channels=out_channels,
                                           latent_dim=latent_dim,
                                           img_size=img_size,
                                           skip_channels=skip_ch,
                                           top_channels=top_ch)

    def forward(self, inputs):
        x = inputs["image"]
        s1, s2, s3, s4, top = self.encoder(x)
        pooled = self.gap(top).view(x.size(0), -1)
        latent = self.to_latent(pooled)
        recon = self.decoder(latent, (s1, s2, s3, s4))
        return {"reconstructed": recon, "latent": latent, "features": top, "input": x}

    def compute_loss(self, outputs):
        return combined_loss(outputs["reconstructed"], outputs["input"])

    def compute_metrics(self, outputs):
        return {"psnr": float(psnr_metric(outputs["reconstructed"], outputs["input"])),
                "ssim": float(ssim_metric(outputs["reconstructed"], outputs["input"]))}

    def get_metrics(self):
        return ["psnr", "ssim"]

    def compute_anomaly_scores(self, outputs):
        preds, targets = outputs["reconstructed"], outputs["input"]
        return torch.mean((preds - targets) ** 2, dim=[1, 2, 3])
