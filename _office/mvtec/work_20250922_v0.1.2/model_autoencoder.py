import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ssim import ssim


# BACKBONE_DIR = r"D:\Non_Documents\2025\1_project\1_image_processing\modeling\mvtec_office\backbones"
# BACKBONE_DIR = '/mnt/d/backbones'
BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
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
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 32 * 32, out_features=latent_dim)
        )
        self.from_linear = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256 * 32 * 32),
            nn.Unflatten(dim=1, unflattened_size=(256, 32, 32)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
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
# Conv Block / Deconv Block / Autoencoder
###########################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.deconv_block(x)


class AutoEncoder(Baseline):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        feat_size = img_size // 8
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * feat_size * feat_size, latent_dim),
        )
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * feat_size * feat_size),
            nn.Unflatten(dim=1, unflattened_size=(256, feat_size, feat_size)),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )


###########################################################
# Adaptive Pooling AutoEncoder
###########################################################

class AdaptiveAE(Baseline):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, input_size=256, pool_size=8):
        super().__init__()
        self.input_size = input_size
        self.pool_size = pool_size

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pool_size * pool_size, latent_dim),
        )
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * pool_size * pool_size),
            nn.Unflatten(dim=1, unflattened_size=(256, pool_size, pool_size)),
        )

        layers = []
        in_ch = 256
        ch_list = [128, 64, 32, 16]
        step = 0
        cur_size = pool_size
        while cur_size < input_size:
            out_ch = ch_list[step] if step < len(ch_list) else 16
            layers.append(DeconvBlock(in_ch, out_ch))   # 2배 업샘플
            in_ch = out_ch
            cur_size *= 2
            step += 1

        layers.append(nn.ConvTranspose2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.pool(latent)
        latent_vec = self.to_linear(latent)
        latent = self.from_linear(latent_vec)
        recon = self.decoder(latent)
        return recon, latent_vec


###########################################################
# Fully Convolutional AutoEncoder
###########################################################

class ConvAE(Baseline):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, latent_dim),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            # nn.InstanceNorm2d(latent_dim, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(latent_dim, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.bottleneck(latent)
        recon = self.decoder(latent)
        return recon, latent


###########################################################
# Patch-based AutoEncoder
###########################################################

class PatchAE(Baseline):
    def __init__(self, base_ae, patch_size=256, stride=256):
        """
        Patch-based AutoEncoder
        - base_ae: 내부적으로 사용할 AE (예: Baseline, FullyConvAE 등)
        - patch_size: 입력 이미지를 자를 patch 크기
        - stride: patch 간격 (겹치게 하고 싶으면 stride < patch_size)
        """
        super().__init__()
        self.base_ae = base_ae
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, images):
        """(N, C, H, W) -> (N*num_patches, C, patch_size, patch_size)"""
        patches = images.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        # patches shape: (N, C, nH, nW, patch_size, patch_size)
        N, C, nH, nW, PH, PW = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, PH, PW)
        return patches, nH, nW

    def reconstruct_from_patches(self, patch_recons, nH, nW, H, W):
        """patch_recons: (N*num_patches, C, patch_size, patch_size) -> (N, C, H, W)"""
        N_patches, C, PH, PW = patch_recons.shape
        N = N_patches // (nH * nW)

        patch_recons = patch_recons.view(N, nH, nW, C, PH, PW).permute(0, 3, 1, 4, 2, 5)
        # (N, C, nH, PH, nW, PW)
        recon = patch_recons.contiguous().view(N, C, nH * PH, nW * PW)

        return recon[:, :, :H, :W]

    def forward(self, x):
        N, C, H, W = x.shape
        patches, nH, nW = self.extract_patches(x)
        patch_recons, *_ = self.base_ae(patches)
        recon = self.reconstruct_from_patches(patch_recons, nH, nW, H, W)
        return recon, None  # latent는 생략


###########################################################
# ResNet-based AutoEncoder
###########################################################

class ResNetAE(Baseline):
    def __init__(self, backbone="resnet18", in_channels=3, out_channels=3, latent_dim=256, img_size=256):
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size

        if backbone == "resnet18":
            base = models.resnet18(weights=None)
            feat_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=None)
            feat_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        weight_path = get_backbone_path(backbone)
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            base.load_state_dict(state_dict)
            print(f" > Loaded pretrained weights for {backbone} from {weight_path}")
        else:
            print(f" > No local weight file found for {backbone}, using random init.")

        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        feat_size = img_size // 32  # downsample factor of ResNet
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim * feat_size * feat_size, latent_dim),
        )
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, feat_dim * feat_size * feat_size),
            nn.Unflatten(dim=1, unflattened_size=(feat_dim, feat_size, feat_size)),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(feat_dim, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.to_linear(latent)
        latent = self.from_linear(latent)
        recon = self.decoder(latent)
        return recon, latent


class ResNetUNetAE(Baseline):
    def __init__(self, backbone="resnet18", pretrained=False, out_channels=3):
        super().__init__()
        assert backbone in ["resnet18", "resnet34", "resnet50"], "Only resnet18/34/50 supported"

        # Load backbone
        if backbone == "resnet18":
            net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            enc_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            net = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            enc_channels = [64, 64, 128, 256, 512]
        else:  # resnet50
            net = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            enc_channels = [64, 256, 512, 1024, 2048]

        # Encoder layers
        self.enc1 = nn.Sequential(net.conv1, net.bn1, net.relu)   # 1/2
        self.enc2 = nn.Sequential(net.maxpool, net.layer1)        # 1/4
        self.enc3 = net.layer2                                    # 1/8
        self.enc4 = net.layer3                                    # 1/16
        self.enc5 = net.layer4                                    # 1/32

        # Decoder layers (mirror structure)
        self.dec4 = self._up_block(enc_channels[4], enc_channels[3])
        self.dec3 = self._up_block(enc_channels[3], enc_channels[2])
        self.dec2 = self._up_block(enc_channels[2], enc_channels[1])
        self.dec1 = self._up_block(enc_channels[1], enc_channels[0])
        self.final = nn.ConvTranspose2d(enc_channels[0], out_channels,
                                        kernel_size=4, stride=2, padding=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # 1/2
        e2 = self.enc2(e1)  # 1/4
        e3 = self.enc3(e2)  # 1/8
        e4 = self.enc4(e3)  # 1/16
        e5 = self.enc5(e4)  # 1/32

        # Decoder with skip connections
        d4 = self.dec4(e5)               # up to 1/16
        d4 = d4 + e4                     # skip
        d3 = self.dec3(d4)               # up to 1/8
        d3 = d3 + e3
        d2 = self.dec2(d3)               # up to 1/4
        d2 = d2 + e2
        d1 = self.dec1(d2)               # up to 1/2
        d1 = d1 + e1
        out = self.final(d1)             # back to full size

        return out, e5  # return reconstruction + latent


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


class SSIMMetric(nn.Module):
    def __init__(self, data_range=2.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        return ssim(preds, targets, data_range=self.data_range, size_average=True)


###########################################################
# Memory-Augmented AutoEncoder
###########################################################

class MemAE(Baseline):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, mem_dim=1000, shrink_thres=0.0025):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 32 * 32, latent_dim)

        # Memory Module
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(mem_dim, latent_dim))

        self.fc_dec = nn.Linear(latent_dim, 256 * 32 * 32)
        self.unflatten = nn.Unflatten(1, (256, 32, 32))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.flatten(z)
        z = self.fc_enc(z)  # (B, latent_dim)

        # Memory addressing
        att_weight = F.softmax(F.linear(z, self.memory), dim=1)  # (B, mem_dim)
        if self.shrink_thres > 0:
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        z_hat = torch.mm(att_weight, self.memory)  # (B, latent_dim)

        out = self.fc_dec(z_hat)
        out = self.unflatten(out)
        recon = self.decoder(out)
        return recon, z_hat

    @staticmethod
    def hard_shrink_relu(x, lambd=0.0025, epsilon=1e-12):
        return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + epsilon)


class ConvMemAE(Baseline):
    def __init__(self, in_channels=3, out_channels=3,
                 latent_dim=256, mem_dim=1000, shrink_thres=0.0025):
        super().__init__(in_channels, out_channels, latent_dim)

        # Encoder (Conv-only)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, latent_dim),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Memory module
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(mem_dim, latent_dim))

        # Decoder
        self.decoder = nn.Sequential(
            DeconvBlock(latent_dim, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.bottleneck(feat)

        B, C, H, W = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        # Memory addressing
        att_weight = F.softmax(F.linear(feat_flat, self.memory), dim=1)
        if self.shrink_thres > 0:
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        feat_mem = torch.mm(att_weight, self.memory)  # (B*H*W, C)

        feat_mem = feat_mem.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        recon = self.decoder(feat_mem)
        return recon, feat_mem

    @staticmethod
    def hard_shrink_relu(x, lambd=0.0025, epsilon=1e-12):
        return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + epsilon)


###########################################################
# Denoising AutoEncoder
###########################################################

class DenoisingAE(Baseline):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 32 * 32, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256 * 32 * 32)
        self.unflatten = nn.Unflatten(1, (256, 32, 32))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1)
        )

    def forward(self, x):
        noisy_x = x + self.noise_std * torch.randn_like(x)
        z = self.encoder(noisy_x)
        z = self.flatten(z)
        z = self.fc_enc(z)
        out = self.fc_dec(z)
        out = self.unflatten(out)
        recon = self.decoder(out)
        return recon, z


class ConvDenoisingAE(Baseline):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, noise_std=0.1):
        super().__init__(in_channels, out_channels, latent_dim)
        self.noise_std = noise_std

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, latent_dim),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            DeconvBlock(latent_dim, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        noisy_x = x + self.noise_std * torch.randn_like(x)
        latent = self.encoder(noisy_x)
        latent = self.bottleneck(latent)
        recon = self.decoder(latent)
        return recon, latent
