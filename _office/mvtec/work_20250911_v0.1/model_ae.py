import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

from model_base import TimmFeatureExtractor, ConvBlock, DeconvBlock


# =============================================================================
# Vanilla AutoEncoder with Backbones
# =============================================================================

class VanillaEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, backbone=None, layers=None):
        super().__init__()

        if backbone is not None and layers is not None:
            # TimmFeatureExtractor 사용
            self.use_backbone = True
            self.backbone_extractor = TimmFeatureExtractor(
                backbone=backbone,
                layers=layers,
                pre_trained=True,
                requires_grad=True
            )
            # Backbone output dimensions 계산
            self.feature_dims = self.backbone_extractor.out_dims
            total_features = sum(self.feature_dims)

            # Feature fusion layer
            self.feature_fusion = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(total_features, latent_dim)
            )
        else:
            # 기존 Conv-based encoder (backbone=None인 경우)
            self.use_backbone = False
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
        if self.use_backbone:
            # TimmFeatureExtractor 기반 처리
            features_dict = self.backbone_extractor(x)

            # Multi-scale features 결합
            fused_features = []
            for layer_name in self.backbone_extractor.layers:
                feat = features_dict[layer_name]
                # Global Average Pooling for each scale
                pooled = F.adaptive_avg_pool2d(feat, (1, 1))
                fused_features.append(pooled)

            # Concatenate all scale features
            combined = torch.cat(fused_features, dim=1)  # [B, sum(dims), 1, 1]
            latent = self.feature_fusion(combined)

            # Return latent and multi-scale features
            return latent, features_dict

        else:
            # 기존 Conv-based 방식 (backbone=None)
            features = self.conv_blocks(x)
            pooled = self.pool(features)
            pooled = pooled.view(pooled.size(0), -1)
            latent = self.fc(pooled)
            return latent, features


class VanillaDecoder(nn.Module):
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
            nn.Tanh(),
        ]
        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VanillaAE(nn.Module):
    """Vanilla AutoEncoder with optional TimmFeatureExtractor backbone."""

    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256,
                 backbone=None, layers=None):
        super().__init__()
        self.backbone = backbone
        self.layers = layers

        self.encoder = VanillaEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            backbone=backbone,
            layers=layers
        )
        self.decoder = VanillaDecoder(out_channels, latent_dim, img_size)
        self.model_type = "vanilla_ae"

    def compute_anomaly_map(self, original, reconstructed):
        anomaly_map = torch.mean((original - reconstructed)**2, dim=1, keepdim=True)
        return anomaly_map

    def compute_anomaly_score(self, anomaly_map: torch.Tensor):
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score

    def forward(self, images):
        latent, features = self.encoder(images)
        reconstructed = self.decoder(latent)

        if self.training:
            return reconstructed, latent, features
        else:
            return reconstructed
            # anomaly_map = self.compute_anomaly_map(images, reconstructed)
            # pred_score = self.compute_anomaly_score(anomaly_map)
            # return {'pred_score': pred_score, 'anomaly_map': anomaly_map}


# =============================================================================
# Unet-style AutoEncoder with Backbones
# =============================================================================

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, backbone=None, layers=None):
        super().__init__()

        if backbone is not None and layers is not None:
            # TimmFeatureExtractor 사용
            self.use_backbone = True
            self.backbone_extractor = TimmFeatureExtractor(
                backbone=backbone,
                layers=layers,
                pre_trained=True,
                requires_grad=True
            )
            self.feature_dims = self.backbone_extractor.out_dims

            # Latent projection from deepest features
            deepest_dim = self.feature_dims[-1]  # 가장 깊은 layer의 차원
            self.latent_projection = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(deepest_dim, latent_dim)
            )
        else:
            # 기존 Conv-based encoder (backbone=None인 경우)
            self.use_backbone = False
            self.conv1 = ConvBlock(in_channels, 32)
            self.conv2 = ConvBlock(32, 64)
            self.conv3 = ConvBlock(64, 128)
            self.conv4 = ConvBlock(128, 256)
            self.conv5 = ConvBlock(256, 512)

            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        if self.use_backbone:
            # TimmFeatureExtractor 기반 처리
            features_dict = self.backbone_extractor(x)

            # Skip connections: multi-scale features
            skip_connections = []
            for layer_name in self.backbone_extractor.layers[:-1]:  # 마지막 제외
                skip_connections.append(features_dict[layer_name])

            # Latent from deepest features
            deepest_features = features_dict[self.backbone_extractor.layers[-1]]
            latent = self.latent_projection(deepest_features)
            return latent, deepest_features, skip_connections
        else:
            # 기존 Conv-based U-Net encoder (backbone=None)
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
    def __init__(self, out_channels=3, latent_dim=512, img_size=256, backbone_dims=None):
        super().__init__()
        self.img_size = img_size
        self.backbone_dims = backbone_dims
        self.use_backbone = backbone_dims is not None

        # 공통 초기 설정
        if self.img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {self.img_size}")
        self.start_size = self.img_size // 32

        self.fc = nn.Linear(latent_dim, 512 * self.start_size * self.start_size)
        self.unflatten = nn.Unflatten(1, (512, self.start_size, self.start_size))

        if self.use_backbone:
            # TimmFeatureExtractor 기반 decoder
            self.deconv1 = DeconvBlock(512, 256)

            # Skip connection 차원에 맞춰 동적 조정
            skip_dim_3 = backbone_dims[-2] if len(backbone_dims) > 1 else 256
            skip_dim_2 = backbone_dims[-3] if len(backbone_dims) > 2 else 128
            skip_dim_1 = backbone_dims[-4] if len(backbone_dims) > 3 else 64

            self.deconv2 = DeconvBlock(256 + skip_dim_3, 128)
            self.deconv3 = DeconvBlock(128 + skip_dim_2, 64)
            self.deconv4 = DeconvBlock(64 + skip_dim_1, 32)
            self.deconv5 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)

        else:
            # 기존 Conv-based decoder (backbone=None)
            self.deconv1 = DeconvBlock(512, 256)
            self.deconv2 = DeconvBlock(256 + 256, 128)
            self.deconv3 = DeconvBlock(128 + 128, 64)
            self.deconv4 = DeconvBlock(64 + 64, 32)
            self.deconv5 = nn.ConvTranspose2d(32 + 32, out_channels, 4, 2, 1)

        self.final_activation = nn.Tanh()

    def forward(self, latent, skip_connections):
        """Forward with backbone-aware skip connection handling."""
        x = self.fc(latent)
        x = self.unflatten(x)

        d1 = self.deconv1(x)  # 256, H/16

        if self.use_backbone:
            # Backbone skip connections - 크기 및 차원 맞춤
            skip_3 = skip_connections[-1] if len(skip_connections) > 0 else None
            skip_2 = skip_connections[-2] if len(skip_connections) > 1 else None
            skip_1 = skip_connections[-3] if len(skip_connections) > 2 else None

            # Interpolate skip connections to match decoder sizes
            if skip_3 is not None:
                skip_3 = F.interpolate(skip_3, size=d1.shape[-2:], mode='bilinear', align_corners=False)
                d2 = self.deconv2(torch.cat([d1, skip_3], dim=1))
            else:
                # Skip connection이 없으면 decoder만 사용
                d2 = self.deconv2(d1)

            if skip_2 is not None:
                skip_2 = F.interpolate(skip_2, size=d2.shape[-2:], mode='bilinear', align_corners=False)
                d3 = self.deconv3(torch.cat([d2, skip_2], dim=1))
            else:
                d3 = self.deconv3(d2)

            if skip_1 is not None:
                skip_1 = F.interpolate(skip_1, size=d3.shape[-2:], mode='bilinear', align_corners=False)
                d4 = self.deconv4(torch.cat([d3, skip_1], dim=1))
            else:
                d4 = self.deconv4(d3)

            d5 = self.deconv5(d4)

        else:
            # 기존 Conv-based skip connections (backbone=None)
            d2 = self.deconv2(torch.cat([d1, skip_connections[3]], dim=1))
            d3 = self.deconv3(torch.cat([d2, skip_connections[2]], dim=1))
            d4 = self.deconv4(torch.cat([d3, skip_connections[1]], dim=1))
            d5 = self.deconv5(torch.cat([d4, skip_connections[0]], dim=1))

        reconstructed = self.final_activation(d5)
        return reconstructed


class UNetAE(nn.Module):
    """U-Net AutoEncoder with optional TimmFeatureExtractor backbone."""

    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, img_size=256,
                 backbone=None, layers=None):
        super().__init__()

        self.backbone = backbone
        self.layers = layers

        # Encoder with optional backbone (None이면 기존 방식)
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            backbone=backbone,
            layers=layers
        )

        # Decoder with backbone dimension awareness
        backbone_dims = None
        if backbone is not None and layers is not None:
            # TimmFeatureExtractor 초기화하여 차원 정보 획득
            temp_extractor = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=False)
            backbone_dims = temp_extractor.out_dims

        self.decoder = UNetDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            img_size=img_size,
            backbone_dims=backbone_dims
        )

    def compute_anomaly_map(self, original, reconstructed):
        anomaly_map = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        return anomaly_map

    def compute_anomaly_score(self, anomaly_map):
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score

    def forward(self, images):
        latent, features, skip_connections = self.encoder(images)
        reconstructed = self.decoder(latent, skip_connections)

        if self.training:
            return reconstructed, latent, features
        else:
            return reconstructed
            # anomaly_map = self.compute_anomaly_map(images, reconstructed)
            # pred_score = self.compute_anomaly_score(anomaly_map)
            # return {'pred_score': pred_score, 'anomaly_map': anomaly_map}


class AELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, reconstructed, original):
        return F.mse_loss(reconstructed, original, reduction=self.reduction)


if __name__ == "__main__":
    pass
