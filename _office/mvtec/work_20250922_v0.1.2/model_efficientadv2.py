import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional


# Backbone directory path
# BACKBONE_DIR = '/mnt/d/backbones'
BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "resnet50": "resnet50-0676ba61.pth",
}


def get_backbone_path(backbone: str) -> str:
    """Get local weight path for backbone model."""
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


class ResNet50TeacherEncoder(nn.Module):
    """ResNet50-based teacher encoder with multi-scale features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet50 model
        self.model = models.resnet50(weights=None)

        # Load pretrained weights
        if pretrained:
            weight_path = get_backbone_path("resnet50")
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                print(f" > Loaded pretrained weights for ResNet50 from {weight_path}")
            else:
                print(f" > No local weight file found for ResNet50, using random init.")

        # Extract feature layers
        self.layer0 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1
        )
        self.layer1 = self.model.layer2
        self.layer2 = self.model.layer3
        self.layer3 = self.model.layer4

        # Feature dimensions for ResNet50
        self.feature_dims = [256, 512, 1024, 2048]

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Remove classifier
        self.model = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []

        x = self.layer0(x)  # 1/4 resolution, 256 channels
        features.append(x)

        x = self.layer1(x)  # 1/8 resolution, 512 channels
        features.append(x)

        x = self.layer2(x)  # 1/16 resolution, 1024 channels
        features.append(x)

        x = self.layer3(x)  # 1/32 resolution, 2048 channels
        features.append(x)

        return features


class ResNet50StudentEncoder(nn.Module):
    """ResNet50-based student encoder (trainable)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet50 model
        self.model = models.resnet50(weights=None)

        # Load pretrained weights
        if pretrained:
            weight_path = get_backbone_path("resnet50")
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                print(f" > Loaded pretrained weights for ResNet50 from {weight_path}")
            else:
                print(f" > No local weight file found for ResNet50, using random init.")

        # Extract feature layers
        self.layer0 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1
        )
        self.layer1 = self.model.layer2
        self.layer2 = self.model.layer3
        self.layer3 = self.model.layer4

        # Feature dimensions for ResNet50
        self.feature_dims = [256, 512, 1024, 2048]

        # Remove classifier
        self.model = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []

        x = self.layer0(x)  # 1/4 resolution, 256 channels
        features.append(x)

        x = self.layer1(x)  # 1/8 resolution, 512 channels
        features.append(x)

        x = self.layer2(x)  # 1/16 resolution, 1024 channels
        features.append(x)

        x = self.layer3(x)  # 1/32 resolution, 2048 channels
        features.append(x)

        return features


class TeacherAdapterV2(nn.Module):
    """Teacher adapter with Manual model's multi-scale processing."""

    def __init__(self, teacher: ResNet50TeacherEncoder, out_channels: int):
        super().__init__()
        self.teacher = teacher

        # Multi-scale adapter like Manual model
        total_channels = sum(teacher.feature_dims)
        self.projection = nn.Conv2d(
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale feature processing."""
        features = self.teacher(x)

        # Use largest feature map as reference
        target_h, target_w = features[0].shape[2:]

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )
            aligned_features.append(feat)

        # Concatenate and project
        concat_feat = torch.cat(aligned_features, dim=1)
        projected = self.projection(concat_feat)

        return projected


class StudentAdapterV2(nn.Module):
    """Student adapter with Manual model's multi-scale processing."""

    def __init__(self, student: ResNet50StudentEncoder, out_channels: int):
        super().__init__()
        self.student = student

        # Multi-scale adapter like Manual model
        total_channels = sum(student.feature_dims)
        self.projection = nn.Conv2d(
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale feature processing."""
        features = self.student(x)

        # Use largest feature map as reference
        target_h, target_w = features[0].shape[2:]

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )
            aligned_features.append(feat)

        # Concatenate and project
        concat_feat = torch.cat(aligned_features, dim=1)
        projected = self.projection(concat_feat)

        return projected


class AutoEncoderDecoderV2(nn.Module):
    """Decoder for EfficientADV2 autoencoder part."""

    def __init__(self, feature_channels: int, img_size: int = 256):
        super().__init__()
        self.img_size = img_size

        # ResNet50의 layer0 출력은 1/4 해상도 (256x256 -> 64x64)
        # MultiScaleAdapter는 가장 큰 feature map(layer0 출력)을 기준으로 함
        feature_map_size = img_size // 4  # 64x64

        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(feature_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Final conv to get RGB channels
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to image."""
        return self.decoder(x)


class EfficientADV2(nn.Module):
    """EfficientADV2 with Manual model's feature processing approach."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 out_channels: int = 128, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.backbone_name = backbone
        self.out_channels = out_channels

        # Teacher encoder (frozen)
        self.teacher_net = ResNet50TeacherEncoder(pretrained)

        # Student encoder (trainable)
        self.student_net = ResNet50StudentEncoder(pretrained)

        # Teacher adapter (frozen like Manual model)
        self.teacher_adapter = TeacherAdapterV2(self.teacher_net, out_channels)

        # Student adapter (trainable like Manual model)
        self.student_adapter = StudentAdapterV2(self.student_net, out_channels)

        # Autoencoder decoder (trainable)
        self.decoder = AutoEncoderDecoderV2(out_channels, img_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            reconstructed: Reconstructed image
            teacher_features: Teacher adapter output
            student_features: Student adapter output
        """
        # Teacher features (frozen)
        with torch.no_grad():
            teacher_features = self.teacher_adapter(x)

        # Student features (trainable)
        student_features = self.student_adapter(x)

        # Reconstruction using teacher features
        reconstructed = self.decoder(teacher_features)

        return reconstructed, teacher_features, student_features

    def _align_feature_maps(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """Align student features to teacher features size."""
        if t_feat.shape[2:] != s_feat.shape[2:]:
            s_feat = F.interpolate(
                s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False
            )
        return s_feat

    def compute_anomaly_map(self, reconstructed: torch.Tensor, 
                        original: torch.Tensor,
                        teacher_features: torch.Tensor,
                        student_features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly map with Manual model's approach + 5% border masking."""
        
        # Align student features
        student_features = self._align_feature_maps(teacher_features, student_features)
        
        # Reconstruction error map
        recon_error = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        
        # Feature difference (normalized like Manual model)
        t_feat = F.normalize(teacher_features, p=2, dim=1)
        s_feat = F.normalize(student_features, p=2, dim=1)
        feature_diff = (t_feat - s_feat).pow(2).mean(dim=1, keepdim=True)
        
        # **수정된 부분: feature difference를 reconstruction error 크기에 맞춤**
        if feature_diff.shape[2:] != recon_error.shape[2:]:
            feature_diff = F.interpolate(
                feature_diff, size=recon_error.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # **디버깅용 출력 (선택사항)**
        # print(f"Debug - recon_error shape: {recon_error.shape}")
        # print(f"Debug - feature_diff shape: {feature_diff.shape}")
        # print(f"Debug - teacher_features shape: {teacher_features.shape}")
        
        # Combine reconstruction error and feature difference
        alpha = 0.5  # Weight for combining two error types
        anomaly_map = alpha * recon_error + (1 - alpha) * feature_diff
        
        # Apply 5% border masking (Manual model's technique)
        b, c, h, w = anomaly_map.shape
        bh, bw = int(h * 0.05), int(w * 0.05)
        
        # Create mask (1 for valid region, 0 for border)
        mask = torch.ones_like(anomaly_map)
        mask[:, :, :bh, :] = 0
        mask[:, :, -bh:, :] = 0
        mask[:, :, :, :bw] = 0
        mask[:, :, :, -bw:] = 0
        
        # Apply mask
        anomaly_map = anomaly_map * mask
        
        return anomaly_map

    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute image-level anomaly score (Manual model's approach)."""
        # Get max score from non-zero regions only
        valid_regions = anomaly_map > 0
        if valid_regions.any():
            score = torch.amax(anomaly_map, dim=(-2, -1))
        else:
            score = torch.zeros(anomaly_map.size(0), device=anomaly_map.device)
        return score

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict anomaly maps and scores."""
        self.eval()
        reconstructed, teacher_features, student_features = self.forward(images)
        anomaly_map = self.compute_anomaly_map(
            reconstructed, images, teacher_features, student_features
        )
        pred_score = self.compute_anomaly_score(anomaly_map)

        return {
            "anomaly_map": anomaly_map,
            "pred_score": pred_score,
            "reconstructed": reconstructed
        }


class EfficientADV2Loss(nn.Module):
    """Combined loss for EfficientADV2 training with Manual model's approach."""

    def __init__(self, recon_weight: float = 0.5, distill_weight: float = 0.5,
                 pretrain_penalty: bool = False, penalty_weight: float = 1e-4):
        super().__init__()
        self.recon_weight = recon_weight
        self.distill_weight = distill_weight
        self.pretrain_penalty = pretrain_penalty
        self.penalty_weight = penalty_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor,
                teacher_features: torch.Tensor, student_features: torch.Tensor,
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute combined loss with Manual model's normalization."""

        # Reconstruction loss
        recon_loss = self.mse_loss(reconstructed, original)

        # Align features
        if teacher_features.shape[2:] != student_features.shape[2:]:
            student_features = F.interpolate(
                student_features, size=teacher_features.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Knowledge distillation loss with normalization (Manual model's approach)
        t_feat = F.normalize(teacher_features, p=2, dim=1)
        s_feat = F.normalize(student_features, p=2, dim=1)
        distill_loss = self.mse_loss(s_feat, t_feat)

        # Combined loss
        total_loss = self.recon_weight * recon_loss + self.distill_weight * distill_loss

        # Add L2 penalty if enabled
        if self.pretrain_penalty and model is not None:
            l2_penalty = sum(p.pow(2).sum() for p in model.student_adapter.parameters())
            l2_penalty += sum(p.pow(2).sum() for p in model.decoder.parameters())
            total_loss = total_loss + self.penalty_weight * l2_penalty

        return total_loss


class EfficientADV2Metric(nn.Module):
    """Metrics for EfficientADV2 evaluation."""

    def __init__(self):
        super().__init__()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor,
                teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between teacher and student features."""

        # Align features
        if teacher_features.shape[2:] != student_features.shape[2:]:
            student_features = F.interpolate(
                student_features, size=teacher_features.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Flatten features
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)
        student_flat = student_features.view(student_features.size(0), -1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(teacher_flat, student_flat, dim=1)
        return similarity.mean()
