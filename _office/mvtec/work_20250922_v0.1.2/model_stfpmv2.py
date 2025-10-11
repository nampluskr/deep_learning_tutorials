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


def gat_backbone_path(backbone: str) -> str:
    """Get local weight path for backbone model."""
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


class ResNet50FeatureExtractor(nn.Module):
    """ResNet50 feature extractor with multi-scale features."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load ResNet50 model
        self.model = models.resnet50(weights=None)
        
        # Load pretrained weights
        if pretrained:
            weight_path = gat_backbone_path("resnet50")
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
        self.feature_dims = [256, 512, 1024, 2048]  # layer1, layer2, layer3, layer4
        
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


class MultiScaleAdapter(nn.Module):
    """Multi-scale feature adapter inspired by Manual model."""
    
    def __init__(self, feature_dims: List[int], out_channels: int):
        super().__init__()
        self.feature_dims = feature_dims
        self.out_channels = out_channels
        
        # 1x1 conv projection (like Manual model)
        total_channels = sum(feature_dims)
        self.projection = nn.Conv2d(
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Align and concatenate multi-scale features."""
        # Use the largest feature map size as target
        target_h, target_w = features[0].shape[2:]
        
        aligned_features = []
        for feat in features:
            if feat.shape[2:] != (target_h, target_w):
                # Upsample to target size
                feat = F.interpolate(
                    feat, size=(target_h, target_w), 
                    mode='bilinear', align_corners=False
                )
            aligned_features.append(feat)
        
        # Concatenate along channel dimension
        concat_feat = torch.cat(aligned_features, dim=1)
        
        # Project to output channels
        projected = self.projection(concat_feat)
        
        return projected


class STFPMV2(nn.Module):
    """STFPMV2 with Manual model's feature processing approach."""
    
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, 
                 out_channels: int = 128):
        super().__init__()
        self.backbone_name = backbone
        self.out_channels = out_channels
        
        # Teacher network (frozen)
        self.teacher_extractor = ResNet50FeatureExtractor(pretrained)
        for param in self.teacher_extractor.parameters():
            param.requires_grad = False
        
        # Student network (trainable backbone)
        self.student_extractor = ResNet50FeatureExtractor(pretrained)
        
        # Teacher adapter (frozen like Manual model)
        self.teacher_adapter = MultiScaleAdapter(
            self.teacher_extractor.feature_dims, out_channels
        )
        for param in self.teacher_adapter.parameters():
            param.requires_grad = False
        
        # Student adapter (trainable like Manual model)
        self.student_adapter = MultiScaleAdapter(
            self.student_extractor.feature_dims, out_channels
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning teacher and student features."""
        # Teacher features (frozen)
        with torch.no_grad():
            teacher_features = self.teacher_extractor(x)
            teacher_proj = self.teacher_adapter(teacher_features)
        
        # Student features (trainable)
        student_features = self.student_extractor(x)
        student_proj = self.student_adapter(student_features)
        
        return teacher_proj, student_proj
    
    def _align_feature_maps(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """Align student features to teacher features size."""
        if t_feat.shape[2:] != s_feat.shape[2:]:
            s_feat = F.interpolate(
                s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False
            )
        return s_feat
    
    def compute_anomaly_map(self, teacher_features: torch.Tensor, 
                           student_features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly map with Manual model's approach + 5% border masking."""
        # Align features
        student_features = self._align_feature_maps(teacher_features, student_features)
        
        # Normalize features (like Manual model)
        t_feat = F.normalize(teacher_features, p=2, dim=1)
        s_feat = F.normalize(student_features, p=2, dim=1)
        
        # Compute L2 distance
        diff = (t_feat - s_feat).pow(2).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply 5% border masking (Manual model's technique)
        b, c, h, w = diff.shape
        bh, bw = int(h * 0.05), int(w * 0.05)
        
        # Create mask (1 for valid region, 0 for border)
        mask = torch.ones_like(diff)
        mask[:, :, :bh, :] = 0
        mask[:, :, -bh:, :] = 0
        mask[:, :, :, :bw] = 0
        mask[:, :, :, -bw:] = 0
        
        # Apply mask
        anomaly_map = diff * mask
        
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
        teacher_features, student_features = self.forward(images)
        anomaly_map = self.compute_anomaly_map(teacher_features, student_features)
        pred_score = self.compute_anomaly_score(anomaly_map)
        
        return {
            "anomaly_map": anomaly_map,
            "pred_score": pred_score
        }


class STFPMV2Loss(nn.Module):
    """Loss function for STFPMV2 with Manual model's approach."""
    
    def __init__(self, pretrain_penalty: bool = False, penalty_weight: float = 1e-4):
        super().__init__()
        self.pretrain_penalty = pretrain_penalty
        self.penalty_weight = penalty_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features: torch.Tensor, student_features: torch.Tensor,
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute knowledge distillation loss like Manual model."""
        # Align features
        if teacher_features.shape[2:] != student_features.shape[2:]:
            student_features = F.interpolate(
                student_features, size=teacher_features.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # Normalize features before loss computation (Manual model's approach)
        t_feat = F.normalize(teacher_features, p=2, dim=1)
        s_feat = F.normalize(student_features, p=2, dim=1)
        
        # MSE loss between normalized features
        loss = self.mse_loss(s_feat, t_feat)
        
        # Add L2 penalty if enabled
        if self.pretrain_penalty and model is not None:
            l2_penalty = sum(p.pow(2).sum() for p in model.student_adapter.parameters())
            loss = loss + self.penalty_weight * l2_penalty
        
        return loss


class STFPMV2Metric(nn.Module):
    """Metrics for STFPMV2 evaluation."""
    
    def __init__(self):
        super().__init__()

    def forward(self, teacher_features: torch.Tensor, 
                student_features: torch.Tensor) -> torch.Tensor:
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
