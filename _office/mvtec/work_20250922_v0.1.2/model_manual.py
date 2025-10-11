import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional


# Backbone directory path (from existing code)
BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "efficientnet_b7": "efficientnet_b7_lukemelas-c5b4e57e.pth",
    "wide_resnet101_2": "wide_resnet101_2-32ee1156.pth",
}


def get_backbone_path(backbone: str) -> str:
    """Get local weight path for backbone model."""
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


class TeacherEfficientNet(nn.Module):
    """Teacher network using EfficientNet-B7."""
    
    def __init__(self, backbone_path: str, device: torch.device):
        super().__init__()
        if not os.path.isfile(backbone_path):
            raise FileNotFoundError(f"Backbone weight not found: {backbone_path}")

        self.backbone = models.efficientnet_b7(weights=None)
        state = torch.load(backbone_path, map_location=device)
        self.backbone.load_state_dict(state, strict=False)
        self.backbone.classifier = nn.Identity()
        self.backbone = self.backbone.to(device).eval()
        self.stage_idxs = [2, 4, 6]  # Middle 3 layers

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        feats = []
        out = x
        for i, block in enumerate(self.backbone.features):
            out = block(out)
            if i in self.stage_idxs:
                feats.append(out)
        return feats


class TeacherAdapter(nn.Module):
    """Adapter for teacher features with fixed projection."""
    
    def __init__(self, teacher: TeacherEfficientNet, out_channels: int, device: torch.device):
        super().__init__()
        self.teacher = teacher
        self.device = device
        self.proj = nn.Conv2d(
            in_channels=self._calc_teacher_channels(),
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        ).to(device)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _calc_teacher_channels(self) -> int:
        """Calculate total teacher feature channels."""
        dummy = torch.randn(1, 3, 256, 256, device=self.device)
        feats = self.teacher(dummy)
        return sum(f.shape[1] for f in feats)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feature alignment and projection."""
        feats = self.teacher(x)
        th, tw = feats[0].shape[2:]
        aligned = []
        
        for f in feats:
            if f.shape[2:] != (th, tw):
                f = F.interpolate(f, size=(th, tw), mode='bilinear', align_corners=False)
            aligned.append(f)
        
        cat = torch.cat(aligned, dim=1)
        return self.proj(cat)


class StudentWideResNet(nn.Module):
    """Student network using Wide-ResNet-101-2."""
    
    def __init__(self, backbone_path: str, device: torch.device):
        super().__init__()
        if not os.path.isfile(backbone_path):
            raise FileNotFoundError(f"ResNet weight not found: {backbone_path}")

        self.backbone = models.wide_resnet101_2(weights=None)
        state = torch.load(backbone_path, map_location=device)
        self.backbone.load_state_dict(state, strict=False)
        self.backbone.fc = nn.Identity()
        self.backbone = self.backbone.to(device)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer1 (required)
        x = self.backbone.layer1(x)

        # Extract features from layer2, layer3, layer4
        feats = []
        x = self.backbone.layer2(x)  # 1/8
        feats.append(x)
        x = self.backbone.layer3(x)  # 1/16
        feats.append(x)
        x = self.backbone.layer4(x)  # 1/32
        feats.append(x)
        return feats


class StudentAdapter(nn.Module):
    """Adapter for student features with trainable projection."""
    
    def __init__(self, student: StudentWideResNet, out_channels: int, device: torch.device):
        super().__init__()
        self.student = student
        self.device = device
        self.proj = nn.Conv2d(
            in_channels=self._calc_student_channels(),
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        ).to(device)

    def _calc_student_channels(self) -> int:
        """Calculate total student feature channels."""
        dummy = torch.randn(1, 3, 256, 256, device=self.device)
        with torch.no_grad():
            feats = self.student(dummy)
        return sum(f.shape[1] for f in feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feature alignment and projection."""
        feats = self.student(x)
        th, tw = feats[0].shape[2:]
        aligned = []
        
        for f in feats:
            if f.shape[2:] != (th, tw):
                f = F.interpolate(f, size=(th, tw), mode='bilinear', align_corners=False)
            aligned.append(f)
        
        cat = torch.cat(aligned, dim=1)
        return self.proj(cat)


class ManualEfficientAD(nn.Module):
    """Manual implementation of EfficientAD with teacher-student architecture."""
    
    def __init__(self, teacher_backbone_path: str, student_backbone_path: str, 
                 out_channels: int = 128, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_channels = out_channels

        # Initialize teacher and student networks
        self.teacher_net = TeacherEfficientNet(teacher_backbone_path, self.device)
        self.student_net = StudentWideResNet(student_backbone_path, self.device)
        
        # Initialize adapters
        self.teacher_adapter = TeacherAdapter(self.teacher_net, out_channels, self.device)
        self.student_adapter = StudentAdapter(self.student_net, out_channels, self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            teacher_features: Teacher adapter output
            student_features: Student adapter output
        """
        # Teacher features (frozen)
        with torch.no_grad():
            teacher_features = self.teacher_adapter(x)
        
        # Student features (trainable)
        student_features = self.student_adapter(x)
        
        return teacher_features, student_features

    def _align_feature_maps(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """Align student features to teacher features size."""
        if t_feat.shape[2:] != s_feat.shape[2:]:
            s_feat = F.interpolate(
                s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False
            )
        return s_feat

    def compute_anomaly_map(self, teacher_features: torch.Tensor, 
                           student_features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly map with 5% border masking."""
        # Align features
        student_features = self._align_feature_maps(teacher_features, student_features)
        
        # Normalize features
        t_feat = F.normalize(teacher_features, p=2, dim=1)
        s_feat = F.normalize(student_features, p=2, dim=1)
        
        # Compute L2 distance
        diff = (t_feat - s_feat).pow(2).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply 5% border masking
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
        """Compute image-level anomaly score (max score from valid region)."""
        # Get max score from non-zero regions
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


class ManualEfficientADLoss(nn.Module):
    """Loss function for Manual EfficientAD training."""
    
    def __init__(self, pretrain_penalty: bool = False, penalty_weight: float = 1e-4):
        super().__init__()
        self.pretrain_penalty = pretrain_penalty
        self.penalty_weight = penalty_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features: torch.Tensor, student_features: torch.Tensor,
                model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            teacher_features: Teacher adapter output
            student_features: Student adapter output  
            model: Model for L2 penalty (optional)
        """
        # Align features
        if teacher_features.shape[2:] != student_features.shape[2:]:
            student_features = F.interpolate(
                student_features, size=teacher_features.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # MSE loss between teacher and student features
        loss = self.mse_loss(student_features, teacher_features)
        
        # Add L2 penalty if enabled
        if self.pretrain_penalty and model is not None:
            l2_penalty = sum(p.pow(2).sum() for p in model.student_adapter.parameters())
            loss = loss + self.penalty_weight * l2_penalty
        
        return loss


class ManualEfficientADMetric(nn.Module):
    """Metrics for Manual EfficientAD evaluation."""
    
    def __init__(self):
        super().__init__()

    def forward(self, teacher_features: torch.Tensor, 
                student_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between teacher and student features.
        
        Returns:
            Average cosine similarity
        """
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
