import os
from collections.abc import Sequence
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.models as models

from feature_extractor import TimmFeatureExtractor
from trainer import BaseTrainer


BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "efficientnet_b7": "efficientnet_b7_lukemelas-c5b4e57e.pth",
    "wide_resnet101_2": "wide_resnet101_2-32ee1156.pth",
}


def gat_backbone_path(backbone: str) -> str:
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


###########################################################
# Student-Teacher Model version 1
###########################################################

class STMV1(nn.Module):
    """Manual implementation of EfficientAD with teacher-student architecture."""

    def __init__(self, teacher_backbone_path=None, student_backbone_path=None,
                 out_channels: int = 128, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_channels = out_channels

        self.teacher_backbone_path = teacher_backbone_path or gat_backbone_path("efficientnet_b7")
        self.student_backbone_path = student_backbone_path or gat_backbone_path("wide_resnet101_2")

        # Initialize teacher and student networks
        self.teacher_net = TeacherEfficientNet(self.teacher_backbone_path, self.device)
        self.student_net = StudentWideResNet(self.student_backbone_path, self.device)

        # Initialize adapters
        self.teacher_adapter = TeacherAdapter(self.teacher_net, out_channels, self.device)
        self.student_adapter = StudentAdapter(self.student_net, out_channels, self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


###########################################################
# Loss and Metric for Student-Teacher Model
###########################################################

class STMLoss(nn.Module):
    def __init__(self, pretrain_penalty: bool = False, penalty_weight: float = 1e-4):
        super().__init__()
        self.pretrain_penalty = pretrain_penalty
        self.penalty_weight = penalty_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features: torch.Tensor, student_features: torch.Tensor,
                model: Optional[nn.Module] = None) -> torch.Tensor:

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


class STMMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_features: torch.Tensor,
                student_features: torch.Tensor) -> torch.Tensor:
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


#############################################################
# Trainer for STMV1 Model
#############################################################

class STMTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None,
                 use_amp=True, pretrain_penalty=False):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.pretrain_penalty = pretrain_penalty

        student_params = list(self.model.student_adapter.parameters())
        self.optimizer = optimizer or optim.AdamW(student_params, lr=1e-4, weight_decay=1e-5)
        self.loss_fn = loss_fn or STMLoss(pretrain_penalty=pretrain_penalty)
        self.metrics = metrics or {'similarity': STMMetric()}

        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def run_epoch(self, loader, mode='train', desc=""):
        """Run one epoch of training or validation."""
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0

        # Set model mode
        if mode == 'train':
            self.model.student_adapter.train()
            self.model.teacher_adapter.eval()
        else:
            self.model.eval()

        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                else:
                    # Handle tuple format (img, label, path) or (img_teacher, img_student), label, path
                    if len(batch) == 3 and isinstance(batch[0], tuple):
                        # Training format with augmentation: ((img_teacher, img_student), label, path)
                        img_teacher, img_student = batch[0]
                        img_teacher = img_teacher.to(self.device, non_blocking=True)
                        img_student = img_student.to(self.device, non_blocking=True)
                        # Use student image for processing
                        images = img_student
                    else:
                        # Standard format: (img, label, path)
                        images = batch[0].to(self.device)

                batch_size = images.size(0)
                num_images += batch_size

                if mode == 'train' and self.use_amp:
                    with torch.cuda.amp.autocast():
                        teacher_features, student_features = self.model(images)
                        loss = self.loss_fn(teacher_features, student_features,
                                            model=self.model if self.pretrain_penalty else None)
                else:
                    teacher_features, student_features = self.model(images)
                    loss = self.loss_fn(teacher_features, student_features,
                                      model=self.model if self.pretrain_penalty else None)

                if mode == 'train':
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.student_adapter.parameters(), max_norm=5.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.student_adapter.parameters(), max_norm=5.0)
                        self.optimizer.step()

                results["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        metric_value = metric_fn(teacher_features, student_features)
                        results[name] += metric_value.item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})

        return {name: value / num_images for name, value in results.items()}

    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):
        return self.test_feature_based(test_loader, output_dir, show_image, img_prefix,
            skip_normal, skip_anomaly, num_max)
