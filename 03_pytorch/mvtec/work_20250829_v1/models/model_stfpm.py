# models/model_stfpm.py
# STFPM: Student-Teacher Feature Pyramid Matching (원본 anomalib 구조 반영)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import ResNetFeatureExtractor


class STFPMModel(nn.Module):
    """STFPM: Student-Teacher Feature Pyramid Matching
    Reference: anomalib/models/stfpm/torch_model.py
    """

    def __init__(self, backbone: str = "resnet18", layers=["layer1", "layer2", "layer3"]):
        super().__init__()
        self.teacher = ResNetFeatureExtractor(backbone, layers)
        self.student = ResNetFeatureExtractor(backbone, layers)

        # Freeze teacher parameters
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Return teacher & student features"""
        t_feats = self.teacher(x)
        s_feats = self.student(x)
        return t_feats, s_feats

    def compute_anomaly_scores(self, outputs):
        """Compute mean reconstruction error across feature maps"""
        t_feats, s_feats = outputs
        anomaly_map = compute_anomaly_map(t_feats, s_feats, out_size=(t_feats[-1].shape[-2], t_feats[-1].shape[-1]))
        score = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
        return score


class STFPMLoss(nn.Module):
    """Feature Matching Loss between Teacher and Student"""

    def __init__(self):
        super().__init__()

    def forward(self, outputs, _: torch.Tensor = None) -> torch.Tensor:
        t_feats, s_feats = outputs
        loss = 0.0
        for t, s in zip(t_feats, s_feats):
            loss += F.mse_loss(s, t)
        return loss / len(t_feats)


# -------------------------
# Anomaly Map
# -------------------------

def compute_anomaly_map(teacher_features, student_features, out_size=(256, 256)):
    """Generate anomaly map for STFPM
    Args:
        teacher_features: list of feature maps from teacher
        student_features: list of feature maps from student
    """
    anomaly_maps = []
    for t, s in zip(teacher_features, student_features):
        anomaly_maps.append(torch.mean((t - s) ** 2, dim=1, keepdim=True))

    anomaly_map = torch.mean(torch.stack(anomaly_maps), dim=0)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode="bilinear", align_corners=False)
    return anomaly_map
