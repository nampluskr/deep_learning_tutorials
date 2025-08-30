# models/model_padim.py
# PaDiM: Patch Distribution Modeling (원본 anomalib 구조 반영)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.model_base import ResNetFeatureExtractor


class PadimModel(nn.Module):
    """PaDiM: Patch Distribution Modeling
    Reference: anomalib/models/padim/torch_model.py
    """

    def __init__(self, backbone: str = "resnet18", layers=["layer1", "layer2", "layer3"]):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(backbone, layers)
        self.mean = None
        self.inv_cov = None

    def forward(self, x: torch.Tensor):
        """Extract multi-scale feature maps"""
        feats = self.feature_extractor(x)
        return feats

    def fit(self, train_loader, device="cuda"):
        """Fit Gaussian distribution of patch features using training (normal) data"""
        features = []
        for batch in train_loader:
            imgs = batch["image"].to(device)
            feats = self.forward(imgs)
            # Global average pooling → flatten
            feats = [F.adaptive_avg_pool2d(f, (32, 32)) for f in feats]  # unify size
            feats = torch.cat([f.flatten(2) for f in feats], dim=1)      # [B, C, N]
            features.append(feats.permute(0, 2, 1).reshape(-1, feats.size(1)))
        features = torch.cat(features, dim=0).cpu().numpy()

        # Fit multivariate Gaussian (mean & covariance)
        self.mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False) + 0.01 * np.eye(features.shape[1])
        self.inv_cov = np.linalg.inv(cov)

    def compute_anomaly_scores(self, feats: list) -> torch.Tensor:
        """Compute Mahalanobis distance per patch"""
        pooled = [F.adaptive_avg_pool2d(f, (32, 32)) for f in feats]
        pooled = torch.cat([f.flatten(2) for f in pooled], dim=1)       # [B,C,N]
        pooled = pooled.permute(0, 2, 1)                               # [B,N,C]

        mean = torch.tensor(self.mean, device=pooled.device).float()
        inv_cov = torch.tensor(self.inv_cov, device=pooled.device).float()

        dists = []
        for b in range(pooled.size(0)):
            diff = pooled[b] - mean
            dist = torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=1))
            dists.append(dist.view(32, 32))
        return torch.stack(dists, dim=0)  # [B,32,32]


class PadimLoss(nn.Module):
    """Dummy loss (no training for PaDiM)"""
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets=None):
        return torch.tensor(0.0, requires_grad=True, device=outputs[0].device)


# -------------------------
# Anomaly Map
# -------------------------

def compute_anomaly_map(distance_map: torch.Tensor, out_size=(256, 256)):
    """Generate anomaly map from Mahalanobis distance map"""
    anomaly_map = F.interpolate(distance_map.unsqueeze(1),
                                size=out_size,
                                mode="bilinear",
                                align_corners=False)
    return anomaly_map.squeeze(1)
