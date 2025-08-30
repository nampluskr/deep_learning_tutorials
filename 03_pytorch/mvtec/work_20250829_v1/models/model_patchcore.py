# models/model_patchcore.py
# PatchCore: Memory Bank based Anomaly Detection (원본 anomalib 구조 반영)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import ResNetFeatureExtractor
import numpy as np
from sklearn.neighbors import NearestNeighbors


class PatchcoreModel(nn.Module):
    """PatchCore: Memory Bank based Anomaly Detection
    Reference: anomalib/models/patchcore/torch_model.py
    """

    def __init__(self, backbone: str = "resnet18", layers=["layer2", "layer3"], n_neighbors: int = 1):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(backbone, layers)
        self.embedding_dim = 0
        self.memory_bank = None
        self.n_neighbors = n_neighbors
        self.knn = None

    def forward(self, x: torch.Tensor):
        """Extract backbone features"""
        feats = self.feature_extractor(x)
        # Global average pooling + flatten
        pooled = [F.adaptive_avg_pool2d(f, (32, 32)) for f in feats]
        pooled = torch.cat([f.flatten(2) for f in pooled], dim=1)  # [B,C,N]
        pooled = pooled.permute(0, 2, 1)  # [B,N,C]
        return pooled

    def build_memory_bank(self, train_loader, device="cuda"):
        """Populate memory bank from training data"""
        features = []
        for batch in train_loader:
            imgs = batch["image"].to(device)
            feats = self.forward(imgs)  # [B,N,C]
            feats = feats.reshape(-1, feats.size(-1))  # flatten patches
            features.append(feats.detach().cpu().numpy())
        features = np.concatenate(features, axis=0)
        self.memory_bank = features

        # Build kNN index
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self.knn.fit(self.memory_bank)

    def compute_anomaly_scores(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest memory bank features"""
        feats = feats.reshape(-1, feats.size(-1)).detach().cpu().numpy()
        distances, _ = self.knn.kneighbors(feats)
        scores = torch.tensor(distances, device="cuda", dtype=torch.float32)
        return scores.view(-1)  # flatten distances


class PatchcoreLoss(nn.Module):
    """Dummy loss (PatchCore does not require training)"""
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets=None):
        return torch.tensor(0.0, requires_grad=True)


# -------------------------
# Anomaly Map
# -------------------------

def compute_anomaly_map(patch_scores: torch.Tensor, image_size=(256, 256), patch_size=32):
    """Generate anomaly map from patch scores
    Args:
        patch_scores: [B, H', W'] patch-wise anomaly scores
        image_size: target resize size
    """
    if patch_scores.dim() == 2:  # flatten → reshape to square grid
        side = int(patch_scores.size(1) ** 0.5)
        patch_scores = patch_scores.view(-1, side, side)

    anomaly_map = patch_scores.unsqueeze(1)  # [B,1,H',W']
    anomaly_map = F.interpolate(anomaly_map, size=image_size, mode="bilinear", align_corners=False)
    return anomaly_map.squeeze(1)
