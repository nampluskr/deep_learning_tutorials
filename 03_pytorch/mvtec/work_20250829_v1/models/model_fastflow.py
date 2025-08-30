# models/model_fastflow.py
# FastFlow: Flow-based Anomaly Detection (원본 anomalib 참고)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import ResNetFeatureExtractor
import math


# -------------------------
# Flow Building Blocks
# -------------------------

class ActNorm(nn.Module):
    """Activation Normalization (Glow)"""

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.initialized = False
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean([0, 2, 3], keepdim=True)
            std = x.std([0, 2, 3], keepdim=True)
            self.bias.data.copy_(-mean)
            self.logs.data.copy_(torch.log(1 / (std + self.eps)))
        self.initialized = True

    def forward(self, x, reverse=False):
        if not self.initialized:
            self.initialize(x)

        if reverse:
            return (x - self.bias) * torch.exp(-self.logs)
        else:
            return (x + self.bias) * torch.exp(self.logs)


class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 Convolution"""

    def __init__(self, num_channels):
        super().__init__()
        w_init = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init.unsqueeze(-1).unsqueeze(-1))

    def forward(self, x, reverse=False):
        if reverse:
            weight = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        else:
            weight = self.weight
        return F.conv2d(x, weight)


class AffineCoupling(nn.Module):
    """Affine Coupling Layer"""

    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, reverse=False):
        x_a, x_b = x.chunk(2, 1)
        h = self.net(x_a)
        shift, scale = h.chunk(2, 1)
        scale = torch.sigmoid(scale + 2.0)  # stabilize

        if reverse:
            x_b = (x_b - shift) / (scale + 1e-6)
        else:
            x_b = scale * x_b + shift

        return torch.cat([x_a, x_b], dim=1)


class FlowStep(nn.Module):
    """One Step of Flow (ActNorm → 1x1Conv → Coupling)"""

    def __init__(self, in_channels):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.invconv = InvertibleConv1x1(in_channels)
        self.coupling = AffineCoupling(in_channels)

    def forward(self, x, reverse=False):
        if reverse:
            x = self.coupling(x, reverse=True)
            x = self.invconv(x, reverse=True)
            x = self.actnorm(x, reverse=True)
        else:
            x = self.actnorm(x, reverse=False)
            x = self.invconv(x, reverse=False)
            x = self.coupling(x, reverse=False)
        return x


# -------------------------
# FastFlow Model
# -------------------------

class FastflowModel(nn.Module):
    """FastFlow: Flow-based anomaly detection (anomalib 원본)"""

    def __init__(self, backbone="resnet18", flow_steps=8):
        super().__init__()
        self.backbone = ResNetFeatureExtractor(backbone, layers=["layer2", "layer3"])
        self.flow_steps = nn.ModuleList([FlowStep(128) for _ in range(flow_steps)])

    def forward(self, x):
        feats = self.backbone(x)
        f = feats[-1]  # use last feature map
        z = f
        for step in self.flow_steps:
            z = step(z)
        return z

    def compute_anomaly_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        # Negative log-likelihood (simplified): higher magnitude → more anomalous
        scores = torch.mean(outputs ** 2, dim=[1, 2, 3])
        return scores


# -------------------------
# FastFlow Loss
# -------------------------

class FastflowLoss(nn.Module):
    """NLL Loss for FastFlow"""

    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, _: torch.Tensor = None):
        loss = torch.mean(outputs ** 2)
        return loss


# -------------------------
# Anomaly Map
# -------------------------

def compute_anomaly_map(flow_outputs: torch.Tensor, out_size=(256, 256)):
    """Generate anomaly map for FastFlow"""
    anomaly_map = torch.mean(flow_outputs ** 2, dim=1, keepdim=True)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode="bilinear", align_corners=False)
    return anomaly_map
