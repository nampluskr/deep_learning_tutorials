import os
from collections.abc import Sequence
from tqdm import tqdm
from time import time
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from feature_extractor import TimmFeatureExtractor
from trainer import BaseTrainer


###########################################################
# anomalib/models/image/fre/torch_model.py
###########################################################

class TiedAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.weight = nn.Parameter(torch.empty(latent_dim, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = F.linear(features, self.weight, self.encoder_bias)
        return F.linear(encoded, self.weight.t(), self.decoder_bias)


class FRE(nn.Module):
    def __init__(
        self,
        backbone: str,
        layer: str,
        input_dim: int = 65536,
        latent_dim: int = 220,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.fre_model = TiedAE(input_dim, latent_dim)
        self.layer = layer
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer],
        ).eval()

    def get_features(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.feature_extractor.eval()
        features_in = self.feature_extractor(batch)[self.layer]
        batch_size = len(features_in)
        if self.pooling_kernel_size > 1:
            features_in = F.avg_pool2d(input=features_in, kernel_size=self.pooling_kernel_size)
        feature_shapes = features_in.shape
        features_in = features_in.view(batch_size, -1).detach()
        features_out = self.fre_model(features_in)
        return features_in, features_out, feature_shapes

    def forward(self, images):
        return self.model.get_features(images)

    def predict(self, batch: torch.Tensor):
        features_in, features_out, feature_shapes = self.get_features(batch)
        fre = torch.square(features_in - features_out).reshape(feature_shapes)
        anomaly_map = torch.sum(fre, 1)  # NxCxHxW --> NxHxW
        score = torch.sum(anomaly_map, (1, 2))  # NxHxW --> N
        anomaly_map = torch.unsqueeze(anomaly_map, 1)
        anomaly_map = F.interpolate(anomaly_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return dict(pred_score=score, anomaly_map=anomaly_map)


#############################################################
# Trainer for STFPM Model
#############################################################

class FRETrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(params=model.fre_model.parameters(), lr=1e-3)
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        super().__init__(model, optimizer, loss_fn, metrics, device)
        self.epoch_period = 5

    def train_step(self, batch):
        images = batch["image"].to(self.device)

        self.optimizer.zero_grad()
        features_in, features_out, _ = self.model.get_features(images)
        loss = self.loss_fn(features_in, features_out)
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                results[name] = metric_fn(features_in, features_out).item()
        return results
    

if __name__ == "__main__":
    model = FRE(
        backbone="resnet50",
        layer="layer3",
        input_dim=65536,
        latent_dim=220,
        pre_trained=True,
        pooling_kernel_size=4
    )
    input_tensor = torch.randn(32, 3, 256, 256)
    output = model(input_tensor)
    print(output.pred_score.shape)
    print(output.anomaly_map.shape)