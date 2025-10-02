import os
from collections.abc import Sequence
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# from ssim import ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchmetrics
from trainer import BaseTrainer


###########################################################
# Conv Block / Deconv Block / Autoencoder
###########################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.deconv_block(x)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        feat_size = img_size // 8
        self.to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * feat_size * feat_size, latent_dim),
        )
        self.from_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * feat_size * feat_size),
            nn.Unflatten(dim=1, unflattened_size=(256, feat_size, feat_size)),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, x):
        latent = self.encoder(x)
        latent = self.to_linear(latent)
        latent = self.from_linear(latent)
        recon = self.decoder(latent)
        return recon, latent

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        recon, *_ = self.forward(images)
        anomaly_map = torch.mean((images - recon)**2, dim=1, keepdim=True)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


############################################################
# Loss Functions and Metrics
############################################################

class AELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon, original):
        return F.mse_loss(recon, original, reduction=self.reduction)


class AECombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, ssim_weight=0.3, reduction="mean", data_range: float = 2.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.reduction = reduction
        self.data_range = data_range

        torchmetrics_reduction = {
            "mean": "elementwise_mean",
            "sum": "sum",
            "none": "none",
        }.get(reduction, "elementwise_mean")

        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=self.data_range,
            reduction=torchmetrics_reduction,
        )

    def forward(self, recon, original):
        mse_loss = F.mse_loss(recon, original, reduction=self.reduction)
        ssim_loss = 1.0 - self.ssim_metric(recon, original)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


class SSIMMetric(nn.Module):
    def __init__(self, data_range=2.0, reduction="mean"):
        super().__init__()
        torchmetrics_reduction = {
            "mean": "elementwise_mean",
            "sum": "sum",
            "none": "none",
        }.get(reduction, "elementwise_mean")

        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            reduction=torchmetrics_reduction,
        )

    def forward(self, preds, targets):
        return self.ssim_metric(preds, targets)


# class LPIPSMetric(nn.Module):
#     def __init__(self, reduction="mean", net_type="alex"):
#         super().__init__()
#         torchmetrics_reduction = {
#             "mean": "elementwise_mean",
#             "sum": "sum",
#             "none": "none",
#         }.get(reduction, "elementwise_mean")

#         self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
#             net_type=net_type,
#             model_path=os.path.join("/home/namu/myspace/NAMU/project_2025/backbones",
#                 f"{net_type}_lpips.pth"),
#             pretrained=True,
#             reduction=torchmetrics_reduction,
#         )

#     def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         return self.lpips_metric(preds, targets)

#############################################################
# Trainer for AutoEncoder Model
#############################################################

class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
        if loss_fn is None:
            loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3)
        if metrics is None:
            metrics = {"ssim": SSIMMetric(data_range=2.0, reduction="elementwise_mean")}

        super().__init__(model, optimizer, loss_fn, metrics, device)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        self.optimizer.zero_grad()
        recon, latent = self.model(images)
        loss = self.loss_fn(recon, images)
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                results[name] = metric_fn(images, recon).item()
        return results
