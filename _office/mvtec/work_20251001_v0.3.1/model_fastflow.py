from collections.abc import Callable

import timm
from FrEIA.framework import SequenceINN
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer

import torch
from torch import nn
from torch.nn import functional as F

from feature_extractor import get_local_weight_path
from all_in_one_block import AllInOneBlock
from trainer import BaseTrainer


###########################################################
# anomalib\models\images\fastflow\loss.py
###########################################################

class FastflowLoss(nn.Module):
    @staticmethod
    def forward(hidden_variables: list[torch.Tensor], jacobians: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss


###########################################################
# anomalib\models\images\fastflow\anomaly_map.py
###########################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        flow_maps: list[torch.Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        return torch.mean(flow_maps, dim=-1)


###########################################################
# anomalib\models\images\fastflow\torch_model.py
###########################################################

def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        # NOTE: setting padding="same" in nn.Conv2d breaks the onnx export so manual padding required.
        # TODO(ashwinvaidya17): Use padding="same" in nn.Conv2d once PyTorch v2.1 is released
        # CVS-122671
        padding_dims = (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
        padding = (*padding_dims, *padding_dims)
        return nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.ReLU(),
            nn.ZeroPad2d(padding),
            nn.Conv2d(hidden_channels, out_channels, kernel_size),
        )
    return subnet_conv


def create_fast_flow_block(
    input_dimensions: list[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:

    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        if backbone in {"cait_m48_448", "deit_base_distilled_patch16_384"}:
            self.feature_extractor = timm.create_model(backbone, pretrained=Fasle)
            channels = [768]
            scales = [16]

        elif backbone in {"resnet18", "wide_resnet50_2"}:
            self.feature_extractor = timm.create_model(backbone, pretrained=False,
                features_only=True, out_indices=[1, 2, 3])
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales, strict=True):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    ),
                )
        else:
            msg = (
                f"Backbone {backbone} is not supported. List of available backbones are "
                "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
            )
            raise ValueError(msg)

        if pre_trained:
            weights_path = get_local_weight_path(backbone)
            state_dict = torch.load(weights_path, map_location='cpu')
            self.feature_extractor.load_state_dict(state_dict, strict=False)

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.fast_flow_blocks = nn.ModuleList()
        for channel, scale in zip(channels, scales, strict=True):
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                ),
            )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def forward(self, input_tensor: torch.Tensor):
        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        hidden_variables: list[torch.Tensor] = []
        log_jacobians: list[torch.Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        return hidden_variables, log_jacobians

    def predict(self, input_tensor: torch.Tensor):
        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        hidden_variables: list[torch.Tensor] = []
        log_jacobians: list[torch.Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def predict(self, x: torch.Tensor):
        hidden_variables, _ = self.forward(x)
        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return {"pred_score": pred_score, "anomaly_map": anomaly_map}

    def _get_cnn_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        features = self.feature_extractor(input_tensor)
        return [self.norms[i](feature) for i, feature in enumerate(features)]

    def _get_cait_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)
        for i in range(41):  # paper Table 6. Block Index = 40
            feature = self.feature_extractor.blocks[i](feature)
        batch_size, _, num_channels = feature.shape
        feature = self.feature_extractor.norm(feature)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        return [feature]

    def _get_vit_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        feature = self.feature_extractor.patch_embed(input_tensor)
        cls_token = self.feature_extractor.cls_token.expand(feature.shape[0], -1, -1)
        if self.feature_extractor.dist_token is None:
            feature = torch.cat((cls_token, feature), dim=1)
        else:
            feature = torch.cat(
                (
                    cls_token,
                    self.feature_extractor.dist_token.expand(feature.shape[0], -1, -1),
                    feature,
                ),
                dim=1,
            )
        feature = self.feature_extractor.pos_drop(feature + self.feature_extractor.pos_embed)
        for i in range(8):  # paper Table 6. Block Index = 7
            feature = self.feature_extractor.blocks[i](feature)
        feature = self.feature_extractor.norm(feature)
        feature = feature[:, 2:, :]
        batch_size, _, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        return [feature]


#############################################################
# Trainer for FastFlow Model
#############################################################

import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FastFlowTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            params=self._get_trainable_params(),
            lr=1e-4,
            weight_decay=0.0
        )
        self.loss_fn = loss_fn or FastflowLoss()
        self.metrics = metrics or {}

    def _get_trainable_params(self):
        params = []
        for block in self.model.fast_flow_blocks:
            params += list(block.parameters())
        return params

    def _training_step(self, batch):
        self.optimizer.zero_grad()
        images = batch["image"].to(self.device)
        hidden_vars, log_jacobians = self.model(images)
        loss = self.loss_fn(hidden_vars, log_jacobians)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validation_step(self, batch):
        images = batch["image"].to(self.device)
        with torch.no_grad():
            predictions = self.model.predict(images)
        return predictions

    def run_epoch(self, loader, mode='train', desc=""):
        if mode == 'train':
            self.model.feature_extractor.eval()
            self.model.fast_flow_blocks.train()
        else:
            self.model.eval()

        stats = {"loss": 0.0}
        n_samples = 0
        predictions = []

        pbar = tqdm(loader, desc=desc, leave=False, ascii=True)
        for batch in pbar:
            images = batch["image"].to(self.device)
            batch_size = images.size(0)
            n_samples += batch_size

            if mode == 'train':
                loss = self._training_step(batch)
                stats["loss"] += loss * batch_size
            else:
                pred = self._validation_step(batch)
                predictions.append(pred)

            pbar.set_postfix({"loss": f"{stats['loss']/n_samples:.4f}"})

        avg_stats = {k: v / n_samples for k, v in stats.items()}
        return avg_stats, predictions

    def fit(self, train_loader, num_epochs, valid_loader=None, weight_path=None):
        print("\n > Start FastFlow training...")
        history = {"loss": []}
        if valid_loader is not None:
            history["val_loss"] = []

        train_start = time.time()
        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_metrics, _ = self.run_epoch(train_loader, mode='train', desc=f"Train [{epoch}/{num_epochs}]")
            history["loss"].append(train_metrics["loss"])

            val_metrics = {"loss": 0.0}
            if valid_loader is not None:
                val_metrics, _ = self.run_epoch(valid_loader, mode='valid', desc=f"Valid [{epoch}/{num_epochs}]")
                history["val_loss"].append(val_metrics["loss"])
                current_loss = val_metrics["loss"]
                if current_loss < best_loss:
                    best_loss = current_loss
                    if weight_path:
                        self.save_model(weight_path.replace(".pth", "_best.pth"))
            else:
                current_loss = train_metrics["loss"]

            print(f" [{epoch:3d}/{num_epochs}] loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f} ({time.time() - epoch_start:.1f}s)")

        total_time = time.time() - train_start
        h, r = divmod(int(total_time), 3600)
        m, s = divmod(r, 60)
        print(f" > Training finished in {h:02d}:{m:02d}:{s:02d}")
        if weight_path:
            self.save_model(weight_path)
        return history

    def save_model(self, weight_path):
        if weight_path is None:
            return
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        ckpt = {
            "fast_flow_blocks_state_dict": [block.state_dict() for block in self.model.fast_flow_blocks],
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(ckpt, weight_path)
        print(f" > Model saved to: {weight_path}")

    def load_model(self, weight_path):
        if not os.path.isfile(weight_path):
            print(f" > No checkpoint found at: {weight_path}")
            return
        ckpt = torch.load(weight_path, map_location=self.device)
        if "fast_flow_blocks_state_dict" in ckpt:
            for block, state in zip(self.model.fast_flow_blocks, ckpt["fast_flow_blocks_state_dict"]):
                block.load_state_dict(state)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f" > Model loaded from: {weight_path}")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model = FastFlow(input_size=(256, 256), backbone="wide_resnet50_2").to(device)
    x = torch.randn(16, 3, 256, 256).to(device)

    outputs = model(x)

    predictions = model.predict(x)
    print()
    print(f"pred_socre:  {predictions['pred_score'].shape}")
    print(f"anoamly_map: {predictions['anomaly_map'].shape}")
