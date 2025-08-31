"""FastFlow model implementation combining torch_model.py + loss.py + anomaly_map.py."""

from collections.abc import Callable
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
import timm

# Import FrEIA components
try:
    from FrEIA.framework import SequenceINN
    HAS_FREIA = True
except ImportError:
    HAS_FREIA = False
    print("Warning: FrEIA not available. FastFlow model will not work without FrEIA installation.")

# Import local flow components instead of anomalib
from .flow_components import AllInOneBlock

from .model_base import load_backbone_weights, get_local_weight_path


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


# =============================================================================
# Anomaly Map Generator (from anomaly_map.py)
# =============================================================================

class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmaps from FastFlow hidden variables."""

    def __init__(self, input_size: tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        """Generate anomaly heatmap from hidden variables."""
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


# =============================================================================
# Loss Function (from loss.py)
# =============================================================================

class FastflowLoss(nn.Module):
    """FastFlow Loss Module."""

    @staticmethod
    def forward(hidden_variables: list[torch.Tensor], jacobians: list[torch.Tensor]) -> torch.Tensor:
        """Calculate the FastFlow loss."""
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss


# =============================================================================
# Flow Block Creation Functions
# =============================================================================

def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """Subnet Convolutional Function."""

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        # Manual padding instead of padding="same" for ONNX compatibility
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
    """Create NF Fast Flow Block."""
    if not HAS_FREIA:
        raise ImportError("FrEIA is required for FastFlow model. Please install FrEIA.")
    
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


# =============================================================================
# Main FastFlow Model (from torch_model.py)
# =============================================================================

class FastflowModel(nn.Module):
    """FastFlow model for unsupervised anomaly detection via 2D normalizing flows."""

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
            self.feature_extractor = timm.create_model(backbone, pretrained=False)
            if pre_trained:
                load_backbone_weights(backbone, self.feature_extractor)
            channels = [768]
            scales = [16]
        elif backbone in {"resnet18", "wide_resnet50_2"}:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=False,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            if pre_trained:
                load_backbone_weights(backbone, self.feature_extractor)
            
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # For resnets, self.norms are trainable LayerNorm
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

        # Freeze feature extractor
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # Create fast flow blocks
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

    def forward(self, input_tensor: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
        """Forward-pass the input to the FastFlow Model."""
        self.feature_extractor.eval()
        
        # Get features based on backbone type
        if "vit" in str(type(self.feature_extractor)).lower() or "vision_transformer" in str(type(self.feature_extractor)).lower():
            features = self._get_vit_features(input_tensor)
        elif "cait" in str(type(self.feature_extractor)).lower():
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        # Compute hidden variables and jacobians through normalizing flows
        hidden_variables: list[torch.Tensor] = []
        log_jacobians: list[torch.Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        if self.training:
            return hidden_variables, log_jacobians

        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def _get_cnn_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Get CNN-based features."""
        features = self.feature_extractor(input_tensor)
        return [self.norms[i](feature) for i, feature in enumerate(features)]

    def _get_cait_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Get Class-Attention-Image-Transformers (CaiT) features."""
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
        """Get Vision Transformers (ViT) features."""
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


if __name__ == "__main__":
    pass