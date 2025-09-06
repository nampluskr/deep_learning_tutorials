import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from typing import List, Tuple
import logging

import timm
from FrEIA.framework import SequenceINN
from FrEIA.modules import InvertibleModule
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer
from scipy.stats import special_ortho_group


# ===================================================================
# AllInOneBlock from anomalib.models.components.flow
# ===================================================================

def _global_scale_sigmoid_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid activation for global scaling."""
    return 10 * torch.sigmoid(input_tensor - 2.0)


def _global_scale_softplus_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply softplus activation for global scaling."""
    softplus = nn.Softplus(beta=0.5)
    return 0.1 * softplus(input_tensor)


def _global_scale_exp_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply exponential activation for global scaling."""
    return torch.exp(input_tensor)


class AllInOneBlock(InvertibleModule):
    """Module combining common operations in normalizing flows.

    This block combines affine coupling, permutation, and global affine
    transformation ('ActNorm'). It supports GIN coupling blocks, learned
    householder permutations, inverted pre-permutation, and soft clamping
    mechanism from Real-NVP.
    """

    def __init__(
        self,
        dims_in: List[Tuple[int]],
        dims_c: List[Tuple[int]] = None,
        subnet_constructor: Callable = None,
        affine_clamping: float = 2.0,
        gin_block: bool = False,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
        permute_soft: bool = False,
        learned_householder_permutation: int = 0,
        reverse_permutation: bool = False,
    ):
        if dims_c is None:
            dims_c = []
        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        self.input_rank = len(dims_in[0]) - 1
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            if tuple(dims_c[0][1:]) != tuple(dims_in[0][1:]):
                raise ValueError(f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}")
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            self.permute_function = {0: F.linear, 1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and channels > 512:
            logger.warning(f"Soft permutation will take a very long time with {channels} channels")

        # Global scale initialization
        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - torch.log(torch.tensor([10.0 / global_affine_init - 1.0]))
            self.global_scale_activation = _global_scale_sigmoid_activation
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * torch.log(torch.exp(torch.tensor(0.5 * 10.0 * global_affine_init)) - 1)
            self.global_scale_activation = _global_scale_softplus_activation
        elif global_affine_type == "EXP":
            global_scale = torch.log(torch.tensor(global_affine_init))
            self.global_scale_activation = _global_scale_exp_activation
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.input_rank)) * global_scale)
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            indices = torch.randperm(channels)
            w = torch.zeros((channels, channels))
            w[torch.arange(channels), indices] = 1.0

        if self.householder:
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(
                torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
                requires_grad=False,
            )
            self.w_perm_inv = nn.Parameter(
                torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                requires_grad=False,
            )

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor function")

        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def _construct_householder_permutation(self) -> torch.Tensor:
        """Compute permutation matrix from learned reflection vectors."""
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for _ in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x: torch.Tensor, rev: bool = False):
        """Perform permutation and scaling after coupling operation."""
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale, perm_log_jac)
        return (self.permute_function(x * scale + self.global_offset, self.w_perm), perm_log_jac)

    def _pre_permute(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        """Permute before coupling block."""
        if rev:
            return self.permute_function(x, self.w_perm)
        return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x: torch.Tensor, a: torch.Tensor, rev: bool = False):
        """Perform affine coupling operation."""
        a *= 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * torch.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:], torch.sum(sub_jac, dim=self.sum_dims))
        return ((x - a[:, ch:]) * torch.exp(-sub_jac), -torch.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x: torch.Tensor, c: List = None, rev: bool = False, jac: bool = True):
        """Forward pass through the invertible block."""
        if c is None:
            c = []

        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = torch.split(x[0], self.splits, dim=1)
        x1c = torch.cat([x1, *c], 1) if self.conditional else x1

        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1) ** rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    @staticmethod
    def output_dims(input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        """Get output dimensions of the layer."""
        return input_dims


# ===================================================================
# Subnet Functions for FastFlow
# ===================================================================

def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """Subnet Convolutional Function for FastFlow normalizing flows.

    Args:
        kernel_size: Convolution kernel size
        hidden_ratio: Hidden ratio to compute number of hidden channels

    Returns:
        Callable: Sequential for the subnet constructor
    """
    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
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
    input_dimensions: List[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    """Create NF Fast Flow Block based on Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions: Input dimensions (Channel, Height, Width)
        conv3x3_only: Boolean whether to use conv3x3 only or conv3x3 and conv1x1
        hidden_ratio: Ratio for the hidden layer channels
        flow_steps: Flow steps
        clamp: Clamp value

    Returns:
        SequenceINN: FastFlow Block
    """
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


# ===================================================================
# FastFlow Anomaly Map Generator
# ===================================================================

class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmaps from FastFlow hidden variables."""

    def __init__(self, input_size: Tuple[int, int]):
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: List[torch.Tensor]) -> torch.Tensor:
        """Generate anomaly heatmap from hidden variables.

        Args:
            hidden_variables: List of hidden variables from each NF FastFlow block

        Returns:
            torch.Tensor: Anomaly heatmap with shape (N, 1, H, W)
        """
        flow_maps = []
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


# ===================================================================
# FastFlow Model
# ===================================================================

class FastFlowModel(nn.Module):
    """FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        backbone: str = "resnet18",
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.backbone = backbone

        # Initialize feature extractor based on backbone type
        if backbone in {"cait_m48_448", "deit_base_distilled_patch16_384"}:
            self.feature_extractor = timm.create_model(backbone, pretrained=pre_trained)
            channels = [768]
            scales = [16]
        elif backbone in {"resnet18", "wide_resnet50_2"}:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                out_indices=[1, 2, 3],
            )
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
            supported_backbones = ["cait_m48_448", "deit_base_distilled_patch16_384", "resnet18", "wide_resnet50_2"]
            raise ValueError(f"Backbone {backbone} is not supported. Available: {supported_backbones}")

        # Freeze feature extractor parameters
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # Create FastFlow blocks for each scale
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

        # Anomaly map generator
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def compute_anomaly_map(self, hidden_variables: List[torch.Tensor]) -> torch.Tensor:
        """FastFlow-specific: Generate anomaly map from flow hidden variables.

        Args:
            hidden_variables: List of hidden variables from normalizing flow blocks

        Returns:
            torch.Tensor: Anomaly map with shape (B, 1, H, W)
        """
        return self.anomaly_map_generator(hidden_variables)

    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """FastFlow-specific: Max pooling for image-level anomaly score.

        Args:
            anomaly_map: Pixel-level anomaly map (B, 1, H, W)

        Returns:
            torch.Tensor: Image-level anomaly scores (B,)
        """
        return torch.amax(anomaly_map, dim=(-2, -1))

    def forward(self, images: torch.Tensor):
        """Forward pass for FastFlow model.

        Args:
            images: Input images

        Returns:
            Training mode: (hidden_variables, log_jacobians)
            Inference mode: {'pred_score': tensor, 'anomaly_map': tensor}
        """
        # Always set feature extractor to eval mode
        self.feature_extractor.eval()

        # Extract features based on backbone type
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(images)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(images)
        else:
            features = self._get_cnn_features(images)

        # Compute hidden variables and log jacobians for each FastFlow block
        hidden_variables = []
        log_jacobians = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        if self.training:
            # Training mode: return raw outputs for loss calculation
            return hidden_variables, log_jacobians
        else:
            # Inference mode: compute and return standard anomaly outputs
            anomaly_map = self.compute_anomaly_map(hidden_variables)
            pred_score = self.compute_anomaly_score(anomaly_map)
            return {'pred_score': pred_score, 'anomaly_map': anomaly_map}

    def _get_cnn_features(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extract CNN-based features (ResNet, etc.).

        Args:
            input_tensor: Input tensor

        Returns:
            List[torch.Tensor]: List of normalized features
        """
        features = self.feature_extractor(input_tensor)
        return [self.norms[i](feature) for i, feature in enumerate(features)]

    def _get_cait_features(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extract Class-Attention-Image-Transformers (CaiT) features.

        Args:
            input_tensor: Input tensor

        Returns:
            List[torch.Tensor]: List of features
        """
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)

        # Use block index 40 as per paper Table 6
        for i in range(41):
            feature = self.feature_extractor.blocks[i](feature)

        batch_size, _, num_channels = feature.shape
        feature = self.feature_extractor.norm(feature)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        return [feature]

    def _get_vit_features(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extract Vision Transformers (ViT) features.

        Args:
            input_tensor: Input tensor

        Returns:
            List[torch.Tensor]: List of features
        """
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

        # Use block index 7 as per paper Table 6
        for i in range(8):
            feature = self.feature_extractor.blocks[i](feature)

        feature = self.feature_extractor.norm(feature)
        feature = feature[:, 2:, :]
        batch_size, _, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        return [feature]


# ===================================================================
# FastFlow Loss
# ===================================================================

class FastFlowLoss(nn.Module):
    """FastFlow Loss Module for negative log-likelihood computation."""

    def __init__(self):
        super().__init__()

    def forward(self, hidden_variables: List[torch.Tensor], jacobians: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the FastFlow loss.

        Args:
            hidden_variables: List of hidden variable tensors from normalizing flows
            jacobians: List of log determinants of Jacobian matrices

        Returns:
            torch.Tensor: Scalar loss value (negative log-likelihood)
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss


if __name__ == "__main__":
    # Test FastFlow model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test model creation
    model = FastFlowModel(
        input_size=(256, 256),
        backbone="resnet18",
        flow_steps=4,  # Reduced for testing
        hidden_ratio=1.0
    ).to(device)

    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)

    # Training mode test
    model.train()
    hidden_vars, jacobians = model(x)
    print(f"Training mode - Hidden vars: {len(hidden_vars)}, Jacobians: {len(jacobians)}")

    # Test loss
    loss_fn = FastFlowLoss()
    loss = loss_fn(hidden_vars, jacobians)
    print(f"Training loss: {loss.item():.4f}")

    # Inference mode test
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Inference mode - Pred score: {output['pred_score'].shape}, Anomaly map: {output['anomaly_map'].shape}")

    print("FastFlow model test completed successfully!")