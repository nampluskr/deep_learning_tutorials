
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections.abc import Sequence
import timm
import os
from abc import ABC, abstractmethod
from typing import Any

try:
    from torchvision.models.feature_extraction import create_feature_extractor
    HAS_TORCHVISION_FEATURE_EXTRACTION = True
except ImportError:
    HAS_TORCHVISION_FEATURE_EXTRACTION = False


logger = logging.getLogger(__name__)

BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
    "vgg16": "vgg16-397923af.pth",
    "alexnet": "alexnet-owt-7be5be79.pth",
    "squeezenet1_1": "squeezenet1_1-b8a52dc0.pth",
    # EfficientAD weights
    "efficientad_teacher_small": "pretrained_teacher_small.pth",
    "efficientad_teacher_medium": "pretrained_teacher_medium.pth",
    # LPIPS weights
    "lpips_alex": "lpips_alex.pth",
    "lpips_vgg": "lpips_vgg.pth",
    "lpips_squeeze": "lpips_squeeze.pth"
}


def set_backbone_dir(path):
    """Set global backbone directory"""
    global BACKBONE_DIR
    BACKBONE_DIR = path


def get_backbone_path(backbone):
    """Get local weight path"""
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


def check_backbone_files(backbone_dir=None):
    """Check if backbone files exist in the directory"""
    if backbone_dir is None:
        backbone_dir = BACKBONE_DIR

    print(f" > Checking backbone directory: {backbone_dir}")
    if not os.path.exists(backbone_dir):
        print(f" > Warning: Backbone directory not found: {backbone_dir}")
        print(f" > Continuing with random initialization...")
        return False

    required_files = list(BACKBONE_WEIGHT_FILES.values())
    missing_files = []
    for file in required_files:
        full_path = os.path.join(backbone_dir, file)
        if not os.path.exists(full_path):
            missing_files.append(file)

    if missing_files:
        print(f" > Warning: Missing backbone files: {missing_files}")
        print(f" > Continuing with random initialization...")
        return False

    print(f" > All backbone weights verified in: {backbone_dir}")
    return True

# ===================================================================
# Convolutional Blocks
# ===================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv_block(x)


# ===================================================================
# Feature Extractor
# ===================================================================


class TimmFeatureExtractor(nn.Module):
    def __init__(self, backbone, layers, pre_trained=True, requires_grad=False):
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad

        if isinstance(backbone, nn.Module):
            if not HAS_TORCHVISION_FEATURE_EXTRACTION:
                raise ImportError(
                    "torchvision.models.feature_extraction is required for nn.Module backbones. "
                    "Please update torchvision or use timm backbone strings instead."
                )
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in self.layers},
            )
            layer_metadata = dryrun_find_featuremap_dims(self.feature_extractor, (256, 256), layers=self.layers)
            self.out_dims = [feature_info["num_features"] for layer_name, feature_info in layer_metadata.items()]

        elif isinstance(backbone, str):
            # 먼저 idx 계산
            self.idx = self._map_layer_to_idx()

            # 로컬 weight 경로
            local_weights_path = get_backbone_path(backbone)

            # 모델 생성 (항상 pretrained=False)
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=False,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )

            # 로컬 weight 로딩
            if os.path.exists(local_weights_path):
                logger.info(f"Loading local weights from {local_weights_path}")
                try:
                    state_dict = torch.load(local_weights_path, map_location='cpu')
                    self.feature_extractor.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded local weights for {backbone}")
                except Exception as e:
                    logger.warning(f"Failed to load local weights: {e}. Using random initialization.")
            else:
                logger.warning(f"Local weights not found at {local_weights_path}")
                if pre_trained:
                    logger.warning("Using random initialization instead of pretrained weights.")

            self.out_dims = self.feature_extractor.feature_info.channels()

        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self):
        """Map layer names to indices."""
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:
                msg = f"Layer {layer} not found in model {self.backbone}. Available layers: {layer_names}"
                logger.warning(msg)
                self.layers.remove(layer)

        return idx

    def forward(self, inputs):
        if self.requires_grad:
            features = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(inputs)
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))
        return features


def dryrun_find_featuremap_dims(feature_extractor, input_size, layers):
    device = next(feature_extractor.parameters()).device
    dryrun_input = torch.empty(1, 3, *input_size).to(device)
    dryrun_features = feature_extractor(dryrun_input)
    return {
        layer: {
            "num_features": dryrun_features[layer].shape[1],
            "resolution": dryrun_features[layer].shape[2:],
        }
        for layer in layers
    }


def load_backbone_weights(model_name, model):
    weights_path = get_backbone_path(model_name)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded local weights from {weights_path}")
    else:
        logger.warning(f"Local weights not found at {weights_path}")


def get_feature_extractor(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True, requires_grad=False):
    return TimmFeatureExtractor(
        backbone=backbone,
        layers=layers,
        pre_trained=pre_trained,
        requires_grad=requires_grad
    )


class DynamicBufferMixin(nn.Module, ABC):
    """Mixin that enables loading state dicts with mismatched tensor shapes."""

    def get_tensor_attribute(self, attribute_name: str) -> torch.Tensor:
        """Get a tensor attribute by name."""
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, torch.Tensor):
            return attribute
        msg = f"Attribute with name '{attribute_name}' is not a torch Tensor"
        raise ValueError(msg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args) -> None:
        """Load a state dictionary, resizing buffers if shapes don't match."""
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}
        for param in local_buffers:
            for key in state_dict:
                if (
                    key.startswith(prefix)
                    and key[len(prefix) :].split(".")[0] == param
                    and local_buffers[param].shape != state_dict[key].shape
                ):
                    attribute = self.get_tensor_attribute(param)
                    attribute.resize_(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args)


class MultiVariateGaussian(DynamicBufferMixin, nn.Module):
    """Multi Variate Gaussian Distribution."""

    def __init__(self) -> None:
        """Initialize empty buffers for mean and inverse covariance."""
        super().__init__()

        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_covariance", torch.empty(0))

        self.mean: torch.Tensor
        self.inv_covariance: torch.Tensor

    @staticmethod
    def _cov(
        observations: torch.Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: int | None = None,
        aweights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate covariance matrix similar to numpy.cov."""
        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            ddof = 1 if bias == 0 else 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        x_transposed = observations_m.t() if weights is None else torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Calculate multivariate Gaussian distribution parameters."""
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        # Stabilize the covariance matrix by adding a small regularization term
        stabilized_covariance = covariance.permute(2, 0, 1) + 1e-5 * identity

        # Check if the device is MPS and fallback to CPU if necessary
        if device.type == "mps":
            # Move stabilized covariance to CPU for inversion
            self.inv_covariance = torch.linalg.inv(stabilized_covariance.cpu()).to(device)
        else:
            # Calculate inverse covariance as we need only the inverse
            self.inv_covariance = torch.linalg.inv(stabilized_covariance)

        return [self.mean, self.inv_covariance]

    def fit(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Fit multivariate Gaussian distribution to input embeddings."""
        return self.forward(embedding)


if __name__ == "__main__":
    pass
