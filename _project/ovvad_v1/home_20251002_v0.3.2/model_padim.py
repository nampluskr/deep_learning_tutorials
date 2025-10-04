from typing import Any
from random import sample
from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F

from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d

from feature_extractor import TimmFeatureExtractor
from tiler import Tiler
from trainer import BaseTrainer


#############################################################
# anomalib/src/anomalib/models/components/feature_extractors/utils.py
#############################################################

from torch.fx.graph_module import GraphModule

def dryrun_find_featuremap_dims(
    feature_extractor: GraphModule,
    input_size: tuple[int, int],
    layers: list[str],
) -> dict[str, dict[str, int | tuple[int, int]]]:
    device = next(feature_extractor.parameters()).device
    dryrun_input = torch.empty(1, 3, *input_size).to(device)
    was_training = feature_extractor.training
    feature_extractor.eval()
    with torch.no_grad():
        dryrun_features = feature_extractor(dryrun_input)
    if was_training:
        feature_extractor.train()
    return {
        layer: {
            "num_features": dryrun_features[layer].shape[1],
            "resolution": dryrun_features[layer].shape[2:],
        }
        for layer in layers
    }


#############################################################
# anomalib/src/anomalib/models/components/base/dynamic_buffer.py
#############################################################

class DynamicBufferMixin(nn.Module, ABC):
    def get_tensor_attribute(self, attribute_name: str) -> torch.Tensor:
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, torch.Tensor):
            return attribute

        msg = f"Attribute with name '{attribute_name}' is not a torch Tensor"
        raise ValueError(msg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args) -> None:
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


#############################################################
# anomalib/src/anomalib/models/components/stats/multi_variate_gaussian.py
#############################################################

class MultiVariateGaussian(DynamicBufferMixin, nn.Module):
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
        return self.forward(embedding)


#############################################################
# anomalib/src/anomalib/models/components/filters/blur.py
#############################################################

def compute_kernel_size(sigma_val: float) -> int:
    return 2 * int(4.0 * sigma_val + 0.5) + 1


class GaussianBlur2d(nn.Module):
    def __init__(
        self,
        sigma: float | tuple[float, float],
        channels: int = 1,
        kernel_size: int | tuple[int, int] | None = None,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        super().__init__()
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.channels = channels

        if kernel_size is None:
            kernel_size = (compute_kernel_size(sigma[0]), compute_kernel_size(sigma[1]))
        else:
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.kernel: torch.Tensor
        self.register_buffer("kernel", get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)

        self.kernel = self.kernel.view(1, 1, *self.kernel.shape[-2:])

        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        # convolve the tensor with the kernel.
        output = F.conv2d(input_tensor, self.kernel, groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out


###########################################################
# anomalib\models\images\padim\anomaly_map.py
###########################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(self, sigma: int = 4) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    @staticmethod
    def compute_distance(embedding: torch.Tensor, stats: list[torch.Tensor]) -> torch.Tensor:
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        return distances.clamp(0).sqrt()

    @staticmethod
    def up_sample(distance: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        return F.interpolate(
            distance,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

    def smooth_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return self.blur(anomaly_map)

    def compute_anomaly_map(
        self,
        embedding: torch.Tensor,
        mean: torch.Tensor,
        inv_covariance: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        if image_size:
            score_map = self.up_sample(score_map, image_size)
        return self.smooth_anomaly_map(score_map)

    def forward(self, **kwargs) -> torch.Tensor:
        if not ("embedding" in kwargs and "mean" in kwargs and "inv_covariance" in kwargs):
            msg = f"Expected keys `embedding`, `mean` and `covariance`. Found {kwargs.keys()}"
            raise ValueError(msg)

        embedding: torch.Tensor = kwargs["embedding"]
        mean: torch.Tensor = kwargs["mean"]
        inv_covariance: torch.Tensor = kwargs["inv_covariance"]
        image_size: tuple[int, int] | torch.Size = kwargs.get("image_size")

        return self.compute_anomaly_map(embedding, mean, inv_covariance, image_size=image_size)


###########################################################
# anomalib\models\images\padim\torch_model.py
###########################################################

# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}


def _deduce_dims(
    feature_extractor: TimmFeatureExtractor,
    input_size: tuple[int, int],
    layers: list[str],
) -> tuple[int, int]:
    dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

    # the first layer in `layers` has the largest resolution
    first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
    n_patches = torch.tensor(first_layer_resolution).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore[misc]

    return n_features_original, n_patches


class PaDim(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=layers,
            pre_trained=pre_trained,
        ).eval()
        self.n_features_original = sum(self.feature_extractor.out_dims)
        self.n_features = n_features or _N_FEATURES_DEFAULTS.get(self.backbone)
        if self.n_features is None:
            msg = (
                f"n_features must be specified for backbone {self.backbone}. "
                f"Default values are available for: {sorted(_N_FEATURES_DEFAULTS.keys())}"
            )
            raise ValueError(msg)

        if not (0 < self.n_features <= self.n_features_original):
            msg = f"For backbone {self.backbone}, 0 < n_features <= {self.n_features_original}, found {self.n_features}"
            raise ValueError(msg)

        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(self.n_features_original), self.n_features)),
        )
        self.idx: torch.Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator()

        self.gaussian = MultiVariateGaussian()
        self.memory_bank: list[torch.tensor] = []

    def forward(self, input_tensor: torch.Tensor):
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        self.memory_bank.append(embeddings)
        return embeddings

    def predict(self, input_tensor: torch.Tensor):
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        anomaly_map = self.anomaly_map_generator(
            embedding=embeddings,
            mean=self.gaussian.mean,
            inv_covariance=self.gaussian.inv_covariance,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        return torch.index_select(embeddings, 1, idx)

    def fit(self) -> None:
        if len(self.memory_bank) == 0:
            msg = "Memory bank is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        self.memory_bank = torch.vstack(self.memory_bank)

        # fit gaussian
        self.gaussian.fit(self.memory_bank)

        # clear memory bank, redcues gpu usage
        self.memory_bank = []


#############################################################
# Trainer for PaDiM Model
#############################################################

from tqdm import tqdm

class PaDimTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        super().__init__(model, optimizer, loss_fn, metrics, device)
        self.epoch_period = 1

    def on_train_start(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader, desc="Extracting embeddings", leave=False, ascii=True)
        for batch in pbar:
            images = batch["image"].to(self.device)
            _ = self.model(images)

    def on_train_end(self, train_results):
        self.model.fit()
        super().on_train_end(train_results)

    def train_step(self, batch):
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def train_epoch(self, train_loader):
        return {"loss": 0.0}

if __name__ == "__main__":
    pass