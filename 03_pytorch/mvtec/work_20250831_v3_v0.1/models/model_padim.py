import torch
from torch import nn
from torch.nn import functional as F
from random import sample
from typing import NamedTuple

from .model_base import TimmFeatureExtractor, dryrun_find_featuremap_dims, MultiVariateGaussian


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


class GaussianBlur2d(nn.Module):
    """Gaussian blur module."""
    
    def __init__(self, kernel_size, sigma, channels=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        
        # Create Gaussian kernel
        self.register_buffer('kernel', self._create_gaussian_kernel())
        
    def _create_gaussian_kernel(self):
        """Create 2D Gaussian kernel."""
        kernel_size = self.kernel_size[0]
        sigma = self.sigma[0]
        
        # Create 1D Gaussian
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel
        kernel_2d = g[:, None] * g[None, :]
        kernel_2d = kernel_2d.expand(self.channels, 1, kernel_size, kernel_size)
        
        return kernel_2d
    
    def forward(self, x):
        return F.conv2d(x, self.kernel, padding='same', groups=self.channels)


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, sigma=4):
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    @staticmethod
    def compute_distance(embedding, stats):
        """Compute anomaly score for each patch position using Mahalanobis distance."""
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        return distances.clamp(0).sqrt()

    @staticmethod
    def up_sample(distance, image_size):
        """Up sample anomaly score to match the input image size."""
        return F.interpolate(
            distance,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

    def smooth_anomaly_map(self, anomaly_map):
        """Apply Gaussian smoothing to the anomaly map."""
        return self.blur(anomaly_map)

    def compute_anomaly_map(self, embedding, mean, inv_covariance, image_size=None):
        """Compute anomaly map from feature embeddings and distribution parameters."""
        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        if image_size:
            score_map = self.up_sample(score_map, image_size)
        return self.smooth_anomaly_map(score_map)

    def forward(self, **kwargs):
        """Generate anomaly map from the provided embeddings and statistics."""
        if not ("embedding" in kwargs and "mean" in kwargs and "inv_covariance" in kwargs):
            msg = f"Expected keys `embedding`, `mean` and `covariance`. Found {kwargs.keys()}"
            raise ValueError(msg)

        embedding = kwargs["embedding"]
        mean = kwargs["mean"]
        inv_covariance = kwargs["inv_covariance"]
        image_size = kwargs.get("image_size")

        return self.compute_anomaly_map(embedding, mean, inv_covariance, image_size=image_size)


# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}


def _deduce_dims(feature_extractor, input_size, layers):
    """Run a dry run to deduce the dimensions of the extracted features."""
    dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

    # the first layer in `layers` has the largest resolution
    first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
    n_patches = torch.tensor(first_layer_resolution).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)

    return n_features_original, n_patches


class PadimModel(nn.Module):
    """Padim Module."""

    def __init__(self, backbone="resnet18", layers=["layer1", "layer2", "layer3"], 
                 pre_trained=True, n_features=None):
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=layers,
            pre_trained=False,  # 변경
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
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator()

        self.gaussian = MultiVariateGaussian()
        self.memory_bank = []

    def forward(self, input_tensor):
        """Forward-pass image-batch (N, C, H, W) into model to extract features."""
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            self.memory_bank.append(embeddings)
            return embeddings

        anomaly_map = self.anomaly_map_generator(
            embedding=embeddings,
            mean=self.gaussian.mean,
            inv_covariance=self.gaussian.inv_covariance,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features):
        """Generate embedding from hierarchical feature map."""
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        return torch.index_select(embeddings, 1, idx)

    def fit(self):
        """Fits a Gaussian model to the current contents of the memory bank."""
        if len(self.memory_bank) == 0:
            msg = "Memory bank is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        self.memory_bank = torch.vstack(self.memory_bank)

        # fit gaussian
        self.gaussian.fit(self.memory_bank)

        # clear memory bank, reduces gpu usage
        self.memory_bank = []