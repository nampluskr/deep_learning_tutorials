"""PatchCore model implementation combining torch_model.py + anomaly_map.py + components."""

from collections.abc import Sequence
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .model_base import TimmFeatureExtractor


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


# =============================================================================
# Gaussian Blur Component
# =============================================================================

class GaussianBlur2d(nn.Module):
    """Gaussian Blur module for smoothing anomaly maps."""

    def __init__(self, kernel_size: tuple[int, int], sigma: tuple[float, float], channels: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        
        # Create Gaussian kernel
        self.register_buffer('kernel', self._create_gaussian_kernel())
        
    def _create_gaussian_kernel(self):
        """Create 2D Gaussian kernel."""
        kernel_size_x, kernel_size_y = self.kernel_size
        sigma_x, sigma_y = self.sigma
        
        # Create 1D Gaussian kernels for each dimension
        x = torch.arange(kernel_size_x, dtype=torch.float32) - kernel_size_x // 2
        y = torch.arange(kernel_size_y, dtype=torch.float32) - kernel_size_y // 2
        
        # Compute Gaussian values
        gauss_x = torch.exp(-0.5 * (x / sigma_x) ** 2)
        gauss_y = torch.exp(-0.5 * (y / sigma_y) ** 2)
        
        # Normalize
        gauss_x /= gauss_x.sum()
        gauss_y /= gauss_y.sum()
        
        # Create 2D kernel
        kernel_2d = gauss_x[:, None] * gauss_y[None, :]
        kernel_2d = kernel_2d.expand(self.channels, 1, kernel_size_x, kernel_size_y)
        
        return kernel_2d
    
    def forward(self, x):
        """Apply Gaussian blur to input tensor."""
        return F.conv2d(x, self.kernel, padding='same', groups=self.channels)


# =============================================================================
# Sparse Random Projection for dimensionality reduction
# =============================================================================

class SparseRandomProjection:
    """Sparse random projection for dimensionality reduction."""
    
    def __init__(self, eps=0.9):
        self.eps = eps
        self.projection_matrix = None
        self.n_components = None
    
    def fit(self, X):
        """Fit the random projection."""
        n_samples, n_features = X.shape
        self.n_components = max(1, int(n_features * (1 - self.eps)))
        
        # Create sparse random projection matrix
        self.projection_matrix = torch.randn(n_features, self.n_components, device=X.device) / (self.n_components ** 0.5)
        
    def transform(self, X):
        """Transform data using random projection."""
        if self.projection_matrix is None:
            raise ValueError("Must fit before transform")
        return torch.matmul(X, self.projection_matrix)


# =============================================================================
# K-Center Greedy Coreset Selection
# =============================================================================

class KCenterGreedy:
    """k-center-greedy method for coreset selection."""

    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances to None."""
        self.min_distances = None

    def update_distances(self, cluster_centers: list[int]) -> None:
        """Update minimum distances given cluster centers."""
        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index of the next sample based on maximum minimum distance."""
        if isinstance(self.min_distances, torch.Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)

        return idx

    def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
        """Greedily form a coreset to minimize maximum distance to cluster centers."""
        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: list[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        for _ in tqdm(range(self.coreset_size), desc="Selecting Coreset Indices"):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: list[int] | None = None) -> torch.Tensor:
        """Select coreset from the embedding."""
        idxs = self.select_coreset_idxs(selected_idxs)
        return self.embedding[idxs]


# =============================================================================
# Dynamic Buffer Mixin
# =============================================================================

class DynamicBufferMixin(nn.Module):
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


# =============================================================================
# Anomaly Map Generator
# =============================================================================

class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap for PatchCore."""

    def __init__(self, sigma: int = 4) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Compute pixel-level anomaly heatmap from patch scores."""
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]))
        return self.blur(anomaly_map)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Generate smoothed anomaly map from patch scores."""
        return self.compute_anomaly_map(patch_scores, image_size)


# =============================================================================
# Main PatchCore Model
# =============================================================================

class PatchcoreModel(DynamicBufferMixin, nn.Module):
    """PatchCore PyTorch model for anomaly detection."""

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors

        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=self.layers,
        ).eval()
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank: torch.Tensor
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.Tensor] = []

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input tensor through the model."""
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        # Ensure memory bank is not empty
        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores"
            raise ValueError(msg)

        # Apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # Reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # Compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # Reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # Get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embedding by concatenating multi-scale feature maps."""
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """Reshape embedding tensor for patch-wise processing."""
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    def subsample_embedding(self, sampling_ratio: float) -> None:
        """Subsample the memory_banks embeddings using coreset selection."""
        if len(self.embedding_store) == 0:
            msg = "Embedding store is empty. Cannot perform coreset selection."
            raise ValueError(msg)

        # Coreset Subsampling
        self.memory_bank = torch.vstack(self.embedding_store)
        self.embedding_store.clear()

        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between two sets of vectors."""
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest neighbors in memory bank for input embeddings."""
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute image-level anomaly scores."""
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper


if __name__ == "__main__":
    pass