from abc import ABC
from typing import Dict, Any
from collections.abc import Sequence
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from sklearn.utils.random import sample_without_replacement
from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d

from feature_extractor import TimmFeatureExtractor
from tiler import Tiler
from trainer import BaseTrainer


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


class NotFittedError(ValueError, AttributeError):
    """Exception raised when model is used before fitting."""


class SparseRandomProjection:
    def __init__(self, eps: float = 0.1, random_state: int | None = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: torch.Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int) -> torch.Tensor:
        # Density 'auto'. Factorize density
        density = 1 / np.sqrt(n_features)

        if density == 1:
            # skip index generation if totally dense
            binomial = torch.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((self.n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(self.n_components) * components
        else:
            # Sparse matrix is not being generated here as it is stored as dense anyways
            components = torch.zeros((self.n_components, n_features), dtype=torch.float32)
            for i in range(self.n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = torch.distributions.Binomial(total_count=n_features, probs=density).sample()
                # get nnz_idx column indices
                # pylint: disable=not-callable
                c_idx = torch.tensor(
                    sample_without_replacement(
                        n_population=n_features,
                        n_samples=nnz_idx,
                        random_state=self.random_state,
                    ),
                    dtype=torch.int32,
                )
                data = torch.distributions.Binomial(total_count=1, probs=0.5).sample(sample_shape=c_idx.size()) * 2 - 1
                # assign data to only those columns
                components[i, c_idx] = data

            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)

        return components

    @staticmethod
    def _johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int | np.integer:
        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: torch.Tensor) -> "SparseRandomProjection":
        n_samples, n_features = embedding.shape
        device = embedding.device

        self.n_components = self._johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)

        # Generate projection matrix
        # torch can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix on the device
        self.sparse_random_matrix = self._sparse_random_matrix(n_features=n_features).to(device)

        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.sparse_random_matrix is None:
            msg = "`fit()` has not been called on SparseRandomProjection yet."
            raise NotFittedError(msg)

        return embedding @ self.sparse_random_matrix.T.float()


class KCenterGreedy:
    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        self.min_distances = None

    def update_distances(self, cluster_centers: list[int]) -> None:
        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        if isinstance(self.min_distances, torch.Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)

        return idx

    def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
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
        for _ in tqdm(range(self.coreset_size), desc="Selecting Coreset Indices."):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: list[int] | None = None) -> torch.Tensor:
        idxs = self.select_coreset_idxs(selected_idxs)
        return self.embedding[idxs]


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


class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
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
        return self.compute_anomaly_map(patch_scores, image_size)


class PatchCore(DynamicBufferMixin, nn.Module):
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
        self.embedding_store: list[torch.tensor] = []

    def forward(self, input_tensor):
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
        self.embedding_store.append(embedding)
        return embedding

    def predict(self, input_tensor):
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

        # Ensure memory bank is not empty
        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores"
            raise ValueError(msg)

        # apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    # @deprecate(args={"embeddings": None}, since="2.1.0", reason="Use the default memory bank instead.")
    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        if embeddings is not None:
            del embeddings

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
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
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


class PatchCoreTrainer(BaseTrainer):
    def __init__(self,
                 model: PatchCore,
                 cfg: Any,
                 device: torch.device | None = None):
        super().__init__()          # 인자를 넘기지 않음
        self.cfg = cfg              # 사용자가 전달한 config 저장
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델을 지정된 device에 올리고, trainer 내부에 보관
        self.model = model.to(self.device)

        # 코어셋 비율 (예: 0.1 → 전체 임베딩의 10%만 메모리‑뱅크에 저장)
        self.coreset_sampling_ratio = 0.1
        # self.coreset_sampling_ratio = cfg.coreset_sampling_ratio

        # 메모리‑뱅크가 만들어졌는지 여부
        self.memory_built = False
        self.loss_fn = None
        self.metrics = None or {}

    def run_epoch(self,
                  loader,
                  epoch: int = 0,
                  num_epochs: int = 0,
                  mode: str = "train",
                  desc: str = "") -> Dict[str, float]:

        pbar = tqdm(loader, desc=desc, leave=False, ascii=True)
        stats: Dict[str, float] = {"loss": 0.0}
        n_samples = 0

        for batch in pbar:
            imgs = batch["image"].to(self.device)          # (B,3,H,W)
            _ = self.model(imgs)
            loss = torch.tensor(0.0, device=self.device)

            if hasattr(self, "metrics") and self.metrics:
                for name, fn in self.metrics.items():
                    stats[name] = fn(self.model, loader, self.device)

            batch_sz = imgs.size(0)
            n_samples += batch_sz
            stats["loss"] += loss.item() * batch_sz
            pbar.set_postfix({"loss": f"{stats['loss']/n_samples:.4f}"})

        stats["loss"] = stats["loss"] / max(n_samples, 1)
        return stats

    @torch.no_grad()
    def _build_memory_bank(self) -> None:
        if self.memory_built:
            return

        print("\n[INFO] Building PatchCore memory bank (coreset sampling) ...")
        self.model.subsample_embedding(self.coreset_sampling_ratio)
        self.memory_built = True
        print(f"[INFO] Memory bank size : {self.model.memory_bank.shape[0]} patches")

    def fit(self,
            train_loader,
            num_epochs: int,
            valid_loader=None,
            weight_path: str | None = None):

        # 1) BaseTrainer.fit 실행 (run_epoch 은 위에서 오버라이드됨)
        history = super().fit(
            train_loader=train_loader,
            num_epochs=num_epochs,
            valid_loader=valid_loader,
            weight_path=None,          # weight 저장은 아래에서 직접 수행
        )

        # 2) 메모리‑뱅크(코어셋) 생성
        self._build_memory_bank()

        # 3) (선택) checkpoint 저장
        if weight_path is not None:
            ckpt = {
                "memory_bank": self.model.memory_bank.cpu(),
                "cfg": self.cfg,
            }
            torch.save(ckpt, weight_path)
            print(f"[INFO] PatchCore checkpoint saved to {weight_path}")

        return history

    @torch.no_grad()
    def infer(self, img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        if img_tensor.dim() == 3:          # (3,H,W) → (1,3,H,W)
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(self.device)
        return self.model.predict(img_tensor)

    def load_memory(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.memory_bank = ckpt["memory_bank"].to(self.device)
        self.cfg = ckpt["cfg"]
        self.memory_built = True
        print(f"[INFO] Loaded PatchCore memory bank from {ckpt_path}")

    def save_memory(self, ckpt_path: str):
        ckpt = {
            "memory_bank": self.model.memory_bank.cpu(),
            "cfg": self.cfg,
        }
        torch.save(ckpt, ckpt_path)
        print(f"[INFO] PatchCore memory bank saved to {ckpt_path}")



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchcoreModel(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    ).to(device)
    input_tensor = torch.randn(32, 3, 256, 256).to(device)
    output = model(input_tensor)
    print(output.shape)     # torch.Size([32768, 1536])
