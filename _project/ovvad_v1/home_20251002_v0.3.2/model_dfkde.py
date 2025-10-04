from abc import ABC
import logging
import random
from enum import Enum
from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from feature_extractor import TimmFeatureExtractor
from trainer import BaseTrainer

logger = logging.getLogger(__name__)


###########################################################
# anomalib/src/anomalib/models/components/base/dynamic_buffer.py
###########################################################

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


###########################################################
# anomalib/src/anomalib/models/components/stats/kde.py
###########################################################

class GaussianKDE(DynamicBufferMixin):
    def __init__(self, dataset: torch.Tensor | None = None) -> None:
        super().__init__()

        if dataset is not None:
            self.fit(dataset)

        self.register_buffer("bw_transform", torch.empty(0))
        self.register_buffer("dataset", torch.empty(0))
        self.register_buffer("norm", torch.empty(0))

        self.bw_transform = torch.empty(0)
        self.dataset = torch.empty(0)
        self.norm = torch.empty(0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = torch.exp(-embedding / 2) * self.norm
            estimate[i] = torch.mean(embedding)

        return estimate

    def fit(self, dataset: torch.Tensor) -> None:
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2

        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)

        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(tensor, dim=1)
        tensor -= mean[:, None]
        return torch.matmul(tensor, tensor.T) / (tensor.size(1) - 1)
    

###########################################################
# anomalib/src/anomalib/models/components/dimensionality_reduction/pca.py
###########################################################

class PCA(DynamicBufferMixin):
    def __init__(self, n_components: int | float) -> None:
        super().__init__()
        self.n_components = n_components

        self.register_buffer("singular_vectors", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("num_components", torch.empty(0))

        self.singular_vectors: torch.Tensor
        self.singular_values: torch.Tensor
        self.mean: torch.Tensor
        self.num_components: torch.Tensor

    def fit(self, dataset: torch.Tensor) -> None:
        mean = dataset.mean(dim=0)
        dataset -= mean

        _, sig, v_h = torch.linalg.svd(dataset.double(), full_matrices=False)
        num_components: int
        if self.n_components <= 1:
            variance_ratios = torch.cumsum(sig * sig, dim=0) / torch.sum(sig * sig)
            num_components = torch.nonzero(variance_ratios >= self.n_components)[0]
        else:
            num_components = int(self.n_components)

        self.num_components = torch.tensor([num_components], device=dataset.device)

        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components].float()
        self.singular_values = sig[:num_components].float()
        self.mean = mean

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        mean = dataset.mean(dim=0)
        dataset -= mean
        num_components = int(self.n_components)
        self.num_components = torch.tensor([num_components], device=dataset.device)

        v_h = torch.linalg.svd(dataset)[-1]
        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components]
        self.mean = mean

        return torch.matmul(dataset, self.singular_vectors)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        features -= self.mean
        return torch.matmul(features, self.singular_vectors)

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features, self.singular_vectors.transpose(-2, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.transform(features)


###########################################################
# anomalib/src/anomalib/models/components/classification/kde_classifier.py
###########################################################

class FeatureScalingMethod(str, Enum):
    NORM = "norm"  # scale to unit vector length
    SCALE = "scale"  # scale to max length observed in training


class KDEClassifier(nn.Module):
    def __init__(
        self,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.n_pca_components = n_pca_components
        self.feature_scaling_method = feature_scaling_method
        self.max_training_points = max_training_points

        self.pca_model = PCA(n_components=self.n_pca_components)
        self.kde_model = GaussianKDE()

        self.register_buffer("max_length", torch.empty([]))
        self.max_length = torch.empty([])

    def pre_process(
        self,
        feature_stack: torch.Tensor,
        max_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_length is None:
            max_length = torch.max(torch.linalg.norm(feature_stack, ord=2, dim=1))

        if self.feature_scaling_method == FeatureScalingMethod.NORM:
            feature_stack /= torch.linalg.norm(feature_stack, ord=2, dim=1)[:, None]
        elif self.feature_scaling_method == FeatureScalingMethod.SCALE:
            feature_stack /= max_length
        else:
            msg = "Unknown pre-processing mode. Available modes are: Normalized and Scale."
            raise RuntimeError(msg)
        return feature_stack, max_length

    def fit(self, embeddings: torch.Tensor) -> bool:
        if embeddings.shape[0] < self.n_pca_components:
            logger.info("Not enough features to commit. Not making a model.")
            return False

        # if max training points is non-zero and smaller than number of staged features, select random subset
        if embeddings.shape[0] > self.max_training_points:
            selected_idx = torch.tensor(
                random.sample(range(embeddings.shape[0]), self.max_training_points),
                device=embeddings.device,
            )
            selected_features = embeddings[selected_idx]
        else:
            selected_features = embeddings

        feature_stack = self.pca_model.fit_transform(selected_features)
        feature_stack, max_length = self.pre_process(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)

        return True

    def compute_kde_scores(self, features: torch.Tensor, as_log_likelihood: bool | None = False) -> torch.Tensor:
        features = self.pca_model.transform(features)
        features, _ = self.pre_process(features, self.max_length)
        # Scores are always assumed to be passed as a density
        kde_scores = self.kde_model(features)

        # add small constant to avoid zero division in log computation
        kde_scores += 1e-300

        if as_log_likelihood:
            kde_scores = torch.log(kde_scores)

        return kde_scores

    @staticmethod
    def compute_probabilities(scores: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(0.05 * (scores - 12)))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        scores = self.compute_kde_scores(features, as_log_likelihood=True)
        return self.compute_probabilities(scores)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict(features)


###########################################################
# anomalib\models\images\dfkde\torch_model.py
###########################################################

class DFKDE(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool = True,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.feature_extractor = TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers).eval()

        self.classifier = KDEClassifier(
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )
        self.memory_bank: list[torch.tensor] = []

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch)
        for layer in layer_outputs:
            batch_size = len(layer_outputs[layer])
            layer_outputs[layer] = F.adaptive_avg_pool2d(input=layer_outputs[layer], output_size=(1, 1))
            layer_outputs[layer] = layer_outputs[layer].view(batch_size, -1)
        return torch.cat(list(layer_outputs.values())).detach()

    def forward(self, batch: torch.Tensor):
        # 1. apply feature extraction
        features = self.get_features(batch)
        self.memory_bank.append(features)
        return features

    def predict(self, batch: torch.Tensor):
        # 1. apply feature extraction
        features = self.get_features(batch)
        # 2. apply density estimation
        scores = self.classifier(features)
        return dict(pred_score=scores)

    def fit(self) -> None:
        if len(self.memory_bank) == 0:
            msg = "Memory bank is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        self.memory_bank = torch.vstack(self.memory_bank)

        # fit gaussian
        self.classifier.fit(self.memory_bank)

        # clear memory bank, redcues gpu size
        self.memory_bank = []


#############################################################
# Trainer for DFKDE Model
#############################################################

from tqdm import tqdm

class DFKDETrainer(BaseTrainer):
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
    
    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
            skip_normal=False, skip_anomaly=False, num_max=-1, imagenet_normalize=True):
        print(" > DFKDE model does not support anomaly map visualization.")
        print(" > Use validation_epoch for evaluation metrics only.")


if __name__ == "__main__":
    pass