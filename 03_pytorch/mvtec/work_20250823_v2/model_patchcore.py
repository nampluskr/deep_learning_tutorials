import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import faiss
from tqdm import tqdm


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor for PatchCore using ResNet backbone"""
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'], weights=None):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers

        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=weights)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.features = {}
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        for name, module in self.backbone.named_modules():
            if any(layer in name for layer in self.layers):
                handle = module.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)

    def forward(self, x):
        self.features.clear()
        _ = self.backbone(x)
        outputs = []
        for name in self.features:
            if any(layer in name for layer in self.layers):
                outputs.append(self.features[name])
        return outputs

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()


class PatchCore(nn.Module):
    """PatchCore anomaly detection"""
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'],
                 memory_reduction=0.1, patch_size=32):
        super().__init__()
        self.model_type = "patchcore"
        self.feature_extractor = ResNetFeatureExtractor(backbone, layers, weights=None)
        self.memory_bank = None
        self.memory_reduction = memory_reduction
        self.index = None
        self.patch_size = patch_size

    @torch.no_grad()
    def _embed_features(self, features):
        """Pool and flatten multi-scale features"""
        pooled = [F.adaptive_avg_pool2d(f, (self.patch_size, self.patch_size)) for f in features]
        flat = [p.flatten(1) for p in pooled]
        embedding = torch.cat(flat, dim=1)  # (B, D)
        return embedding

    @torch.no_grad()
    def build_memory_bank(self, data_loader, device):
        """Build memory bank from normal training set"""
        features_list = []
        for batch in tqdm(data_loader, desc="Building Memory Bank"):
            inputs = batch["image"].to(device)
            features = self.feature_extractor(inputs)
            embedding = self._embed_features(features)
            features_list.append(embedding.cpu())
        features_all = torch.cat(features_list, dim=0)

        # Subsample (memory reduction)
        n_samples = int(len(features_all) * self.memory_reduction)
        indices = torch.randperm(len(features_all))[:n_samples]
        self.memory_bank = features_all[indices]

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.memory_bank.shape[1])
        self.index.add(self.memory_bank.numpy().astype("float32"))
        print(f"Memory bank built with {self.memory_bank.shape[0]} samples, dim={self.memory_bank.shape[1]}")

    def forward(self, batch_data):
        x = batch_data['input']
        features = self.feature_extractor(x)
        embedding = self._embed_features(features)
        return {"features": embedding, "input": x}

    def compute_anomaly_scores(self, outputs):
        """Compute anomaly scores using nearest neighbor search"""
        if self.index is None:
            raise RuntimeError("Memory bank not built. Call build_memory_bank().")

        feats = outputs["features"].cpu().numpy().astype("float32")
        D, _ = self.index.search(feats, k=1)  # nearest neighbor distance
        scores = torch.tensor(D.squeeze(), device=outputs["features"].device)
        return scores

    def train_step(self, data, optimizer, loss_fn, metrics, device):
        """PatchCore is non-parametric (no gradient training)"""
        return {"loss": 0.0}

    @torch.no_grad()
    def validate_step(self, data, loss_fn, metrics, device):
        inputs = data["image"].to(device)
        outputs = self.forward({"input": inputs})
        scores = self.compute_anomaly_scores(outputs)

        results = {"loss": scores.mean().item()}
        for name, metric_fn in metrics.items():
            results[name] = float(metric_fn(scores))
        return results


# Loss & Metrics
def patchcore_loss(pred, target=None):
    """PatchCore does not train with gradient descent (dummy loss)"""
    return torch.tensor(0.0, device=pred["features"].device)


def patchcore_anomaly_score_metric(scores):
    """Mean anomaly score"""
    return scores.mean().item()


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCore(backbone="resnet18", layers=["layer2", "layer3"]).to(device)

    dummy = {"input": torch.randn(4, 3, 256, 256).to(device)}
    outputs = model(dummy)
    print("Features shape:", outputs["features"].shape)
