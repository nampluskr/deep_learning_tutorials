import os
from collections.abc import Sequence
from typing import Union, Tuple

import timm
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torch.fx.graph_module import GraphModule
from torchvision.models import resnet18, resnet34, resnet50
from torch.fx.graph_module import GraphModule


BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "wide_resnet50_2-32ee1156.pth",
    "efficientnet_b5": "efficientnet_b5_lukemelas-1a07897c.pth",
}

def get_local_weight_path(backbone: str):
    filename = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.pth")
    weight_path = os.path.join(BACKBONE_DIR, filename)
    if os.path.isfile(weight_path):
        print(f" > [Info] {backbone} weight is found in {weight_path}.")
    else:
        print(f" > [Warning] {backbone} weight not found in {weight_path}. ")
    return weight_path


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


class TimmFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad

        if isinstance(backbone, nn.Module):
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in self.layers},
            )
            layer_metadata = dryrun_find_featuremap_dims(self.feature_extractor, (256, 256), layers=self.layers)
            self.out_dims = [feature_info["num_features"] for layer_name, feature_info in layer_metadata.items()]

        elif isinstance(backbone, str):
            self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=False,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            self.out_dims = self.feature_extractor.feature_info.channels()
            if pre_trained:
                weight_path = get_local_weight_path(backbone)
                state_dict = torch.load(weight_path, map_location="cpu")
                self.feature_extractor.load_state_dict(state_dict, strict=False)
        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self) -> list[int]:
        idx_list = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        layer_names = [info["module"] for info in model.feature_info.info]

        invalid_layers = []
        for layer in self.layers:
            try:
                idx_list.append(layer_names.index(layer))
            except ValueError:
                print(f"[Warning] Layer '{layer}' not found in model "
                      f"'{self.backbone}'. Available: {layer_names}")
                invalid_layers.append(layer)
        self.layers = [l for l in self.layers if l not in invalid_layers]
        return idx_list

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.requires_grad:
            feats = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                feats = self.feature_extractor(inputs)

        if not isinstance(feats, dict):
            feats = dict(zip(self.layers, feats, strict=True))
        return feats


def load_pretrained_model(backbone, device="cpu"):
    if backbone == "resnet18":
        model = resnet18(weights=None)
    elif backbone == "resnet34":
        model = resnet34(weights=None)
    elif backbone == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. ")
    model.to(device)

    weight_path = get_local_weight_path(backbone)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 1:
        for name in ["resnet18", "resnet34", "resnet50"]:
            extractor = TimmFeatureExtractor(
                backbone=name,
                layers=["layer1", "layer2", "layer3"],
                pre_trained=True,
                requires_grad=False,
            ).to(device)
            x = torch.randn(1, 3, 256, 256).to(device)
            feats = extractor(x)
            for layer, feature in feats.items():
                print(f"{name}: {layer}: {feature.shape}")

    if 1:
        for name in ["resnet18", "resnet34", "resnet50"]:
            model = load_pretrained_model(name, device=device)
            model.eval()
            x = torch.randn(1, 3, 256, 256).to(device)
            with torch.no_grad():
                out = model(x)
            print(f"{name} forward output shape: {out.shape}")
