import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections.abc import Sequence
import timm
import os

try:
    from torchvision.models.feature_extraction import create_feature_extractor
    HAS_TORCHVISION_FEATURE_EXTRACTION = True
except ImportError:
    HAS_TORCHVISION_FEATURE_EXTRACTION = False

try:
    from torch.fx.graph_module import GraphModule
    HAS_TORCH_FX = True
except ImportError:
    HAS_TORCH_FX = False

logger = logging.getLogger(__name__)


BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet50": "resnet50-0676ba61.pth", 
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
}


def get_backbone_path(backbone):
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join("backbones", filename)
    else:
        return os.path.join("backbones", f"{backbone}.pth")


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
            local_weights_path = get_backbone_path(backbone)
            self.idx = self._map_layer_to_idx()
            
            if os.path.exists(local_weights_path):
                logger.info(f"Loading local weights from {local_weights_path}")
                self.feature_extractor = timm.create_model(
                    backbone,
                    pretrained=False,
                    pretrained_cfg=None,
                    features_only=True,
                    exportable=True,
                    out_indices=self.idx,
                )
                try:
                    state_dict = torch.load(local_weights_path, map_location='cpu')
                    self.feature_extractor.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded local weights for {backbone}")
                except Exception as e:
                    logger.warning(f"Failed to load local weights: {e}. Using random initialization.")
            else:
                logger.warning(f"Local weights not found at {local_weights_path}")
                if pre_trained:
                    logger.warning("Internet connection unavailable. Using random initialization instead of pretrained weights.")
                
                self.feature_extractor = timm.create_model(
                    backbone,
                    pretrained=False,
                    pretrained_cfg=None,
                    features_only=True,
                    exportable=True,
                    out_indices=self.idx,
                )
            
            self.out_dims = self.feature_extractor.feature_info.channels()

        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self):
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


if __name__ == "__main__":
    pass
