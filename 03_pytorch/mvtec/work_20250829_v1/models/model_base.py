# models/model_base.py
# 공통 모델 블록 (ResNetFeatureExtractor, EncoderBlock, DecoderBlock)

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using ResNet backbone.
    특정 레이어(layer1~layer4) 출력을 반환하여 feature pyramid로 사용.
    """

    def __init__(self, backbone="resnet18", layers=["layer1", "layer2", "layer3"]):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers

        if backbone == "resnet18":
            backbone_model = models.resnet18(weights=None)
        elif backbone == "resnet34":
            backbone_model = models.resnet34(weights=None)
        elif backbone == "resnet50":
            backbone_model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4

        # 레이어 매핑
        self._layers = {
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }

    def forward(self, x: torch.Tensor):
        outputs = []
        x = self.stem(x)
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = self._layers[name]
            x = layer(x)
            if name in self.layers:
                outputs.append(x)
        return outputs


# -------------------------
# Autoencoder / VAE용 블록
# -------------------------

class EncoderBlock(nn.Module):
    """Conv-BN-ReLU + Downsampling (stride=2)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """ConvTranspose-BN-ReLU, 마지막 레이어는 Sigmoid"""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, final_layer=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch) if not final_layer else None
        self.relu = nn.ReLU(inplace=True) if not final_layer else nn.Sigmoid()
        self.final_layer = final_layer

    def forward(self, x):
        x = self.deconv(x)
        if self.final_layer:
            x = self.relu(x)
        else:
            x = self.relu(self.bn(x))
        return x
