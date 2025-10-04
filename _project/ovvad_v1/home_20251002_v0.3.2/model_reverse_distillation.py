from collections.abc import Sequence, Callable
from enum import Enum

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
# anomalib/src/anomalib/models/components/backbone/resnet_decoder.py
#############################################################

from torchvision.models.resnet import conv1x1, conv3x3

class DecoderBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            msg = "BasicBlock only supports groups=1 and base_width=64"
            raise ValueError(msg)
        if dilation > 1:
            msg = "Dilation > 1 not supported in BasicBlock"
            raise NotImplementedError(msg)
        # Both self.conv1 and self.downsample layers downsample the input when stride != 2
        if stride == 2:
            self.conv1 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=2,
                stride=stride,
                groups=groups,
                bias=False,
                dilation=dilation,
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        return self.relu(out)


class DecoderBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 2
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = nn.ConvTranspose2d(
                width,
                width,
                kernel_size=2,
                stride=stride,
                groups=groups,
                bias=False,
                dilation=dilation,
            )
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        return self.relu(out)


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        block: type[DecoderBasicBlock | DecoderBottleneck],
        layers: list[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, DecoderBottleneck):
                    nn.init.constant_(module.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(module, DecoderBasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[DecoderBasicBlock | DecoderBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=2,
                    stride=stride,
                    groups=self.groups,
                    bias=False,
                    dilation=self.dilation,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
                for _ in range(1, blocks)
            ],
        )

        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> list[torch.Tensor]:
        feature_a = self.layer1(batch)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64
        return [feature_c, feature_b, feature_a]


def _resnet(block: type[DecoderBasicBlock | DecoderBottleneck], layers: list[int], **kwargs) -> ResNetDecoder:
    return ResNetDecoder(block, layers, **kwargs)


def de_resnet18() -> ResNetDecoder:
    """ResNet-18 model."""
    return _resnet(DecoderBasicBlock, [2, 2, 2, 2])


def de_resnet34() -> ResNetDecoder:
    """ResNet-34 model."""
    return _resnet(DecoderBasicBlock, [3, 4, 6, 3])


def de_resnet50() -> ResNetDecoder:
    """ResNet-50 model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3])


def de_resnet101() -> ResNetDecoder:
    """ResNet-101 model."""
    return _resnet(DecoderBottleneck, [3, 4, 23, 3])


def de_resnet152() -> ResNetDecoder:
    """ResNet-152 model."""
    return _resnet(DecoderBottleneck, [3, 8, 36, 3])


def de_resnext50_32x4d() -> ResNetDecoder:
    """ResNeXt-50 32x4d model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)


def de_resnext101_32x8d() -> ResNetDecoder:
    """ResNeXt-101 32x8d model."""
    return _resnet(DecoderBottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)


def de_wide_resnet50_2() -> ResNetDecoder:
    """Wide ResNet-50-2 model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3], width_per_group=128)


def de_wide_resnet101_2() -> ResNetDecoder:
    """Wide ResNet-101-2 model."""
    return _resnet(DecoderBottleneck, [3, 4, 23, 3], width_per_group=128)


def get_decoder(name: str) -> ResNetDecoder:
    decoder_map = {
        "resnet18": de_resnet18,
        "resnet34": de_resnet34,
        "resnet50": de_resnet50,
        "resnet101": de_resnet101,
        "resnet152": de_resnet152,
        "resnext50_32x4d": de_resnext50_32x4d,
        "resnext101_32x8d": de_resnext101_32x8d,
        "wide_resnet50_2": de_wide_resnet50_2,
        "wide_resnet101_2": de_wide_resnet101_2,
    }

    if name in decoder_map:
        decoder = decoder_map[name]
    else:
        msg = f"Decoder with architecture {name} not supported"
        raise ValueError(msg)
    return decoder()

#############################################################
# anomalib/src/anomalib/models/image/reverse_distillation/components/bottleneck.py
#############################################################

from torchvision.models.resnet import BasicBlock, Bottleneck


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OCBE(nn.Module):
    def __init__(
        self,
        block: Bottleneck | BasicBlock,
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        # self.conv4 and self.bn4 are from the original code:
        # https://github.com/hq-deng/RD4AD/blob/6554076872c65f8784f6ece8cfb39ce77e1aee12/resnet.py#L412
        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[Bottleneck | BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
                for _ in range(1, blocks)
            ],
        )

        return nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Always assumes that features has length of 3
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], 1)
        output = self.bn_layer(feature_cat)

        return output.contiguous()


def get_bottleneck_layer(backbone: str, **kwargs) -> OCBE:
    return OCBE(BasicBlock, 2, **kwargs) if backbone in {"resnet18", "resnet34"} else OCBE(Bottleneck, 3, **kwargs)


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


#############################################################
# anomalib\models\images\reverse_distillation\anomaly_map.py
#############################################################

class AnomalyMapGenerationMode(str, Enum):
    ADD = "add"
    MULTIPLY = "multiply"


class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        image_size,
        sigma: int = 4,
        mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.MULTIPLY,
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in {AnomalyMapGenerationMode.ADD, AnomalyMapGenerationMode.MULTIPLY}:
            msg = f"Found mode {mode}. Only multiply and add are supported."
            raise ValueError(msg)
        self.mode = mode

    def forward(self, student_features: list[torch.Tensor], teacher_features: list[torch.Tensor]) -> torch.Tensor:
        if self.mode == AnomalyMapGenerationMode.MULTIPLY:
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )  # b c h w
        elif self.mode == AnomalyMapGenerationMode.ADD:
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )

        for student_feature, teacher_feature in zip(student_features, teacher_features, strict=True):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)
            if self.mode == AnomalyMapGenerationMode.MULTIPLY:
                anomaly_map *= distance_map
            elif self.mode == AnomalyMapGenerationMode.ADD:
                anomaly_map += distance_map

        gaussian_blur = GaussianBlur2d(
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma=(self.sigma, self.sigma),
        ).to(student_features[0].device)

        return gaussian_blur(anomaly_map)


###########################################################
# anomalib\models\images\reverse_distillation\loss.py
###########################################################

class ReverseDistillationLoss(nn.Module):
    @staticmethod
    def forward(encoder_features: list[torch.Tensor], decoder_features: list[torch.Tensor]) -> torch.Tensor:
        cos_loss = torch.nn.CosineSimilarity()
        loss_sum = 0
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            loss_sum += torch.mean(
                1
                - cos_loss(
                    encoder_feature.view(encoder_feature.shape[0], -1),
                    decoder_feature.view(decoder_feature.shape[0], -1),
                ),
            )
        return loss_sum


###########################################################
# anomalib\models\images\reverse_distillation\torch_model.py
###########################################################

class ReverseDistillation(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: Sequence[str],
        anomaly_map_mode: AnomalyMapGenerationMode,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        encoder_backbone = backbone
        self.encoder = TimmFeatureExtractor(backbone=encoder_backbone, pre_trained=pre_trained, layers=layers)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.decoder = get_decoder(backbone)

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size, mode=anomaly_map_mode)

    def forward(self, images: torch.Tensor):
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.tiler:
            for i, features in enumerate(encoder_features):
                encoder_features[i] = self.tiler.untile(features)
            for i, features in enumerate(decoder_features):
                decoder_features[i] = self.tiler.untile(features)

        return encoder_features, decoder_features

    def predict(self, images: torch.Tensor):
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.tiler:
            for i, features in enumerate(encoder_features):
                encoder_features[i] = self.tiler.untile(features)
            for i, features in enumerate(decoder_features):
                decoder_features[i] = self.tiler.untile(features)

        anomaly_map = self.anomaly_map_generator(encoder_features, decoder_features)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


#############################################################
# Trainer for Reverse Distillation Model
#############################################################

class ReverseDistillationTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(
                params=list(model.decoder.parameters()) + list(model.bottleneck.parameters()),
                lr=0.005, betas=(0.5, 0.99))
        if loss_fn is None:
            loss_fn = ReverseDistillationLoss()

        super().__init__(model, optimizer, loss_fn, metrics, device)
        self.epoch_period = 5

    def train_step(self, batch):
        images = batch["image"].to(self.device)

        self.optimizer.zero_grad()
        encoder_features, decoder_features = self.model(images)
        loss = self.loss_fn(encoder_features, decoder_features)
        loss.backward()
        self.optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                results[name] = metric_fn(encoder_features, decoder_features).item()
        return results


if __name__ == "__main__":
    model = ReverseDistillation(
        backbone="wide_resnet50_2",
        input_size=(256, 256),
        layers=["layer1", "layer2", "layer3"],
        anomaly_map_mode="multiply"
    )
    input_tensor = torch.randn(1, 3, 256, 256)
    # Training mode
    model.train()
    encoder_features, decoder_features = model(input_tensor)
    print([f.shape for f in encoder_features])
    print([f.shape for f in decoder_features])
    # Evaluation mode
    model.eval()
    predictions = model(input_tensor)
    print(predictions["pred_score"].shape)
    print(predictions["anomaly_map"].shape)