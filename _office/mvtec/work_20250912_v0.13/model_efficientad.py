import os
import math
import random
import logging
from enum import Enum
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_base import BACKBONE_DIR

logger = logging.getLogger(__name__)


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize batch of images using ImageNet mean and standard deviation."""
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    """Reduce the number of elements in a tensor by random sampling."""
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes."""
    M = "medium"
    S = "small"


def get_teacher_weight_path(model_size: str) -> str:
    """Get local teacher weight path."""
    if model_size.lower() in ["s", "small"]:
        filename = "pretrained_teacher_small.pth"
    elif model_size.lower() in ["m", "medium"]:
        filename = "pretrained_teacher_medium.pth"
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    return os.path.join(BACKBONE_DIR, filename)


def load_teacher_weights(model, model_size: str, device: str = "cpu"):
    """Load teacher weights from local backbones directory."""
    weights_path = get_teacher_weight_path(model_size)

    if not os.path.exists(weights_path):
        available_files = os.listdir(BACKBONE_DIR) if os.path.exists(BACKBONE_DIR) else []
        raise FileNotFoundError(
            f"Teacher weights not found at {weights_path}\n"
            f"Available files in {BACKBONE_DIR}: {available_files}\n"
            f"Please download from: https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
        )

    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Successfully loaded teacher weights from {weights_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load teacher weights from {weights_path}: {e}")


class SmallPatchDescriptionNetwork(nn.Module):
    """Small variant of the Patch Description Network."""

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    """Medium-sized patch description network."""

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class Encoder(nn.Module):
    """Encoder module for the autoencoder architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder network."""
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    """Decoder module for the autoencoder architecture."""

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = padding
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Perform a forward pass through the network."""
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    """EfficientAd Autoencoder."""

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)


class EfficientADModel(nn.Module):
    """EfficientAd model with local weight loading and ImageNette augmentation."""

    def __init__(
        self,
        teacher_out_channels: int = 384,
        model_size: str = "small",
        padding: bool = False,
        pad_maps: bool = True,
        use_imagenet_penalty: bool = True,
    ) -> None:
        super().__init__()

        self.pad_maps = pad_maps
        self.use_imagenet_penalty = use_imagenet_penalty
        self.model_size = model_size

        # Validate model size
        if model_size.lower() not in ["small", "s", "medium", "m"]:
            raise ValueError(f"Unknown model size {model_size}. Use 'small' or 'medium'")

        # Create teacher and student networks based on model size
        if model_size.lower() in ["medium", "m"]:
            self.teacher = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding)
            self.student = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)
        else:  # small
            self.teacher = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding)
            self.student = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        # Freeze teacher and load pretrained weights
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Load teacher weights from local backbones directory
        load_teacher_weights(self.teacher, model_size)

        # AutoEncoder
        self.ae = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels = teacher_out_channels

        # Normalization parameters (computed during training)
        self.mean_std = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            },
        )

        # Quantiles for normalization (computed during validation)
        self.quantiles = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            },
        )

    @staticmethod
    def is_set(p_dic: nn.ParameterDict) -> bool:
        """Check if any parameters in the dictionary are non-zero."""
        return any(value.sum() != 0 for _, value in p_dic.items())

    @staticmethod
    def choose_random_aug_image(image: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to input image."""
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        coefficient = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        idx = int(torch.randint(0, len(transform_functions), (1,)).item())
        transform_function = transform_functions[idx]
        return transform_function(image, coefficient)

    def create_imagenet_like_augmentation(self, batch: torch.Tensor) -> torch.Tensor:
        """Create ImageNet-like augmented version of input batch."""
        aug_batch = batch.clone()

        # Apply multiple strong augmentations
        transforms_list = [
            lambda x: torch.clamp(x + 0.3 * torch.randn_like(x), 0, 1),  # Gaussian noise
            lambda x: transforms.functional.adjust_brightness(x, 1.5),    # Brightness
            lambda x: transforms.functional.adjust_contrast(x, 1.5),      # Contrast
            lambda x: transforms.functional.adjust_saturation(x, 0.5),    # Saturation
            lambda x: transforms.functional.adjust_hue(x, 0.2),           # Hue shift
            lambda x: transforms.functional.gaussian_blur(x, 3),          # Blur
        ]

        # Apply 2-3 random transformations
        num_transforms = random.randint(2, 3)
        selected_transforms = random.sample(transforms_list, num_transforms)

        for transform in selected_transforms:
            try:
                aug_batch = transform(aug_batch)
            except Exception:
                continue

        return torch.clamp(aug_batch, 0, 1)

    def compute_anomaly_map(self, map_st: torch.Tensor, map_stae: torch.Tensor) -> torch.Tensor:
        """EfficientAD-specific: Student-Teacher + AutoEncoder map combination."""
        return 0.5 * map_st + 0.5 * map_stae

    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """EfficientAD-specific: Max pooling for image-level score."""
        return torch.amax(anomaly_map, dim=(-2, -1))

    def forward(
        self,
        batch: torch.Tensor,
        batch_imagenet: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict:
        """Forward pass through the model."""
        student_output, distance_st = self.compute_student_teacher_distance(batch)

        if self.training:
            return self.compute_losses(batch, batch_imagenet, distance_st)
        else:
            map_st, map_stae = self.compute_maps(batch, student_output, distance_st, normalize)
            anomaly_map = self.compute_anomaly_map(map_st, map_stae)
            # pred_score = self.compute_anomaly_score(anomaly_map)
            pred_score = torch.amax(anomaly_map.view(anomaly_map.size(0), -1), dim=1) # max_pooling
            return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_student_teacher_distance(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the student-teacher distance vectors."""
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)
        return student_output, distance_st

    def compute_losses(
        self,
        batch: torch.Tensor,
        batch_imagenet: Optional[torch.Tensor],
        distance_st: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute training losses."""
        # Student loss - hard examples
        distance_st_flat = reduce_tensor_elems(distance_st)
        d_hard = torch.quantile(distance_st_flat, 0.999)
        loss_hard = torch.mean(distance_st_flat[distance_st_flat >= d_hard])

        # ImageNet penalty loss
        if self.use_imagenet_penalty:
            if batch_imagenet is None:
                # Create ImageNet-like augmentation if no external ImageNet provided
                batch_imagenet = self.create_imagenet_like_augmentation(batch)

            student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        # Autoencoder and Student AE Loss
        aug_img = self.choose_random_aug_image(batch)
        ae_output_aug = self.ae(aug_img, batch.shape[-2:])

        with torch.no_grad():
            teacher_output_aug = self.teacher(aug_img)
            if self.is_set(self.mean_std):
                teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

        student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

        distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
        distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        return (loss_st, loss_ae, loss_stae)

    def compute_maps(
        self,
        batch: torch.Tensor,
        student_output: torch.Tensor,
        distance_st: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly maps from model outputs."""
        image_size = batch.shape[-2:]

        with torch.no_grad():
            ae_output = self.ae(batch, image_size)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))

        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")

        if self.is_set(self.quantiles) and normalize:
            map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            map_stae = 0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])

        return map_st, map_stae

    def get_maps(self, batch: torch.Tensor, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly maps for a batch of images."""
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        return self.compute_maps(batch, student_output, distance_st, normalize)


if __name__ == "__main__":
    # Test EfficientAD model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Small model
    print("Testing EfficientAD Small model...")
    model_small = EfficientADModel(
        model_size="small",
        teacher_out_channels=384,
        use_imagenet_penalty=True
    ).to(device)

    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)

    # Training mode
    model_small.train()
    loss_st, loss_ae, loss_stae = model_small(x)
    total_loss = loss_st + loss_ae + loss_stae
    print(f"Training losses - ST: {loss_st.item():.4f}, AE: {loss_ae.item():.4f}, STAE: {loss_stae.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    # Inference mode
    model_small.eval()
    with torch.no_grad():
        output = model_small(x)
        print(f"Inference - Pred score: {output['pred_score'].shape}, Anomaly map: {output['anomaly_map'].shape}")

    print("EfficientAD model test completed successfully!")
