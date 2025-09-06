import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path
from collections.abc import Callable, Sequence
import random

from kornia.losses import FocalLoss, SSIMLoss
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2


# ===================================================================
# Perlin Noise Generator (from anomalib)
# ===================================================================

def generate_perlin_noise(
    height: int,
    width: int,
    scale: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a Perlin noise pattern."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle scale parameter
    if scale is None:
        min_scale, max_scale = 0, 6
        scalex = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
        scaley = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
    else:
        scalex, scaley = scale

    # Ensure dimensions are powers of 2 for proper noise generation
    def nextpow2(value: int) -> int:
        return int(2 ** torch.ceil(torch.log2(torch.tensor(value))).int().item())

    pad_h = nextpow2(height)
    pad_w = nextpow2(width)

    # Generate base grid
    delta = (scalex / pad_h, scaley / pad_w)
    d = (pad_h // scalex, pad_w // scaley)

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, scalex, delta[0], device=device),
                torch.arange(0, scaley, delta[1], device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        % 1
    )

    # Generate random gradients
    angles = 2 * torch.pi * torch.rand(int(scalex) + 1, int(scaley) + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1: list[int | None], slice2: list[int | None]) -> torch.Tensor:
        return (
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
            .repeat_interleave(int(d[0]), 0)
            .repeat_interleave(int(d[1]), 1)
        )

    def dot(grad: torch.Tensor, shift: list[float]) -> torch.Tensor:
        return (
            torch.stack(
                (grid[:pad_h, :pad_w, 0] + shift[0], grid[:pad_h, :pad_w, 1] + shift[1]),
                dim=-1,
            )
            * grad[:pad_h, :pad_w]
        ).sum(dim=-1)

    # Calculate noise values at grid points
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    # Interpolate between grid points using quintic curve
    def fade(t: torch.Tensor) -> torch.Tensor:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    t = fade(grid[:pad_h, :pad_w])
    noise = torch.sqrt(torch.tensor(2.0, device=device)) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]),
        torch.lerp(n01, n11, t[..., 0]),
        t[..., 1],
    )

    # Crop to desired dimensions
    return noise[:height, :width]


# MultiRandomChoice class (simplified version)
class MultiRandomChoice(nn.Module):
    """Apply a random subset of transforms from a list."""
    
    def __init__(self, transforms, num_transforms=3, fixed_num_transforms=True):
        super().__init__()
        self.transforms = transforms
        self.num_transforms = num_transforms
        self.fixed_num_transforms = fixed_num_transforms
    
    def forward(self, img):
        """Apply random subset of transforms."""
        if self.fixed_num_transforms:
            selected_transforms = random.sample(self.transforms, self.num_transforms)
        else:
            num_to_apply = random.randint(1, min(len(self.transforms), self.num_transforms))
            selected_transforms = random.sample(self.transforms, num_to_apply)
        
        for transform in selected_transforms:
            try:
                img = transform(img)
            except Exception:
                # Skip transform if it fails
                continue
        return img


class PerlinAnomalyGenerator(v2.Transform):
    """Perlin noise-based synthetic anomaly generator from anomalib."""

    def __init__(
        self,
        anomaly_source_path: Path | str | None = None,
        probability: float = 0.5,
        blend_factor: float | tuple[float, float] = (0.2, 1.0),
        rotation_range: tuple[float, float] = (-90, 90),
    ) -> None:
        super().__init__()
        self.probability = probability
        self.blend_factor = blend_factor

        # Load anomaly source paths
        self.anomaly_source_paths: list[Path] = []
        if anomaly_source_path is not None:
            for img_ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(Path(anomaly_source_path).rglob("*" + img_ext))

        # Initialize perlin rotation transform
        self.perlin_rotation_transform = v2.RandomAffine(
            degrees=rotation_range,
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0,
        )

        # Initialize augmenters
        self.augmenters = MultiRandomChoice(
            transforms=[
                v2.ColorJitter(contrast=(0.5, 2.0)),
                v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
                v2.ColorJitter(hue=[-50 / 360, 50 / 360], saturation=[0.5, 1.5]),
                v2.RandomSolarize(threshold=0.5, p=1.0),
                v2.RandomPosterize(bits=4, p=1.0),
                v2.RandomInvert(p=1.0),
                v2.RandomEqualize(p=1.0),
                v2.RandomAffine(degrees=(-45, 45), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            ],
            num_transforms=3,
            fixed_num_transforms=True,
        )

    def generate_perturbation(
        self,
        height: int,
        width: int,
        device: torch.device | None = None,
        anomaly_source_path: Path | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate perturbed image and mask."""
        # Generate perlin noise
        perlin_noise = generate_perlin_noise(height, width, device=device)

        # Create rotated noise pattern
        perlin_noise = perlin_noise.unsqueeze(0)  # [1, H, W]
        perlin_noise = self.perlin_rotation_transform(perlin_noise).squeeze(0)  # [H, W]

        # Generate binary mask from perlin noise
        mask = torch.where(
            perlin_noise > 0.5,
            torch.ones_like(perlin_noise, device=device),
            torch.zeros_like(perlin_noise, device=device),
        ).unsqueeze(-1)  # [H, W, 1]

        # Generate anomaly source image
        if anomaly_source_path and Path(anomaly_source_path).exists():
            anomaly_source_img = (
                io.read_image(str(anomaly_source_path), mode=io.ImageReadMode.RGB).float().to(device) / 255.0
            )
            if anomaly_source_img.shape[-2:] != (height, width):
                anomaly_source_img = v2.functional.resize(anomaly_source_img, [height, width], antialias=True)
            anomaly_source_img = anomaly_source_img.permute(1, 2, 0)  # [H, W, C]
        else:
            anomaly_source_img = perlin_noise.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, C]
            anomaly_source_img = (anomaly_source_img * 0.5) + 0.25  # Adjust intensity range

        # Apply augmentations to source image
        anomaly_augmented = self.augmenters(anomaly_source_img.permute(2, 0, 1))  # [C, H, W]
        anomaly_augmented = anomaly_augmented.permute(1, 2, 0)  # [H, W, C]

        # Create final perturbation by applying mask
        perturbation = anomaly_augmented * mask

        return perturbation, mask

    def _transform_image(
        self,
        img: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform a single image."""
        if torch.rand(1, device=device) > self.probability:
            return img, torch.zeros((1, h, w), device=device)

        anomaly_source_path = (
            list(self.anomaly_source_paths)[int(torch.randint(len(self.anomaly_source_paths), (1,)).item())]
            if self.anomaly_source_paths
            else None
        )

        perturbation, mask = self.generate_perturbation(h, w, device, anomaly_source_path)
        perturbation = perturbation.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        beta = (
            self.blend_factor
            if isinstance(self.blend_factor, float)
            else torch.rand(1, device=device) * (self.blend_factor[1] - self.blend_factor[0]) + self.blend_factor[0]
            if isinstance(self.blend_factor, tuple)
            else torch.tensor(0.5, device=device)  # Fallback value
        )

        if not isinstance(beta, float):
            beta = beta.view(-1, 1, 1).expand_as(img)

        augmented_img = img * (1 - mask) + beta * perturbation + (1 - beta) * img * mask
        return augmented_img, mask

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation using the mask for single image or batch."""
        device = img.device
        is_batch = len(img.shape) == 4

        if is_batch:
            batch, _, height, width = img.shape
            # Initialize batch outputs
            batch_augmented = []
            batch_masks = []

            for i in range(batch):
                # Apply transform to each image in batch
                augmented, mask = self._transform_image(img[i], height, width, device)
                batch_augmented.append(augmented)
                batch_masks.append(mask)

            return torch.stack(batch_augmented), torch.stack(batch_masks)

        # Handle single image
        return self._transform_image(img, img.shape[1], img.shape[2], device)


# ===================================================================
# SSPCAB Components (Self-Supervised Predictive Convolutional Attention Block)
# ===================================================================

class AttentionModule(nn.Module):
    """Squeeze and excitation block that acts as the attention module in SSPCAB."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        out_channels = in_channels // reduction_ratio
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention module."""
        # Global average pooling
        avg_pooled = inputs.mean(dim=(2, 3))
        
        # Squeeze and excite
        act = self.fc1(avg_pooled)
        act = F.relu(act)
        act = self.fc2(act)
        act = F.sigmoid(act)
        
        # Multiply with input
        return inputs * act.view(act.shape[0], act.shape[1], 1, 1)


class SSPCAB(nn.Module):
    """Self-Supervised Predictive Convolutional Attention Block."""
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        reduction_ratio: int = 8,
    ):
        super().__init__()
        
        self.pad = kernel_size + dilation
        self.crop = kernel_size + 2 * dilation + 1
        
        self.masked_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size)
        self.masked_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size)
        self.masked_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size)
        self.masked_conv4 = nn.Conv2d(in_channels, in_channels, kernel_size)
        
        self.attention_module = AttentionModule(in_channels, reduction_ratio)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SSPCAB block."""
        # Compute masked convolution
        padded = F.pad(inputs, (self.pad,) * 4)
        masked_out = torch.zeros_like(inputs)
        masked_out += self.masked_conv1(padded[..., : -self.crop, : -self.crop])
        masked_out += self.masked_conv2(padded[..., : -self.crop, self.crop :])
        masked_out += self.masked_conv3(padded[..., self.crop :, : -self.crop])
        masked_out += self.masked_conv4(padded[..., self.crop :, self.crop :])
        
        # Apply channel attention module
        return self.attention_module(masked_out)


# ===================================================================
# DRAEM Model Components
# ===================================================================

class EncoderReconstructive(nn.Module):
    """Encoder component of the reconstructive network."""
    
    def __init__(self, in_channels: int, base_width: int, sspcab: bool = False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        
        if sspcab:
            self.block5 = SSPCAB(base_width * 8)
        else:
            self.block5 = nn.Sequential(
                nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode input images to the salient space."""
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp2(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        return self.block5(mp4)


class DecoderReconstructive(nn.Module):
    """Decoder component of the reconstructive network."""
    
    def __init__(self, base_width: int, out_channels: int = 1):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        
        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )

    def forward(self, act5: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from bottleneck features."""
        up1 = self.up1(act5)
        db1 = self.db1(up1)
        
        up2 = self.up2(db1)
        db2 = self.db2(up2)
        
        up3 = self.up3(db2)
        db3 = self.db3(up3)
        
        up4 = self.up4(db3)
        db4 = self.db4(up4)
        
        return self.fin_out(db4)


class ReconstructiveSubNetwork(nn.Module):
    """Autoencoder model for image reconstruction."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_width: int = 128,
        sspcab: bool = False,
    ):
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct input images."""
        encoded = self.encoder(batch)
        return self.decoder(encoded)


class EncoderDiscriminative(nn.Module):
    """Encoder component of the discriminator network."""
    
    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch: torch.Tensor):
        """Convert inputs to salient space through encoder network."""
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp2(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        act5 = self.block5(mp4)
        mp5 = self.mp5(act5)
        act6 = self.block6(mp5)
        return act1, act2, act3, act4, act5, act6


class DecoderDiscriminative(nn.Module):
    """Decoder component of the discriminator network."""
    
    def __init__(self, base_width: int, out_channels: int = 1):
        super().__init__()
        
        self.up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(
        self,
        act1: torch.Tensor,
        act2: torch.Tensor,
        act3: torch.Tensor,
        act4: torch.Tensor,
        act5: torch.Tensor,
        act6: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted anomaly scores from encoder activations."""
        up_b = self.up_b(act6)
        cat_b = torch.cat((up_b, act5), dim=1)
        db_b = self.db_b(cat_b)
        
        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, act4), dim=1)
        db1 = self.db1(cat1)
        
        up2 = self.up2(db1)
        cat2 = torch.cat((up2, act3), dim=1)
        db2 = self.db2(cat2)
        
        up3 = self.up3(db2)
        cat3 = torch.cat((up3, act2), dim=1)
        db3 = self.db3(cat3)
        
        up4 = self.up4(db3)
        cat4 = torch.cat((up4, act1), dim=1)
        db4 = self.db4(cat4)
        
        return self.fin_out(db4)


class DiscriminativeSubNetwork(nn.Module):
    """Discriminative model for anomaly mask prediction."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64):
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Generate predicted anomaly masks."""
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        return self.decoder_segment(act1, act2, act3, act4, act5, act6)


# ===================================================================
# DRAEM Main Model
# ===================================================================

class DRAEMModel(nn.Module):
    """DRAEM PyTorch model with reconstructive and discriminative sub-networks."""
    
    def __init__(self, sspcab: bool = False, anomaly_source_path: str | None = None):
        super().__init__()
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        self.anomaly_generator = PerlinAnomalyGenerator(
            anomaly_source_path=anomaly_source_path,
            probability=0.5,
            blend_factor=(0.1, 1.0)
        )

    def compute_anomaly_map(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                           prediction: torch.Tensor) -> torch.Tensor:
        """DRAEM-specific: Softmax prediction of anomalous class as anomaly map."""
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1:2, ...]  # Keep channel dimension
        return anomaly_map
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """DRAEM-specific: Max pooling for image-level score."""
        return torch.amax(anomaly_map, dim=(-2, -1))

    def forward(self, batch: torch.Tensor):
        """Forward pass through both sub-networks."""
        if self.training:
            # Generate synthetic anomalies during training
            augmented_batch, anomaly_mask = self.anomaly_generator(batch)
            reconstruction = self.reconstructive_subnetwork(augmented_batch)
            concatenated_inputs = torch.cat([augmented_batch, reconstruction], axis=1)
            prediction = self.discriminative_subnetwork(concatenated_inputs)
            
            return {
                'input_image': batch,
                'augmented_image': augmented_batch,
                'reconstruction': reconstruction,
                'anomaly_mask': anomaly_mask,
                'prediction': prediction
            }
        else:
            # Inference mode
            reconstruction = self.reconstructive_subnetwork(batch)
            concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
            prediction = self.discriminative_subnetwork(concatenated_inputs)
            
            anomaly_map = self.compute_anomaly_map(batch, reconstruction, prediction)
            pred_score = self.compute_anomaly_score(anomaly_map)
            return {'pred_score': pred_score, 'anomaly_map': anomaly_map}


# ===================================================================
# DRAEM Loss Function
# ===================================================================

class DRAEMLoss(nn.Module):
    """Overall loss function of the DRAEM model."""
    
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined loss over a batch for the DRAEM model."""
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        
        # Ensure anomaly_mask is proper shape and type for focal loss
        if anomaly_mask.dim() == 4 and anomaly_mask.size(1) == 1:
            # Remove channel dimension for focal loss: [B, 1, H, W] -> [B, H, W]
            mask_for_loss = anomaly_mask.squeeze(1)
        else:
            mask_for_loss = anomaly_mask
            
        # Convert to long tensor for cross entropy
        mask_for_loss = mask_for_loss.long()
        
        focal_loss_val = self.focal_loss(prediction, mask_for_loss)
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        
        return l2_loss_val + ssim_loss_val + focal_loss_val


if __name__ == "__main__":
    # Test DRAEM model
    model = DRAEMModel(sspcab=False)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Training mode
    model.train()
    train_output = model(x)
    print("Training output keys:", train_output.keys())
    
    # Test loss
    loss_fn = DRAEMLoss()
    loss = loss_fn(
        train_output['input_image'], 
        train_output['reconstruction'], 
        train_output['anomaly_mask'],
        train_output['prediction']
    )
    print(f"Training loss: {loss.item():.4f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        inference_output = model(x)
        print(f"Inference pred_score shape: {inference_output['pred_score'].shape}")
        print(f"Inference anomaly_map shape: {inference_output['anomaly_map'].shape}")
    
    print("DRAEM model test completed successfully!")