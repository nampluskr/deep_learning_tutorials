from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2

from kornia.losses import FocalLoss, SSIMLoss

from trainer import BaseTrainer


#############################################################
# anomalib/src/anomalib/data/transforms/multi_random_choice.py
#############################################################

class MultiRandomChoice(v2.Transform):
    def __init__(
        self,
        transforms: Sequence[Callable],
        probabilities: list[float] | None = None,
        num_transforms: int = 1,
        fixed_num_transforms: bool = False,
    ) -> None:
        if not isinstance(transforms, Sequence):
            msg = "Argument transforms should be a sequence of callables"
            raise TypeError(msg)

        if probabilities is None:
            probabilities = [1.0] * len(transforms)
        elif len(probabilities) != len(transforms):
            msg = f"Length of p doesn't match the number of transforms: {len(probabilities)} != {len(transforms)}"
            raise ValueError(msg)

        super().__init__()

        self.transforms = transforms
        total = sum(probabilities)
        self.probabilities = [probability / total for probability in probabilities]

        self.num_transforms = num_transforms
        self.fixed_num_transforms = fixed_num_transforms

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # First determine number of transforms to apply
        num_transforms = (
            self.num_transforms if self.fixed_num_transforms else int(torch.randint(self.num_transforms, (1,)) + 1)
        )
        # Get transforms
        idx = torch.multinomial(torch.tensor(self.probabilities), num_transforms).tolist()
        transform = v2.Compose([self.transforms[i] for i in idx])
        return transform(*inputs)

#############################################################
# anomalib/src/anomalib/data/utils/generators/perlin.py
#############################################################

def generate_perlin_noise(
    height: int,
    width: int,
    scale: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
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


class PerlinAnomalyGenerator(v2.Transform):
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
                v2.RandomPhotometricDistort(
                    brightness=(0.8, 1.2),
                    contrast=(1.0, 1.0),  # No contrast change
                    saturation=(1.0, 1.0),  # No saturation change
                    hue=(0.0, 0.0),  # No hue change
                    p=1.0,
                ),
                v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
                v2.ColorJitter(hue=[-50 / 360, 50 / 360], saturation=[0.5, 1.5]),
                v2.RandomSolarize(threshold=torch.empty(1).uniform_(32 / 255, 128 / 255).item(), p=1.0),
                v2.RandomPosterize(bits=4, p=1.0),
                v2.RandomInvert(p=1.0),
                v2.AutoAugment(),
                v2.RandomEqualize(p=1.0),
                v2.RandomAffine(degrees=(-45, 45), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            ],
            probabilities=None,
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
        if anomaly_source_path:
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
            # Add type guard
            else torch.tensor(0.5, device=device)  # Fallback value
        )

        if not isinstance(beta, float):
            beta = beta.view(-1, 1, 1).expand_as(img)

        augmented_img = img * (1 - mask) + beta * perturbation + (1 - beta) * img * mask
        return augmented_img, mask

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


#############################################################
# anomalib\models\images\stfpm\loss.py
#############################################################

class DRAEMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l2_loss = nn.modules.loss.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        return l2_loss_val + ssim_loss_val + focal_loss_val


###########################################################
# anomalib/src/anomalib/models/components/layers/sspcab.py
###########################################################

class AttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()

        out_channels = in_channels // reduction_ratio
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # reduce feature map to 1d vector through global average pooling
        avg_pooled = inputs.mean(dim=(2, 3))

        # squeeze and excite
        act = self.fc1(avg_pooled)
        act = F.relu(act)
        act = self.fc2(act)
        act = F.sigmoid(act)

        # multiply with input
        return inputs * act.view(act.shape[0], act.shape[1], 1, 1)


class SSPCAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.pad = kernel_size + dilation
        self.crop = kernel_size + 2 * dilation + 1

        self.masked_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )

        self.attention_module = AttentionModule(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # compute masked convolution
        padded = F.pad(inputs, (self.pad,) * 4)
        masked_out = torch.zeros_like(inputs)
        masked_out += self.masked_conv1(padded[..., : -self.crop, : -self.crop])
        masked_out += self.masked_conv2(padded[..., : -self.crop, self.crop :])
        masked_out += self.masked_conv3(padded[..., self.crop :, : -self.crop])
        masked_out += self.masked_conv4(padded[..., self.crop :, self.crop :])

        # apply channel attention module
        return self.attention_module(masked_out)


###########################################################
# anomalib\models\images\draem\torch_model.py
###########################################################


class DRAEM(nn.Module):
    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        self.sspcab = sspcab
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

    def forward(self, batch: torch.Tensor):
        reconstruction = self.reconstructive_subnetwork(batch)
        concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
        prediction = self.discriminative_subnetwork(concatenated_inputs)
        return reconstruction, prediction

    def predict(self, batch: torch.Tensor):
        reconstruction = self.reconstructive_subnetwork(batch)
        concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
        prediction = self.discriminative_subnetwork(concatenated_inputs)

        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

class ReconstructiveSubNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_width: int = 128,
        sspcab: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64) -> None:
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        return self.decoder_segment(act1, act2, act3, act4, act5, act6)


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
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

    def forward(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        act5 = self.block5(mp4)
        mp5 = self.mp5(act5)
        act6 = self.block6(mp5)
        return act1, act2, act3, act4, act5, act6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
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


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels: int, base_width: int, sspcab: bool = False) -> None:
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
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        return self.block5(mp4)


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
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
        # cat with base*1
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

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, act5: torch.Tensor) -> torch.Tensor:
        up1 = self.up1(act5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        return self.fin_out(db4)

#############################################################
# Trainer for DRAEM Model
#############################################################

class DRAEMTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if loss_fn is None:
            loss_fn = DRAEMLoss()
        super().__init__(model, optimizer, loss_fn, metrics, device)

        self.dtd_dir = "/mnt/d/datasets/dtd"
        self.enable_sspcab = model.sspcab
        self.sspcab_lambda = 0.1
        self.epoch_period = 5

        self.augmenter = PerlinAnomalyGenerator(
            anomaly_source_path=self.dtd_dir,
            blend_factor=(0.1, 1.0)
        )

        if self.enable_sspcab:
            self.sspcab_activations = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()

    def setup_sspcab(self):
        def get_activation(name):
            def hook(module, input, output):
                self.sspcab_activations[name] = output
            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))


    @torch.enable_grad()
    def train_step(self, batch):
        images = batch["image"].to(self.device)

        # 1. Anomaly generation
        augmented_images, anomaly_masks = self.augmenter(images)
        augmented_images = augmented_images.to(self.device)
        anomaly_masks = anomaly_masks.to(self.device)

        # 2. Forward pass
        self.optimizer.zero_grad()
        reconstruction, prediction = self.model(augmented_images)

        # 3. Loss 계산
        loss = self.loss_fn(images, reconstruction, anomaly_masks, prediction)

        # 4. SSPCAB loss 추가 (옵션)
        if self.enable_sspcab:
            sspcab_loss = self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"]
            )
            loss = loss + self.sspcab_lambda * sspcab_loss

        # 5. Backward pass
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}


if __name__ == "__main__":

    pass