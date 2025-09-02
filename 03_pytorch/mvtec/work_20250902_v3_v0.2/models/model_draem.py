import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
import random
import numpy as np
from kornia.losses import FocalLoss, SSIMLoss


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


class AttentionModule(nn.Module):
    """Squeeze and excitation block that acts as the attention module in SSPCAB."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        out_channels = in_channels // reduction_ratio
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention module."""
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
    """Self-Supervised Predictive Convolutional Attention Block."""
    
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
        """Forward pass through the SSPCAB block."""
        # compute masked convolution
        padded = F.pad(inputs, (self.pad,) * 4)
        masked_out = torch.zeros_like(inputs)
        masked_out += self.masked_conv1(padded[..., : -self.crop, : -self.crop])
        masked_out += self.masked_conv2(padded[..., : -self.crop, self.crop :])
        masked_out += self.masked_conv3(padded[..., self.crop :, : -self.crop])
        masked_out += self.masked_conv4(padded[..., self.crop :, self.crop :])

        # apply channel attention module
        return self.attention_module(masked_out)


class EncoderReconstructive(nn.Module):
    """Encoder component of the reconstructive network."""
    
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
    ) -> None:
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct input images."""
        encoded = self.encoder(batch)
        return self.decoder(encoded)


class EncoderDiscriminative(nn.Module):
    """Encoder component of the discriminator network."""
    
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
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64) -> None:
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Generate predicted anomaly masks."""
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        return self.decoder_segment(act1, act2, act3, act4, act5, act6)


class SimpleAnomalyGenerator(nn.Module):
    """Simple anomaly generator for DRAEM training."""
    
    def __init__(self, image_size=(256, 256), beta_range=(0.1, 1.0)):
        super().__init__()
        self.image_size = image_size
        self.beta_range = beta_range
    
    def generate_perlin_like_noise(self, batch_size, device):
        """Generate simple noise patterns similar to Perlin noise."""
        noise = torch.randn(batch_size, 1, self.image_size[0] // 8, self.image_size[1] // 8, device=device)
        
        # Smooth the noise
        for _ in range(3):
            noise = F.avg_pool2d(F.pad(noise, (1, 1, 1, 1), mode='reflect'), 3, stride=1, padding=0)
        
        # Upsample to target size
        noise = F.interpolate(noise, size=self.image_size, mode='bilinear', align_corners=False)
        return noise
    
    def generate_blob_mask(self, batch_size, device):
        """Generate blob-like anomaly masks."""
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(1, self.image_size[0], self.image_size[1], device=device)
            
            # Random number of blobs
            num_blobs = random.randint(1, 5)
            for _ in range(num_blobs):
                # Random blob parameters
                center_x = random.randint(self.image_size[1] // 4, 3 * self.image_size[1] // 4)
                center_y = random.randint(self.image_size[0] // 4, 3 * self.image_size[0] // 4)
                radius = random.randint(20, 80)
                
                # Create circular mask
                y, x = torch.meshgrid(
                    torch.arange(self.image_size[0], device=device),
                    torch.arange(self.image_size[1], device=device),
                    indexing='ij'
                )
                dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                blob_mask = (dist <= radius).float()
                
                # Add some randomness
                blob_mask *= torch.rand_like(blob_mask) * 0.5 + 0.5
                
                mask = torch.maximum(mask, blob_mask.unsqueeze(0))
            
            masks.append(mask)
        
        return torch.stack(masks)
    
    def forward(self, images):
        """Generate anomaly augmented images and masks."""
        batch_size = images.shape[0]
        device = images.device
        
        # Choose random augmentation strategy
        if random.random() > 0.5:
            anomaly_masks = self.generate_perlin_like_noise(batch_size, device)
        else:
            anomaly_masks = self.generate_blob_mask(batch_size, device)
        
        # Threshold masks to binary
        anomaly_masks = (anomaly_masks > 0.3).float()
        
        # Random beta factor for blending
        if isinstance(self.beta_range, tuple):
            beta = torch.rand(batch_size, 1, 1, 1, device=device) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        else:
            beta = self.beta_range
        
        # Apply anomalies
        augmented_images = images * (1 - anomaly_masks) + torch.rand_like(images) * anomaly_masks * beta
        augmented_images = torch.clamp(augmented_images, 0, 1)
        
        return augmented_images, anomaly_masks


class DraemModel(nn.Module):
    """DRAEM PyTorch model with reconstructive and discriminative sub-networks."""
    
    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        self.anomaly_generator = SimpleAnomalyGenerator()

    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | InferenceBatch:
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
            
            anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
            return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


class DraemLoss(nn.Module):
    """Overall loss function of the DRAEM model."""
    
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
        """Compute the combined loss over a batch for the DRAEM model."""
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        return l2_loss_val + ssim_loss_val + focal_loss_val


if __name__ == "__main__":
    # Test DRAEM model
    model = DraemModel(sspcab=False)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Training mode
    model.train()
    train_output = model(x)
    print("Training output keys:", train_output.keys())
    
    # Test loss
    loss_fn = DraemLoss()
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
        print(f"Inference pred_score shape: {inference_output.pred_score.shape}")
        print(f"Inference anomaly_map shape: {inference_output.anomaly_map.shape}")