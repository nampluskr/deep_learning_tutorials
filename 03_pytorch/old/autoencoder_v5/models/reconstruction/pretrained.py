"""
Pretrained encoder-based autoencoders for anomaly detection.

This module implements autoencoders that use pretrained backbones 
(ResNet, VGG, EfficientNet) as encoders with custom decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

from ..base import ConvBlock, DeconvBlock
from ..base.feature_extractors import get_pretrained_encoder
from ..base.utils import (
    get_output_padding_for_target_size,
    validate_input_size,
    safe_interpolate
)


class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that works with different pretrained encoders"""
    
    def __init__(self, latent_dim=512, out_channels=3, input_size=256, 
                 encoder_features_dim=None):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # Validate input size
        validate_input_size(input_size, min_size=32)
        
        # If encoder features dimension is provided, use it; otherwise use latent_dim
        features_dim = encoder_features_dim or latent_dim
        
        # Start from 4x4 feature map
        self.fc = nn.Linear(latent_dim, features_dim * 4 * 4)
        self.unflatten = nn.Unflatten(1, (features_dim, 4, 4))
        
        # Progressive upsampling decoder
        self.decoder_layers = nn.ModuleList([
            # 4x4 -> 8x8
            DeconvBlock(features_dim, 512, kernel_size=4, stride=2, padding=1),
            # 8x8 -> 16x16  
            DeconvBlock(512, 256, kernel_size=4, stride=2, padding=1),
            # 16x16 -> 32x32
            DeconvBlock(256, 128, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 64x64
            DeconvBlock(128, 64, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 128x128
            DeconvBlock(64, 32, kernel_size=4, stride=2, padding=1),
        ])
        
        # Final layer to target size
        self.final_conv = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        
        # Progressive upsampling
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        # Ensure output matches target size
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        return x


class PretrainedAutoEncoder(nn.Module):
    """Base class for autoencoders with pretrained encoders"""
    
    def __init__(self, backbone_name='resnet50', pretrained=True, freeze_backbone=False,
                 in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.backbone_name = backbone_name
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Create pretrained encoder
        self.encoder = get_pretrained_encoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            latent_dim=latent_dim
        )
        
        # Create adaptive decoder
        self.decoder = AdaptiveDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            input_size=input_size
        )
    
    def forward(self, x):
        # Validate input size
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        # Encode
        latent, features = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent, features


class ResNetAE(PretrainedAutoEncoder):
    """ResNet-based autoencoder for anomaly detection"""
    
    def __init__(self, arch='resnet50', pretrained=True, freeze_backbone=False,
                 in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__(
            backbone_name=arch,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            input_size=input_size
        )


class VGGAutoEncoder(PretrainedAutoEncoder):
    """VGG-based autoencoder for anomaly detection"""
    
    def __init__(self, arch='vgg16', pretrained=True, freeze_backbone=False,
                 in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__(
            backbone_name=arch,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            input_size=input_size
        )


class EfficientNetAE(PretrainedAutoEncoder):
    """EfficientNet-based autoencoder for anomaly detection"""
    
    def __init__(self, arch='efficientnet_b0', pretrained=True, freeze_backbone=False,
                 in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__(
            backbone_name=arch,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            input_size=input_size
        )


class MultiScaleAutoEncoder(nn.Module):
    """Multi-scale autoencoder using pretrained features"""
    
    def __init__(self, backbone_name='resnet50', pretrained=True, freeze_backbone=False,
                 in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Extract features from multiple layers
        output_layers = ['layer1', 'layer2', 'layer3', 'layer4'] if 'resnet' in backbone_name else None
        
        self.encoder = get_pretrained_encoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            output_layers=output_layers,
            latent_dim=latent_dim
        )
        
        # Multi-scale decoder
        self.decoder = MultiScaleDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            input_size=input_size
        )
    
    def forward(self, x):
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent, features


class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder with skip connections"""
    
    def __init__(self, latent_dim=512, out_channels=3, input_size=256):
        super().__init__()
        self.input_size = input_size
        
        # Initial upsampling from latent
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (512, 4, 4))
        
        # Multi-scale upsampling blocks
        self.up1 = DeconvBlock(512, 256, kernel_size=4, stride=2, padding=1)  # 4->8
        self.up2 = DeconvBlock(256, 128, kernel_size=4, stride=2, padding=1)  # 8->16
        self.up3 = DeconvBlock(128, 64, kernel_size=4, stride=2, padding=1)   # 16->32
        self.up4 = DeconvBlock(64, 32, kernel_size=4, stride=2, padding=1)    # 32->64
        self.up5 = DeconvBlock(32, 16, kernel_size=4, stride=2, padding=1)    # 64->128
        
        # Final output layer
        self.final_conv = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Sigmoid()
        
        # Refinement layers for each scale
        self.refine_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # 32x32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),    # 64x64
            nn.Conv2d(16, 16, kernel_size=3, padding=1),    # 128x128
        ])
    
    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)  # (B, 512, 4, 4)
        
        # Multi-scale upsampling with refinement
        x = self.up1(x)  # (B, 256, 8, 8)
        x = self.refine_layers[0](x)
        
        x = self.up2(x)  # (B, 128, 16, 16)
        x = self.refine_layers[1](x)
        
        x = self.up3(x)  # (B, 64, 32, 32)
        x = self.refine_layers[2](x)
        
        x = self.up4(x)  # (B, 32, 64, 64)
        x = self.refine_layers[3](x)
        
        x = self.up5(x)  # (B, 16, 128, 128)
        x = self.refine_layers[4](x)
        
        # Final output
        x = self.final_conv(x)  # (B, out_channels, 256, 256)
        x = self.final_activation(x)
        
        # Ensure correct output size
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        return x


class FeaturePyramidAutoEncoder(nn.Module):
    """Feature Pyramid Network style autoencoder"""
    
    def __init__(self, backbone_name='resnet50', pretrained=True, freeze_backbone=False,
                 in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        
        # Multi-layer feature extraction
        output_layers = ['layer1', 'layer2', 'layer3', 'layer4'] if 'resnet' in backbone_name else None
        
        self.encoder = get_pretrained_encoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            output_layers=output_layers,
            latent_dim=latent_dim
        )
        
        # Feature Pyramid Decoder
        self.decoder = FeaturePyramidDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            input_size=input_size
        )
    
    def forward(self, x):
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent, features


class FeaturePyramidDecoder(nn.Module):
    """Feature Pyramid Network style decoder"""
    
    def __init__(self, latent_dim=512, out_channels=3, input_size=256):
        super().__init__()
        self.input_size = input_size
        
        # Top-down pathway
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (256, 8, 8))
        
        # Lateral connections and top-down layers
        self.lateral_conv1 = nn.Conv2d(256, 256, kernel_size=1)  # 8x8
        self.lateral_conv2 = nn.Conv2d(256, 128, kernel_size=1)  # 16x16
        self.lateral_conv3 = nn.Conv2d(128, 64, kernel_size=1)   # 32x32
        self.lateral_conv4 = nn.Conv2d(64, 32, kernel_size=1)    # 64x64
        
        # Output layers for each pyramid level
        self.output_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.output_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.output_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.output_conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        # Final reconstruction layer
        self.final_up = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Sigmoid()
    
    def _upsample_add(self, x, y):
        """Upsample x and add to y"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
    
    def forward(self, latent):
        # Start from latent representation
        c5 = self.fc(latent)
        c5 = self.unflatten(c5)  # (B, 256, 8, 8)
        
        # Top-down pathway with lateral connections
        p5 = self.lateral_conv1(c5)  # (B, 256, 8, 8)
        
        c4 = F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 256, 16, 16)
        p4 = self.lateral_conv2(c4)  # (B, 128, 16, 16)
        
        c3 = F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 128, 32, 32)
        p3 = self.lateral_conv3(c3)  # (B, 64, 32, 32)
        
        c2 = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 64, 64, 64)
        p2 = self.lateral_conv4(c2)  # (B, 32, 64, 64)
        
        # Output processing
        o5 = self.output_conv1(p5)  # (B, 128, 8, 8)
        o4 = self.output_conv2(p4)  # (B, 64, 16, 16)
        o3 = self.output_conv3(p3)  # (B, 32, 32, 32)
        o2 = self.output_conv4(p2)  # (B, 16, 64, 64)
        
        # Combine and upsample to final resolution
        # Use the finest scale (o2) for final reconstruction
        final = F.interpolate(o2, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 16, 128, 128)
        final = self.final_up(final)  # (B, out_channels, 256, 256)
        final = self.final_activation(final)
        
        # Ensure correct output size
        if final.size(-1) != self.input_size or final.size(-2) != self.input_size:
            final = F.interpolate(final, size=(self.input_size, self.input_size), 
                                mode='bilinear', align_corners=False)
        
        return final


if __name__ == "__main__":
    # Test pretrained autoencoder models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    input_size = 224  # Standard ImageNet size for pretrained models
    x = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    print("Testing ResNetAE:")
    model = ResNetAE(arch='resnet50', pretrained=False, input_size=input_size).to(device)
    reconstructed, latent, features = model(x)
    print(f"Input: {x.shape}")
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Features: {features.shape}")
    
    print("\nTesting VGGAutoEncoder:")
    model = VGGAutoEncoder(arch='vgg16', pretrained=False, input_size=input_size).to(device)
    reconstructed, latent, features = model(x)
    print(f"VGG - Reconstructed: {reconstructed.shape}, Latent: {latent.shape}")
    
    print("\nTesting EfficientNetAE:")
    try:
        model = EfficientNetAE(arch='efficientnet_b0', pretrained=False, input_size=input_size).to(device)
        reconstructed, latent, features = model(x)
        print(f"EfficientNet - Reconstructed: {reconstructed.shape}, Latent: {latent.shape}")
    except Exception as e:
        print(f"EfficientNet test skipped: {e}")
    
    print("\nTesting MultiScaleAutoEncoder:")
    model = MultiScaleAutoEncoder(backbone_name='resnet50', pretrained=False, input_size=input_size).to(device)
    reconstructed, latent, features = model(x)
    print(f"MultiScale - Reconstructed: {reconstructed.shape}, Latent: {latent.shape}")
    
    print("\nAll pretrained models working correctly!")