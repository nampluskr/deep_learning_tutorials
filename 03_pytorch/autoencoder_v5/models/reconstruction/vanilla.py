"""
Vanilla autoencoder models for anomaly detection.

This module implements basic autoencoder architectures including 
standard encoder-decoder and UNet-style with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ConvBlock, DeconvBlock
from ..base.utils import (
    calculate_conv_output_size, 
    calculate_deconv_output_size,
    get_final_conv_size,
    get_output_padding_for_target_size,
    safe_concatenate,
    validate_input_size
)


class VanillaEncoder(nn.Module):
    """Enhanced encoder with dynamic spatial size handling"""
    
    def __init__(self, in_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Validate input size
        validate_input_size(input_size, min_size=32)

        # Calculate final feature map size
        self.final_size = get_final_conv_size(input_size, num_layers=5)
        if self.final_size <= 0:
            raise ValueError(f"Input size {input_size} is too small. Minimum size is 32.")

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),      # input_size -> input_size/2
            ConvBlock(32, 64),               # -> input_size/4
            ConvBlock(64, 128),              # -> input_size/8
            ConvBlock(128, 256),             # -> input_size/16
            ConvBlock(256, 512),             # -> input_size/32
        )

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Always output 4x4
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.adaptive_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        return latent, features


class VanillaDecoder(nn.Module):
    """Enhanced decoder with dynamic spatial size handling"""
    
    def __init__(self, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Validate input size
        validate_input_size(input_size, min_size=32)

        # Calculate target sizes for reconstruction
        self.final_size = get_final_conv_size(input_size, num_layers=5)
        if self.final_size <= 0:
            raise ValueError(f"Input size {input_size} is too small. Minimum size is 32.")

        # Use adaptive approach - start from fixed 4x4
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (512, 4, 4))

        # Calculate output padding to reach exact target sizes
        target_sizes = []
        size = input_size
        for _ in range(5):
            size = size // 2
            target_sizes.append(size)
        target_sizes.reverse()  # [8, 16, 32, 64, 128] for 256 input

        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(512, 256, output_padding=self._get_output_padding(4, target_sizes[0])),     
            DeconvBlock(256, 128, output_padding=self._get_output_padding(target_sizes[0], target_sizes[1])),  
            DeconvBlock(128, 64, output_padding=self._get_output_padding(target_sizes[1], target_sizes[2])),   
            DeconvBlock(64, 32, output_padding=self._get_output_padding(target_sizes[2], target_sizes[3])),    
        ])
        
        # Final layer to get exact output size
        final_output_padding = self._get_output_padding(target_sizes[3], input_size)
        self.final_conv = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, 
                                           padding=1, output_padding=final_output_padding)
        self.final_activation = nn.Sigmoid()

    def _get_output_padding(self, input_size, target_size):
        """Calculate output padding to reach target size"""
        return get_output_padding_for_target_size(input_size, target_size)

    def forward(self, latent):
        x = self.fc(latent)                     # (B, 512 * 4 * 4)
        x = self.unflatten(x)                   # (B, 512, 4, 4)
        
        for deconv_block in self.deconv_blocks:
            x = deconv_block(x)
            
        x = self.final_conv(x)
        reconstructed = self.final_activation(x)
        return reconstructed


class VanillaAE(nn.Module):
    """Enhanced vanilla autoencoder with dynamic sizing support"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.encoder = VanillaEncoder(in_channels, latent_dim, input_size)
        self.decoder = VanillaDecoder(out_channels, latent_dim, input_size)

    def forward(self, x):
        # Validate input size
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            print(f"Warning: Expected input size {self.input_size}x{self.input_size}, "
                  f"got {x.size(-2)}x{x.size(-1)}. Resizing...")
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


class UnetEncoder(nn.Module):
    """Enhanced UNet-style encoder with dynamic sizing and safe skip connections"""
    
    def __init__(self, in_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size

        # Validate input size
        validate_input_size(input_size, min_size=32)

        # Calculate expected sizes after each conv layer for safety checks
        self.expected_sizes = []
        size = input_size
        for _ in range(5):
            size = calculate_conv_output_size(size, kernel_size=4, stride=2, padding=1)
            self.expected_sizes.append(size)

        # Encoder blocks
        self.conv1 = ConvBlock(in_channels, 32)    
        self.conv2 = ConvBlock(32, 64)             
        self.conv3 = ConvBlock(64, 128)            
        self.conv4 = ConvBlock(128, 256)           
        self.conv5 = ConvBlock(256, 512)           

        # Adaptive pooling for consistent latent representation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        # Forward pass with skip connections
        e1 = self.conv1(x)      # 32 x H/2 x W/2
        e2 = self.conv2(e1)     # 64 x H/4 x W/4
        e3 = self.conv3(e2)     # 128 x H/8 x W/8
        e4 = self.conv4(e3)     # 256 x H/16 x W/16
        e5 = self.conv5(e4)     # 512 x H/32 x W/32

        # Create latent representation
        pooled = self.adaptive_pool(e5)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)

        # Return latent and skip connections (in reverse order for decoder)
        skip_connections = [e1, e2, e3, e4]
        return latent, e5, skip_connections


class UnetDecoder(nn.Module):
    """Enhanced UNet-style decoder with safe skip connections and dynamic sizing"""
    
    def __init__(self, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size

        # Validate input size
        validate_input_size(input_size, min_size=32)

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (512, 4, 4))

        # Decoder blocks with adaptive sizing
        self.deconv1 = DeconvBlock(512, 256)                    
        self.deconv2 = DeconvBlock(256 + 256, 128)             # +256 for skip connection
        self.deconv3 = DeconvBlock(128 + 128, 64)              # +128 for skip connection
        self.deconv4 = DeconvBlock(64 + 64, 32)                # +64 for skip connection
        
        # Final layer with skip connection
        self.final_conv = nn.ConvTranspose2d(32 + 32, out_channels, 
                                           kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, latent, skip_connections):
        x = self.fc(latent)
        x = self.unflatten(x)  # (B, 512, 4, 4)

        # Decoder with safe skip connections
        d1 = self.deconv1(x)                                    # 256 x H/16 x W/16
        d2 = self.deconv2(safe_concatenate(d1, skip_connections[3]))  # 128 x H/8 x W/8
        d3 = self.deconv3(safe_concatenate(d2, skip_connections[2]))  # 64 x H/4 x W/4
        d4 = self.deconv4(safe_concatenate(d3, skip_connections[1]))  # 32 x H/2 x W/2
        d5 = self.final_conv(safe_concatenate(d4, skip_connections[0]))  # out_channels x H x W

        reconstructed = self.final_activation(d5)
        
        # Ensure output matches input size
        if reconstructed.size(-1) != self.input_size or reconstructed.size(-2) != self.input_size:
            reconstructed = F.interpolate(reconstructed, size=(self.input_size, self.input_size), 
                                        mode='bilinear', align_corners=False)
        
        return reconstructed


class UnetAE(nn.Module):
    """Enhanced UNet-style autoencoder with safe skip connections and dynamic sizing"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.encoder = UnetEncoder(in_channels, latent_dim, input_size)
        self.decoder = UnetDecoder(out_channels, latent_dim, input_size)

    def forward(self, x):
        # Validate and resize input if necessary
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            print(f"Warning: Expected input size {self.input_size}x{self.input_size}, "
                  f"got {x.size(-2)}x{x.size(-1)}. Resizing...")
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return reconstructed, latent, features


if __name__ == "__main__":
    # Test vanilla autoencoder models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing VanillaAE:")
    model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512, input_size=256).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    reconstructed, latent, features = model(x)
    print(f"Input: {x.shape}")
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Features: {features.shape}")
    
    print("\nTesting UnetAE:")
    model = UnetAE(in_channels=3, out_channels=3, latent_dim=512, input_size=256).to(device)
    reconstructed, latent, features = model(x)
    print(f"Input: {x.shape}")
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Features: {features.shape}")
    
    # Test different input sizes
    print("\nTesting different input sizes:")
    for input_size in [128, 256, 512]:
        print(f"\nInput size: {input_size}x{input_size}")
        model = VanillaAE(input_size=input_size).to(device)
        x = torch.randn(1, 3, input_size, input_size).to(device)
        reconstructed, latent, features = model(x)
        print(f"  Input: {x.shape} -> Reconstructed: {reconstructed.shape}")
    
    print("\nAll vanilla models working correctly!")