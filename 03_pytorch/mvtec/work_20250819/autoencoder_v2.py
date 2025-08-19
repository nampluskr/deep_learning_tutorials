import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# =============================================================================
# Enhanced Building Blocks for Multi-Resolution Support
# =============================================================================

class AdaptiveConvBlock(nn.Module):
    """Adaptive convolution block that handles variable input sizes"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation='leaky_relu', adaptive=False):
        super().__init__()
        
        self.adaptive = adaptive
        layers = []
        
        if adaptive:
            # Use adaptive padding for better size handling
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        if self.adaptive:
            # Calculate adaptive padding
            kernel_size = self.conv_block[0].kernel_size[0]
            stride = self.conv_block[0].stride[0]
            h, w = x.shape[-2:]
            
            pad_h = max(0, (stride - h % stride) % stride)
            pad_w = max(0, (stride - w % stride) % stride)
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        
        return self.conv_block(x)


class AdaptiveDeconvBlock(nn.Module):
    """Adaptive deconvolution block with flexible upsampling"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation='relu', dropout=False, target_size=None):
        super().__init__()
        
        self.target_size = target_size
        self.stride = stride
        
        if target_size is not None:
            # Use interpolation + conv for precise size control
            self.upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            # Traditional transposed convolution
            self.upsample = None
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
        layers = []
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if dropout:
            layers.append(nn.Dropout2d(0.5))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())

        self.post_conv = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        
        return self.post_conv(x)


# =============================================================================
# Adaptive Vanilla AutoEncoder
# =============================================================================

class AdaptiveVanillaEncoder(nn.Module):
    """Adaptive encoder that handles variable input resolutions"""

    def __init__(self, in_channels=3, latent_dim=512, min_size=16):
        super().__init__()
        
        self.min_size = min_size
        self.latent_dim = latent_dim
        
        # Progressive downsampling layers
        self.conv1 = AdaptiveConvBlock(in_channels, 32, adaptive=True)     # /2
        self.conv2 = AdaptiveConvBlock(32, 64, adaptive=True)              # /4
        self.conv3 = AdaptiveConvBlock(64, 128, adaptive=True)             # /8
        self.conv4 = AdaptiveConvBlock(128, 256, adaptive=True)            # /16
        self.conv5 = AdaptiveConvBlock(256, 512, adaptive=True)            # /32
        
        # Additional layers for very large inputs
        self.conv6 = AdaptiveConvBlock(512, 512, adaptive=True)            # /64
        self.conv7 = AdaptiveConvBlock(512, 512, adaptive=True)            # /128
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        input_size = x.shape[-1]
        features = []
        
        # Progressive downsampling
        x = self.conv1(x)
        features.append(x)
        
        x = self.conv2(x)
        features.append(x)
        
        x = self.conv3(x)
        features.append(x)
        
        x = self.conv4(x)
        features.append(x)
        
        x = self.conv5(x)
        features.append(x)
        
        # Additional downsampling for very large inputs
        if input_size >= 512:
            x = self.conv6(x)
            features.append(x)
            
        if input_size >= 1024:
            x = self.conv7(x)
            features.append(x)
        
        # Global pooling and latent generation
        pooled = self.adaptive_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        
        return latent, x, features


class AdaptiveVanillaDecoder(nn.Module):
    """Adaptive decoder that reconstructs to target resolution"""

    def __init__(self, out_channels=3, latent_dim=512, target_size=256):
        super().__init__()
        
        self.target_size = target_size
        self.out_channels = out_channels
        
        # Calculate initial feature map size
        self.init_size = max(4, target_size // 32)
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        self.unflatten = nn.Unflatten(1, (512, self.init_size, self.init_size))
        
        # Progressive upsampling with target size calculation
        sizes = self._calculate_upsample_sizes(target_size)
        
        self.deconv1 = AdaptiveDeconvBlock(512, 256, target_size=sizes[0])
        self.deconv2 = AdaptiveDeconvBlock(256, 128, target_size=sizes[1])
        self.deconv3 = AdaptiveDeconvBlock(128, 64, target_size=sizes[2])
        self.deconv4 = AdaptiveDeconvBlock(64, 32, target_size=sizes[3])
        
        # Final layer to exact target size
        self.final_conv = nn.Conv2d(32, out_channels, 3, 1, 1)
        self.final_upsample = nn.Upsample(size=(target_size, target_size), 
                                         mode='bilinear', align_corners=False)
        self.final_activation = nn.Sigmoid()

    def _calculate_upsample_sizes(self, target_size):
        """Calculate intermediate sizes for progressive upsampling"""
        current_size = self.init_size
        sizes = []
        
        while current_size < target_size:
            current_size = min(current_size * 2, target_size)
            sizes.append((current_size, current_size))
            
        # Ensure we have exactly 4 sizes
        while len(sizes) < 4:
            sizes.insert(0, sizes[0])
        
        return sizes[:4]

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        # Final upsampling to exact target size
        x = self.final_conv(x)
        x = self.final_upsample(x)
        reconstructed = self.final_activation(x)
        
        return reconstructed


class AdaptiveVanillaAE(nn.Module):
    """Adaptive Vanilla AutoEncoder for variable resolutions"""

    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, target_size=256):
        super().__init__()
        self.target_size = target_size
        self.encoder = AdaptiveVanillaEncoder(in_channels, latent_dim)
        self.decoder = AdaptiveVanillaDecoder(out_channels, latent_dim, target_size)

    def forward(self, x):
        latent, features, skip_features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


# =============================================================================
# Multi-Scale UNet AutoEncoder
# =============================================================================

class MultiScaleUnetEncoder(nn.Module):
    """Multi-scale UNet encoder with pyramid features"""

    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        
        # Multi-scale input processing
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 16, 3, 1, 1),  # 1x scale
            nn.Conv2d(in_channels, 16, 3, 1, 1),  # 0.5x scale  
            nn.Conv2d(in_channels, 16, 3, 1, 1),  # 0.25x scale
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(48, 32, 3, 1, 1)
        
        # Progressive downsampling
        self.conv1 = AdaptiveConvBlock(32, 64, adaptive=True)
        self.conv2 = AdaptiveConvBlock(64, 128, adaptive=True)
        self.conv3 = AdaptiveConvBlock(128, 256, adaptive=True)
        self.conv4 = AdaptiveConvBlock(256, 512, adaptive=True)
        self.conv5 = AdaptiveConvBlock(512, 512, adaptive=True)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Multi-scale processing
        scale_features = []
        
        # Full scale
        feat1 = self.scale_convs[0](x)
        scale_features.append(feat1)
        
        # Half scale
        x_half = F.interpolate(x, size=(h//2, w//2), mode='bilinear', align_corners=False)
        feat2 = self.scale_convs[1](x_half)
        feat2 = F.interpolate(feat2, size=(h, w), mode='bilinear', align_corners=False)
        scale_features.append(feat2)
        
        # Quarter scale
        x_quarter = F.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=False)
        feat3 = self.scale_convs[2](x_quarter)
        feat3 = F.interpolate(feat3, size=(h, w), mode='bilinear', align_corners=False)
        scale_features.append(feat3)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_features, dim=1)
        x = self.fusion_conv(fused)
        
        # Progressive encoding with skip connections
        skip_connections = []
        
        x = self.conv1(x)
        skip_connections.append(x)
        
        x = self.conv2(x)
        skip_connections.append(x)
        
        x = self.conv3(x)
        skip_connections.append(x)
        
        x = self.conv4(x)
        skip_connections.append(x)
        
        x = self.conv5(x)
        
        # Global features
        pooled = self.adaptive_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        
        return latent, x, skip_connections


class MultiScaleUnetDecoder(nn.Module):
    """Multi-scale UNet decoder with adaptive upsampling"""

    def __init__(self, out_channels=3, latent_dim=512, target_size=256):
        super().__init__()
        
        self.target_size = target_size
        self.init_size = max(4, target_size // 32)
        
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        self.unflatten = nn.Unflatten(1, (512, self.init_size, self.init_size))
        
        # Decoder with skip connections
        self.deconv1 = AdaptiveDeconvBlock(512, 512)
        self.deconv2 = AdaptiveDeconvBlock(512 + 512, 256)  # with skip
        self.deconv3 = AdaptiveDeconvBlock(256 + 256, 128)  # with skip
        self.deconv4 = AdaptiveDeconvBlock(128 + 128, 64)   # with skip
        self.deconv5 = AdaptiveDeconvBlock(64 + 64, 32)     # with skip
        
        # Multi-scale output heads
        self.output_heads = nn.ModuleList([
            nn.Conv2d(32, out_channels, 3, 1, 1),  # Main output
            nn.Conv2d(64, out_channels, 3, 1, 1),  # Auxiliary output 1
            nn.Conv2d(128, out_channels, 3, 1, 1), # Auxiliary output 2
        ])
        
        self.final_upsample = nn.Upsample(size=(target_size, target_size), 
                                         mode='bilinear', align_corners=False)
        self.final_activation = nn.Sigmoid()

    def forward(self, latent, skip_connections):
        x = self.fc(latent)
        x = self.unflatten(x)
        
        x = self.deconv1(x)
        x = self.deconv2(torch.cat([x, skip_connections[3]], dim=1))
        aux_out2 = self.output_heads[2](x)
        
        x = self.deconv3(torch.cat([x, skip_connections[2]], dim=1))
        aux_out1 = self.output_heads[1](x)
        
        x = self.deconv4(torch.cat([x, skip_connections[1]], dim=1))
        x = self.deconv5(torch.cat([x, skip_connections[0]], dim=1))
        
        # Main output
        main_out = self.output_heads[0](x)
        
        # Upsample all outputs to target size
        main_out = self.final_upsample(main_out)
        aux_out1 = self.final_upsample(aux_out1)
        aux_out2 = self.final_upsample(aux_out2)
        
        # Apply final activation
        main_out = self.final_activation(main_out)
        aux_out1 = self.final_activation(aux_out1)
        aux_out2 = self.final_activation(aux_out2)
        
        return main_out, aux_out1, aux_out2


class MultiScaleUnetAE(nn.Module):
    """Multi-scale UNet AutoEncoder with auxiliary outputs"""

    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, target_size=256):
        super().__init__()
        self.target_size = target_size
        self.encoder = MultiScaleUnetEncoder(in_channels, latent_dim)
        self.decoder = MultiScaleUnetDecoder(out_channels, latent_dim, target_size)

    def forward(self, x):
        latent, features, skip_connections = self.encoder(x)
        main_out, aux_out1, aux_out2 = self.decoder(latent, skip_connections)
        
        return main_out, latent, features, aux_out1, aux_out2


# =============================================================================
# Patch-based AutoEncoder for Memory Efficiency
# =============================================================================

class PatchBasedEncoder(nn.Module):
    """Patch-based encoder for handling very large images"""

    def __init__(self, in_channels=3, latent_dim=512, patch_size=256, overlap=32):
        super().__init__()
        
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
        # Patch encoder (similar to vanilla encoder)
        self.patch_encoder = AdaptiveVanillaEncoder(in_channels, latent_dim//4)
        
        # Global aggregation
        self.global_conv = nn.Conv2d(latent_dim//4, latent_dim//2, 3, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(latent_dim//2, latent_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Extract patches
        patches = []
        patch_positions = []
        
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
                patch_positions.append((i, j))
        
        # Handle edge cases
        if len(patches) == 0:
            # Image smaller than patch size, use direct processing
            return self.patch_encoder(x)
        
        # Process patches
        patch_features = []
        for patch in patches:
            latent, features, _ = self.patch_encoder(patch)
            patch_features.append(latent)
        
        # Aggregate patch features
        patch_tensor = torch.stack(patch_features, dim=1)  # [B, num_patches, latent_dim//4]
        
        # Global aggregation
        global_latent = torch.mean(patch_tensor, dim=1)  # [B, latent_dim//4]
        global_latent = self.global_fc(global_latent)    # [B, latent_dim]
        
        return global_latent, patch_tensor, patch_positions


class PatchBasedDecoder(nn.Module):
    """Patch-based decoder for reconstructing large images"""

    def __init__(self, out_channels=3, latent_dim=512, target_size=512, patch_size=256):
        super().__init__()
        
        self.target_size = target_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # Patch decoder
        self.patch_decoder = AdaptiveVanillaDecoder(out_channels, latent_dim//4, patch_size)
        
        # Latent expansion for patches
        self.latent_expand = nn.Linear(latent_dim, latent_dim//4)
        
        # Blending network for patch stitching
        self.blend_conv = nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1)

    def forward(self, global_latent, patch_positions=None):
        if patch_positions is None:
            # Simple case: single patch
            patch_latent = self.latent_expand(global_latent)
            return self.patch_decoder(patch_latent)
        
        # Multi-patch reconstruction
        b = global_latent.shape[0]
        output = torch.zeros(b, self.out_channels, self.target_size, self.target_size, 
                           device=global_latent.device)
        weight_map = torch.zeros(b, 1, self.target_size, self.target_size, 
                               device=global_latent.device)
        
        # Expand latent for patch processing
        patch_latent = self.latent_expand(global_latent)
        
        # Reconstruct each patch
        for pos in patch_positions:
            i, j = pos
            reconstructed_patch = self.patch_decoder(patch_latent)
            
            # Accumulate with soft blending
            output[:, :, i:i+self.patch_size, j:j+self.patch_size] += reconstructed_patch
            weight_map[:, :, i:i+self.patch_size, j:j+self.patch_size] += 1
        
        # Normalize by overlap count
        output = output / (weight_map + 1e-8)
        
        return output


class PatchBasedAE(nn.Module):
    """Patch-based AutoEncoder for memory-efficient large image processing"""

    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, 
                 target_size=512, patch_size=256, overlap=32):
        super().__init__()
        
        self.target_size = target_size
        self.patch_size = patch_size
        self.overlap = overlap
        
        self.encoder = PatchBasedEncoder(in_channels, latent_dim, patch_size, overlap)
        self.decoder = PatchBasedDecoder(out_channels, latent_dim, target_size, patch_size)

    def forward(self, x):
        global_latent, patch_features, patch_positions = self.encoder(x)
        reconstructed = self.decoder(global_latent, patch_positions)
        
        return reconstructed, global_latent, patch_features


if __name__ == "__main__":
    # Test different models with various input sizes
    test_sizes = [(256, 256), (512, 512), (1024, 1024), (768, 1024)]
    
    print("Testing Adaptive Vanilla AutoEncoder:")
    for h, w in test_sizes:
        model = AdaptiveVanillaAE(target_size=max(h, w))
        x = torch.randn(2, 3, h, w)
        
        try:
            output, latent, features = model(x)
            print(f"Input: {x.shape} -> Output: {output.shape}, Latent: {latent.shape}")
        except Exception as e:
            print(f"Error with size {(h, w)}: {e}")
    
    print("\nTesting Multi-Scale UNet AutoEncoder:")
    for h, w in test_sizes:
        model = MultiScaleUnetAE(target_size=max(h, w))
        x = torch.randn(2, 3, h, w)
        
        try:
            main_out, latent, features, aux1, aux2 = model(x)
            print(f"Input: {x.shape} -> Main: {main_out.shape}, Aux1: {aux1.shape}, Aux2: {aux2.shape}")
        except Exception as e:
            print(f"Error with size {(h, w)}: {e}")
    
    print("\nTesting Patch-Based AutoEncoder:")
    model = PatchBasedAE(target_size=1024, patch_size=256)
    x = torch.randn(1, 3, 1024, 1024)
    
    try:
        output, latent, patch_features = model(x)
        print(f"Input: {x.shape} -> Output: {output.shape}, Latent: {latent.shape}")
    except Exception as e:
        print(f"Error: {e}")