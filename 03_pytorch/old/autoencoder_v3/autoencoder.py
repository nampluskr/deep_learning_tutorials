import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def load_model(model_type, **model_params):
    """Load and return the specified autoencoder model"""
    available_models = ['vanilla_ae', 'unet_ae']
    in_channels = model_params.get('in_channels', 3)
    out_channels = model_params.get('out_channels', 3)
    latent_dim = model_params.get('latent_dim', 512)

    if model_type == 'vanilla_ae':
        model = VanillaAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim
        )
    elif model_type == 'unet_ae':
        model = UnetAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available models: {available_models}")
    return model


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Basic convolution block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation='leaky_relu'):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)




class DeconvBlock(nn.Module):
    """Basic deconvolution block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation='relu', dropout=False):
        super().__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if dropout:
            layers.append(nn.Dropout2d(0.5))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_block(x)




# =============================================================================
# Vanilla AutoEncoder
# =============================================================================

class VanillaEncoder(nn.Module):
    """Basic encoder with convolutional layers and global average pooling"""
    
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),      # 256 -> 128
            ConvBlock(32, 64),               # 128 -> 64
            ConvBlock(64, 128),              # 64 -> 32
            ConvBlock(128, 256),             # 32 -> 16
            ConvBlock(256, 512),             # 16 -> 8
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        return latent, features




class VanillaDecoder(nn.Module):
    """Basic decoder with deconvolutional layers and sigmoid activation"""
    
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))

        self.deconv_blocks = nn.Sequential(
            DeconvBlock(512, 256),           # 8 -> 16
            DeconvBlock(256, 128),           # 16 -> 32
            DeconvBlock(128, 64),            # 32 -> 64
            DeconvBlock(64, 32),             # 64 -> 128
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid(),
        )

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed




class VanillaAE(nn.Module):
    """Basic autoencoder combining vanilla encoder and decoder"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = VanillaEncoder(in_channels, latent_dim)
        self.decoder = VanillaDecoder(out_channels, latent_dim)

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features




# =============================================================================
# UNet-style Autoencoder with Skip Connections
# =============================================================================

class UnetEncoder(nn.Module):
    """UNet-style encoder with skip connections for feature preservation"""
    
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()

        # Encoder blocks
        self.conv1 = ConvBlock(in_channels, 32)    # 256 -> 128
        self.conv2 = ConvBlock(32, 64)             # 128 -> 64
        self.conv3 = ConvBlock(64, 128)            # 64 -> 32
        self.conv4 = ConvBlock(128, 256)           # 32 -> 16
        self.conv5 = ConvBlock(256, 512)           # 16 -> 8

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        # Forward pass with skip connections
        e1 = self.conv1(x)      # 32 x 128 x 128
        e2 = self.conv2(e1)     # 64 x 64 x 64
        e3 = self.conv3(e2)     # 128 x 32 x 32
        e4 = self.conv4(e3)     # 256 x 16 x 16
        e5 = self.conv5(e4)     # 512 x 8 x 8

        pooled = self.pool(e5)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)

        # Return latent and skip connections
        skip_connections = [e1, e2, e3, e4]
        return latent, e5, skip_connections




class UnetDecoder(nn.Module):
    """UNet-style decoder with skip connections for detailed reconstruction"""
    
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))

        # Decoder blocks
        self.deconv1 = DeconvBlock(512, 256)                    # 8 -> 16
        self.deconv2 = DeconvBlock(256 + 256, 128)             # 16 -> 32 (with skip)
        self.deconv3 = DeconvBlock(128 + 128, 64)              # 32 -> 64 (with skip)
        self.deconv4 = DeconvBlock(64 + 64, 32)                # 64 -> 128 (with skip)
        self.deconv5 = nn.ConvTranspose2d(32 + 32, out_channels,
                                         kernel_size=4, stride=2, padding=1)  # 128 -> 256 (with skip)
        self.final_activation = nn.Sigmoid()

    def forward(self, latent, skip_connections):
        x = self.fc(latent)
        x = self.unflatten(x)

        # Decoder with skip connections
        d1 = self.deconv1(x)                                    # 256 x 16 x 16
        d2 = self.deconv2(torch.cat([d1, skip_connections[3]], dim=1))  # 128 x 32 x 32
        d3 = self.deconv3(torch.cat([d2, skip_connections[2]], dim=1))  # 64 x 64 x 64
        d4 = self.deconv4(torch.cat([d3, skip_connections[1]], dim=1))  # 32 x 128 x 128
        d5 = self.deconv5(torch.cat([d4, skip_connections[0]], dim=1))  # out_channels x 256 x 256

        reconstructed = self.final_activation(d5)
        return reconstructed




class UnetAE(nn.Module):
    """UNet-style autoencoder with skip connections for enhanced detail preservation"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, latent_dim)
        self.decoder = UnetDecoder(out_channels, latent_dim)

    def forward(self, x):
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return reconstructed, latent, features




if __name__ == "__main__":
    # Example usage
    model = UnetAE(in_channels=3, out_channels=3, latent_dim=512)
    x = torch.randn(8, 3, 256, 256)  # Batch of 8 images
    reconstructed, latent, features = model(x)

    print("Reconstructed shape:", reconstructed.shape)
    print("Latent shape:", latent.shape)
    print("Features shape:", features.shape)

    # Output shapes should be:
    # Reconstructed shape: torch.Size([8, 3, 256, 256])
    # Latent shape: torch.Size([8, 512])