"""
Basic building blocks for neural network architectures.

This module provides reusable components like convolution blocks, 
deconvolution blocks, and other common layers used across different
anomaly detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation='leaky_relu', dropout=0.0, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            norm: Whether to apply batch normalization
            activation: Activation function type
            dropout: Dropout probability (0 to disable)
            bias: Whether to use bias in convolution
        """
        super().__init__()

        layers = []
        
        # Convolution layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

        # Normalization
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Activation
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    """Basic deconvolution block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 output_padding=0, norm=True, activation='relu', dropout=0.0, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Deconvolution kernel size
            stride: Deconvolution stride
            padding: Deconvolution padding
            output_padding: Additional padding for output size adjustment
            norm: Whether to apply batch normalization
            activation: Activation function type
            dropout: Dropout probability (0 to disable)
            bias: Whether to use bias in deconvolution
        """
        super().__init__()

        layers = []
        
        # Deconvolution layer
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                        stride, padding, output_padding, bias=bias))

        # Normalization
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_block(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, channels, kernel_size=3, stride=1, padding=1,
                 norm=True, activation='relu', dropout=0.0):
        super().__init__()
        
        self.conv1 = ConvBlock(channels, channels, kernel_size, stride, padding,
                              norm=norm, activation=activation, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, kernel_size, stride, padding,
                              norm=norm, activation=None, dropout=dropout)
        
        # Activation for residual connection
        if activation == 'relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.final_activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.final_activation = nn.GELU()
        elif activation == 'swish':
            self.final_activation = nn.SiLU()
        else:
            self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.final_activation(out)
        return out


class AttentionBlock(nn.Module):
    """Simple attention mechanism for feature enhancement"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        
        self.channel_attention = AttentionBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PixelShuffle2d(nn.Module):
    """2D Pixel shuffle for upsampling alternative to deconvolution"""
    
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 
                             kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class AdaptiveConv2d(nn.Module):
    """Adaptive convolution that adjusts to input size"""
    
    def __init__(self, in_channels, out_channels, target_size=None, mode='bilinear'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.target_size = target_size
        self.mode = mode
    
    def forward(self, x):
        x = self.conv(x)
        if self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode=self.mode, align_corners=False)
        return x


if __name__ == "__main__":
    # Test building blocks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ConvBlock
    print("Testing ConvBlock:")
    conv_block = ConvBlock(3, 64, kernel_size=4, stride=2, padding=1).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out = conv_block(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Test DeconvBlock  
    print("\nTesting DeconvBlock:")
    deconv_block = DeconvBlock(64, 32, kernel_size=4, stride=2, padding=1).to(device)
    out2 = deconv_block(out)
    print(f"Input: {out.shape} -> Output: {out2.shape}")
    
    # Test ResidualBlock
    print("\nTesting ResidualBlock:")
    res_block = ResidualBlock(64).to(device)
    out3 = res_block(out)
    print(f"Input: {out.shape} -> Output: {out3.shape}")
    
    # Test CBAM
    print("\nTesting CBAM:")
    cbam = CBAM(64).to(device)
    out4 = cbam(out)
    print(f"Input: {out.shape} -> Output: {out4.shape}")
    
    print("\nAll building blocks working correctly!")