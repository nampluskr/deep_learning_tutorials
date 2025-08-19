import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# Model Factory Functions
# =============================================================================

def get_model(model_type, **model_params):
    """Get and return the specified autoencoder model"""
    available_models = ['vanilla_ae', 'unet_ae']
    in_channels = model_params.get('in_channels', 3)
    out_channels = model_params.get('out_channels', 3)
    latent_dim = model_params.get('latent_dim', 512)
    input_size = model_params.get('input_size', 256)

    if model_type == 'vanilla_ae':
        model = VanillaAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            input_size=input_size
        )
    elif model_type == 'unet_ae':
        model = UnetAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            input_size=input_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available models: {available_models}")
    return model


def load_model_from_checkpoint(checkpoint_path, model_type=None, **model_params):
    """Load pre-trained model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if model_type is None:
        # Try to get model type from checkpoint
        model_type = checkpoint.get('model_type', 'vanilla_ae')
    
    model = get_model(model_type, **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size after convolution"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def calculate_deconv_output_size(input_size, kernel_size, stride, padding, output_padding=0):
    """Calculate output size after deconvolution"""
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding


def get_final_conv_size(input_size, num_layers=5):
    """Calculate final feature map size after multiple conv layers"""
    size = input_size
    for _ in range(num_layers):
        size = calculate_conv_output_size(size, kernel_size=4, stride=2, padding=1)
    return size


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Basic convolution block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation='leaky_relu', dropout=0.0):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
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
                 output_padding=0, norm=True, activation='relu', dropout=0.0):
        super().__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                        stride, padding, output_padding))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_block(x)


# =============================================================================
# Vanilla AutoEncoder with Dynamic Sizing
# =============================================================================

class VanillaEncoder(nn.Module):
    """Enhanced encoder with dynamic spatial size handling"""
    
    def __init__(self, in_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

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
            DeconvBlock(512, 256, output_padding=self._get_output_padding(4, target_sizes[0])),     # 4 -> target_sizes[0]
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
        expected_size = calculate_deconv_output_size(input_size, kernel_size=4, stride=2, padding=1)
        return max(0, target_size - expected_size)

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
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


# =============================================================================
# UNet-style Autoencoder with Enhanced Skip Connections
# =============================================================================

class UnetEncoder(nn.Module):
    """Enhanced UNet-style encoder with dynamic sizing and safe skip connections"""
    
    def __init__(self, in_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size

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

    def _safe_concatenate(self, x, skip):
        """Safely concatenate tensors with different spatial sizes"""
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            # Resize skip connection to match x
            skip = F.interpolate(skip, size=(x.size(-2), x.size(-1)), 
                               mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, latent, skip_connections):
        x = self.fc(latent)
        x = self.unflatten(x)  # (B, 512, 4, 4)

        # Decoder with safe skip connections
        d1 = self.deconv1(x)                                    # 256 x H/16 x W/16
        d2 = self.deconv2(self._safe_concatenate(d1, skip_connections[3]))  # 128 x H/8 x W/8
        d3 = self.deconv3(self._safe_concatenate(d2, skip_connections[2]))  # 64 x H/4 x W/4
        d4 = self.deconv4(self._safe_concatenate(d3, skip_connections[1]))  # 32 x H/2 x W/2
        d5 = self.final_conv(self._safe_concatenate(d4, skip_connections[0]))  # out_channels x H x W

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
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return reconstructed, latent, features


# =============================================================================
# Model Information and Debugging
# =============================================================================

def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def model_summary(model, input_size=(3, 256, 256)):
    """Print model summary with layer information"""
    total_params, trainable_params = count_parameters(model)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    # Test forward pass
    with torch.no_grad():
        x = torch.randn(1, *input_size)
        try:
            output, latent, features = model(x)
            print(f"Output shape: {output.shape}")
            print(f"Latent shape: {latent.shape}")
            print(f"Features shape: {features.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    # Test different models and input sizes
    input_sizes = [256, 512, 128]
    model_types = ['vanilla_ae', 'unet_ae']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()}")
        print(f"{'='*50}")
        
        for input_size in input_sizes:
            print(f"\nInput size: {input_size}x{input_size}")
            print("-" * 30)
            
            try:
                model = get_model(
                    model_type, 
                    in_channels=3, 
                    out_channels=3, 
                    latent_dim=512,
                    input_size=input_size
                )
                model_summary(model, input_size=(3, input_size, input_size))
                
            except Exception as e:
                print(f"Error creating model: {e}")

    # Test edge cases
    print(f"\n{'='*50}")
    print("Testing edge cases")
    print(f"{'='*50}")
    
    # Very small input
    try:
        model = get_model('vanilla_ae', input_size=16)
        print("16x16 input: Failed to catch minimum size error!")
    except ValueError as e:
        print(f"16x16 input: Correctly caught error - {e}")
    
    # Very large input
    try:
        model = get_model('vanilla_ae', input_size=1024)
        model_summary(model, input_size=(3, 1024, 1024))
        print("1024x1024 input: Success!")
    except Exception as e:
        print(f"1024x1024 input: Error - {e}")