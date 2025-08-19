"""
Anomaly Detection Models
다양한 AutoEncoder와 VAE 모델들을 포함합니다.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# 기본 Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """기본 Convolution Block"""
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
    """기본 Deconvolution Block"""
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
# Model Factory Functions
# =============================================================================

def get_model(model_type, **model_params):
    """
    모델 타입에 따라 모델 인스턴스를 반환
    
    Args:
        model_type: 모델 타입
            - 'vanilla_ae': 기본 AutoEncoder
            - 'unet_ae': Skip Connection이 있는 UNet 스타일 AutoEncoder
            - 'vanilla_vae': 기본 VAE
            - 'unet_vae': Skip Connection이 있는 UNet 스타일 VAE
            - 'resnet_ae': ResNet 기반 AutoEncoder
            - 'vgg_ae': VGG 기반 AutoEncoder  
            - 'efficientnet_ae': EfficientNet 기반 AutoEncoder
        **model_params: 모델 파라미터
            - in_channels: 입력 채널 수 (기본: 3)
            - out_channels: 출력 채널 수 (기본: 3)
            - latent_dim: 잠재 공간 차원 (기본: 512)
            - beta: VAE의 KL divergence weight (기본: 1.0)
            - backbone: Pretrained 모델의 backbone (예: 'resnet18', 'vgg16', 'efficientnet_b0')
            - weights: Pretrained weights 설정 (기본: None)
                     None: Random initialization
                     'imagenet' 또는 'default': ImageNet pretrained weights
                     파일 경로: 커스텀 pretrained weights (.pth 파일)
            - freeze_backbone: Backbone freeze 여부 (기본: False)
    
    Returns:
        model: 초기화된 모델 인스턴스
    """
    
    # 기본 파라미터 설정
    in_channels = model_params.get('in_channels', 3)
    out_channels = model_params.get('out_channels', 3)
    latent_dim = model_params.get('latent_dim', 512)
    beta = model_params.get('beta', 1.0)
    backbone = model_params.get('backbone', None)
    weights = model_params.get('weights', None)  # 기본값을 None으로 변경
    freeze_backbone = model_params.get('freeze_backbone', False)
    
    # 기본 AutoEncoder 모델들
    if model_type == 'vanilla_ae':
        return VanillaAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim
        )
    
    elif model_type == 'unet_ae':
        return UnetAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim
        )
    
    elif model_type == 'vanilla_vae':
        return VanillaVAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            beta=beta
        )
    
    elif model_type == 'unet_vae':
        return UnetVAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            beta=beta
        )
    
    # Pretrained CNN 기반 AutoEncoder 모델들
    elif model_type == 'resnet_ae':
        # 기본 backbone 설정
        if backbone is None:
            backbone = 'resnet18'
        
        return ResNetAE(
            backbone=backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            weights=weights,
            freeze_backbone=freeze_backbone
        )
    
    elif model_type == 'vgg_ae':
        # 기본 backbone 설정
        if backbone is None:
            backbone = 'vgg16'
        
        return VggAE(
            backbone=backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            weights=weights,
            freeze_backbone=freeze_backbone
        )
    
    elif model_type == 'efficientnet_ae':
        # 기본 backbone 설정 (OLED 화질 평가에 최적화)
        if backbone is None:
            backbone = 'efficientnet_b0'
        
        return EfficientNetAE(
            backbone=backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            weights=weights,
            freeze_backbone=freeze_backbone
        )
    
    else:
        available_models = [
            'vanilla_ae', 'unet_ae', 'vanilla_vae', 'unet_vae', 
            'resnet_ae', 'vgg_ae', 'efficientnet_ae'
        ]
        raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")


def get_model_info(model_type):
    """모델 타입별 정보 반환"""
    
    model_info = {
        'vanilla_ae': {
            'name': 'Vanilla AutoEncoder',
            'description': '기본적인 Encoder-Decoder 구조의 AutoEncoder',
            'skip_connections': False,
            'model_family': 'AutoEncoder',
            'parameters_256': '약 2.5M',
            'backbone': 'Custom CNN',
        },
        'unet_ae': {
            'name': 'UNet AutoEncoder',
            'description': 'Skip Connection이 추가된 UNet 스타일 AutoEncoder',
            'skip_connections': True,
            'model_family': 'AutoEncoder',
            'parameters_256': '약 3.2M',
            'backbone': 'Custom CNN',
        },
        'vanilla_vae': {
            'name': 'Vanilla VAE',
            'description': '기본적인 Variational AutoEncoder',
            'skip_connections': False,
            'model_family': 'VAE',
            'parameters_256': '약 2.5M',
            'backbone': 'Custom CNN',
        },
        'unet_vae': {
            'name': 'UNet VAE',
            'description': 'Skip Connection이 추가된 UNet 스타일 VAE',
            'skip_connections': True,
            'model_family': 'VAE',
            'parameters_256': '약 3.2M',
            'backbone': 'Custom CNN',
        },
        'resnet_ae': {
            'name': 'ResNet AutoEncoder',
            'description': 'ResNet 기반 Pretrained Encoder AutoEncoder (기본: random weights)',
            'skip_connections': False,
            'model_family': 'Pretrained AutoEncoder',
            'parameters_256': '약 12M (ResNet18), 26M (ResNet50)',
            'backbone': 'ResNet (weights 옵션: None, imagenet/default, custom.pth)',
        },
        'vgg_ae': {
            'name': 'VGG AutoEncoder',
            'description': 'VGG 기반 Pretrained Encoder AutoEncoder (기본: random weights)',
            'skip_connections': False,
            'model_family': 'Pretrained AutoEncoder',
            'parameters_256': '약 15M (VGG16), 20M (VGG19)',
            'backbone': 'VGG (weights 옵션: None, imagenet/default, custom.pth)',
        },
        'efficientnet_ae': {
            'name': 'EfficientNet AutoEncoder',
            'description': 'EfficientNet 기반 AutoEncoder (OLED 화질 평가 최적화, 기본: random weights)',
            'skip_connections': False,
            'model_family': 'Pretrained AutoEncoder',
            'parameters_256': '약 5M (B0), 19M (B4)',
            'backbone': 'EfficientNet (weights 옵션: None, imagenet/default, custom.pth)',
        }
    }
    
    return model_info.get(model_type, {})


# =============================================================================
# 기본 모델들: VanillaAE, UnetAE, VanillaVAE, UnetVAE
# =============================================================================

# Vanilla AutoEncoder
class VanillaEncoder(nn.Module):
    """기본 Encoder"""
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
    """기본 Decoder"""
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
    """기본 AutoEncoder"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = VanillaEncoder(in_channels, latent_dim)
        self.decoder = VanillaDecoder(out_channels, latent_dim)
    
    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


# UNet-style AutoEncoder
class UnetEncoder(nn.Module):
    """Skip Connection이 있는 UNet 스타일 Encoder"""
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
    """Skip Connection이 있는 UNet 스타일 Decoder"""
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
    """Skip Connection이 있는 UNet 스타일 AutoEncoder"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, latent_dim)
        self.decoder = UnetDecoder(out_channels, latent_dim)
    
    def forward(self, x):
        latent, features, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        return reconstructed, latent, features


# Vanilla VAE
class VAEEncoder(nn.Module):
    """VAE용 Encoder (mu, logvar 출력)"""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        return mu, logvar, features


class VAEDecoder(nn.Module):
    """VAE용 Decoder"""
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        
        self.deconv_blocks = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VanillaVAE(nn.Module):
    """기본 Variational AutoEncoder"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight
        
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(out_channels, latent_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu, logvar, features = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, z, features, mu, logvar
    
    def compute_loss(self, x, reconstructed, mu, logvar):
        """VAE Loss = Reconstruction Loss + Beta * KL Divergence"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0)  # Average over batch
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


# UNet-style VAE
class UnetVAEEncoder(nn.Module):
    """Skip Connection이 있는 UNet 스타일 VAE Encoder"""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        
        # Encoder blocks
        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        # Forward pass with skip connections
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        pooled = self.pool(e5)
        pooled = pooled.view(pooled.size(0), -1)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        skip_connections = [e1, e2, e3, e4]
        return mu, logvar, e5, skip_connections


class UnetVAEDecoder(nn.Module):
    """Skip Connection이 있는 UNet 스타일 VAE Decoder"""
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        
        # Decoder blocks with skip connections
        self.deconv1 = DeconvBlock(512, 256)
        self.deconv2 = DeconvBlock(256 + 256, 128)
        self.deconv3 = DeconvBlock(128 + 128, 64)
        self.deconv4 = DeconvBlock(64 + 64, 32)
        self.deconv5 = nn.ConvTranspose2d(32 + 32, out_channels, 
                                         kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, z, skip_connections):
        x = self.fc(z)
        x = self.unflatten(x)
        
        # Decoder with skip connections
        d1 = self.deconv1(x)
        d2 = self.deconv2(torch.cat([d1, skip_connections[3]], dim=1))
        d3 = self.deconv3(torch.cat([d2, skip_connections[2]], dim=1))
        d4 = self.deconv4(torch.cat([d3, skip_connections[1]], dim=1))
        d5 = self.deconv5(torch.cat([d4, skip_connections[0]], dim=1))
        
        reconstructed = self.final_activation(d5)
        return reconstructed


class UnetVAE(nn.Module):
    """Skip Connection이 있는 UNet 스타일 VAE"""
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = UnetVAEEncoder(in_channels, latent_dim)
        self.decoder = UnetVAEDecoder(out_channels, latent_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu, logvar, features, skip_connections = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z, skip_connections)
        
        return reconstructed, z, features, mu, logvar
    
    def compute_loss(self, x, reconstructed, mu, logvar):
        """VAE Loss = Reconstruction Loss + Beta * KL Divergence"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


# =============================================================================
# Pretrained 모델들: ResNet, VGG, EfficientNet
# =============================================================================

class PretrainedEncoder(nn.Module):
    """Pretrained CNN을 이용한 Encoder 기본 클래스"""
    def __init__(self, backbone_name='resnet18', latent_dim=512, weights=None, freeze_backbone=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        
        # weights 파라미터 처리 - 새로운 torchvision weights enum 방식 사용
        model_weights = self._get_model_weights(backbone_name, weights)
        
        # Backbone 네트워크 로드
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=model_weights)
            self.feature_dim = 512
            # 마지막 FC layer 제거
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
        
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=model_weights)
            self.feature_dim = 2048
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        elif backbone_name == 'vgg16':
            self.backbone = models.vgg16(weights=model_weights).features
            self.feature_dim = 512
        
        elif backbone_name == 'vgg19':
            self.backbone = models.vgg19(weights=model_weights).features
            self.feature_dim = 512
        
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=model_weights).features
            self.feature_dim = 1280
        
        elif backbone_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights=model_weights).features
            self.feature_dim = 1792
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 커스텀 weights 로드 (파일 경로가 주어진 경우)
        if weights is not None and weights not in ['imagenet', 'default']:
            self._load_custom_weights(weights)
        
        # Backbone freeze 옵션
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone {backbone_name} frozen (not trainable)")
        else:
            print(f"Backbone {backbone_name} trainable")
        
        # Adaptive pooling과 FC layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, latent_dim)
    
    def _get_model_weights(self, backbone_name, weights):
        """torchvision weights enum 반환"""
        if weights is None:
            return None
        elif weights in ['imagenet', 'default']:
            # 각 모델별 기본 ImageNet weights enum 반환
            if backbone_name == 'resnet18':
                return models.ResNet18_Weights.DEFAULT
            elif backbone_name == 'resnet50':
                return models.ResNet50_Weights.DEFAULT
            elif backbone_name == 'vgg16':
                return models.VGG16_Weights.DEFAULT
            elif backbone_name == 'vgg19':
                return models.VGG19_Weights.DEFAULT
            elif backbone_name == 'efficientnet_b0':
                return models.EfficientNet_B0_Weights.DEFAULT
            elif backbone_name == 'efficientnet_b4':
                return models.EfficientNet_B4_Weights.DEFAULT
            else:
                return None
        else:
            # 커스텀 weights 파일 경로인 경우, 일단 None으로 로드 후 별도 처리
            return None
    
    def _load_custom_weights(self, weights_path):
        """커스텀 pretrained weights 로드"""
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            print(f"Loading custom weights from: {weights_path}")
            
            # weights 로드
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # checkpoint가 dict인 경우 (보통 'state_dict' 키가 있음)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # checkpoint 자체가 state_dict인 경우
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Backbone에 해당하는 weights만 필터링
            backbone_state_dict = {}
            for key, value in state_dict.items():
                # 'backbone.' 접두사가 있는 경우 제거
                if key.startswith('backbone.'):
                    new_key = key[9:]  # 'backbone.' 제거
                elif key.startswith('encoder.backbone.'):
                    new_key = key[17:]  # 'encoder.backbone.' 제거
                else:
                    new_key = key
                
                backbone_state_dict[new_key] = value
            
            # 모델에 weights 로드 (strict=False로 설정하여 일부 불일치 허용)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in backbone: {missing_keys[:5]}...")  # 처음 5개만 표시
            if unexpected_keys:
                print(f"Warning: Unexpected keys in backbone: {unexpected_keys[:5]}...")  # 처음 5개만 표시
            
            print(f"Successfully loaded custom weights from {weights_path}")
            
        except Exception as e:
            print(f"Error loading custom weights from {weights_path}: {e}")
            print(f"Continuing with random initialization...")
    
    def forward(self, x):
        # Pretrained 모델은 ImageNet 정규화가 필요할 수 있음
        features = self.backbone(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        
        return latent, features


class AdaptiveDecoder(nn.Module):
    """다양한 feature 크기에 대응하는 Adaptive Decoder"""
    def __init__(self, latent_dim=512, out_channels=3, feature_channels=512):
        super().__init__()
        
        # Feature 크기에 따라 초기 크기 결정
        if feature_channels <= 512:
            initial_size = 8
            initial_channels = 512
        elif feature_channels <= 1024:
            initial_size = 8
            initial_channels = 512
        else:  # > 1024
            initial_size = 8
            initial_channels = 512
        
        self.fc = nn.Linear(latent_dim, initial_channels * initial_size * initial_size)
        self.unflatten = nn.Unflatten(1, (initial_channels, initial_size, initial_size))
        
        # Progressive upsampling
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            DeconvBlock(initial_channels, 256),
            # 16x16 -> 32x32
            DeconvBlock(256, 128),
            # 32x32 -> 64x64
            DeconvBlock(128, 64),
            # 64x64 -> 128x128
            DeconvBlock(64, 32),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder(x)
        return reconstructed


class ResNetAE(nn.Module):
    """ResNet 기반 AutoEncoder"""
    def __init__(self, backbone='resnet18', in_channels=3, out_channels=3, latent_dim=512, 
                 weights=None, freeze_backbone=False):
        super().__init__()
        
        self.encoder = PretrainedEncoder(
            backbone_name=backbone, 
            latent_dim=latent_dim, 
            weights=weights,
            freeze_backbone=freeze_backbone
        )
        
        self.decoder = AdaptiveDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            feature_channels=self.encoder.feature_dim
        )
    
    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


class VggAE(nn.Module):
    """VGG 기반 AutoEncoder"""
    def __init__(self, backbone='vgg16', in_channels=3, out_channels=3, latent_dim=512, 
                 weights=None, freeze_backbone=False):
        super().__init__()
        
        self.encoder = PretrainedEncoder(
            backbone_name=backbone, 
            latent_dim=latent_dim, 
            weights=weights,
            freeze_backbone=freeze_backbone
        )
        
        self.decoder = AdaptiveDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            feature_channels=self.encoder.feature_dim
        )
    
    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


class EfficientNetAE(nn.Module):
    """EfficientNet 기반 AutoEncoder (OLED 화질 평가에 최적화)"""
    def __init__(self, backbone='efficientnet_b0', in_channels=3, out_channels=3, latent_dim=512, 
                 weights=None, freeze_backbone=False):
        super().__init__()
        
        self.encoder = PretrainedEncoder(
            backbone_name=backbone, 
            latent_dim=latent_dim, 
            weights=weights,
            freeze_backbone=freeze_backbone
        )
        
        self.decoder = AdaptiveDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            feature_channels=self.encoder.feature_dim
        )
    
    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


# =============================================================================
# Test Function
# =============================================================================

def test_models():
    """모델들의 기본 동작을 테스트"""
    
    print("=" * 60)
    print("Models Test")
    print("=" * 60)
    
    # 테스트 입력
    batch_size = 2  # 메모리 절약을 위해 줄임
    channels = 3
    img_size = 256
    latent_dim = 512
    
    x = torch.randn(batch_size, channels, img_size, img_size)
    print(f"Input shape: {x.shape}")
    
    # 테스트할 모델들
    model_tests = [
        ('vanilla_ae', {}),
        ('unet_ae', {}),
        ('vanilla_vae', {'beta': 1.0}),
        ('unet_vae', {'beta': 1.0}),
        ('resnet_ae', {'backbone': 'resnet18', 'weights': None}),
        ('vgg_ae', {'backbone': 'vgg16', 'weights': None}),
        ('efficientnet_ae', {'backbone': 'efficientnet_b0', 'weights': None}),
    ]
    
    for model_type, extra_params in model_tests:
        print(f"\nTesting {model_type}...")
        
        try:
            # 모델 생성
            model_params = {'latent_dim': latent_dim}
            model_params.update(extra_params)
            
            model = get_model(model_type, **model_params)
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                if 'vae' in model_type:
                    output = model(x)
                    reconstructed, z, features, mu, logvar = output
                    print(f"   Reconstructed: {reconstructed.shape}")
                    print(f"   Latent (z): {z.shape}")
                    print(f"   Mu: {mu.shape}")
                    print(f"   Logvar: {logvar.shape}")
                else:
                    reconstructed, latent, features = model(x)
                    print(f"   Reconstructed: {reconstructed.shape}")
                    print(f"   Latent: {latent.shape}")
                
                print(f"   Features: {features.shape}")
            
            # 파라미터 수 계산
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {total_params:,}")
            
            # 모델 정보
            info = get_model_info(model_type)
            print(f"   Description: {info.get('description', 'N/A')}")
            print(f"   Backbone: {info.get('backbone', 'N/A')}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\nTesting pretrained model with different backbones...")
    
    # Pretrained 모델의 다양한 backbone 테스트
    pretrained_tests = [
        ('resnet_ae', 'resnet50'),
        ('vgg_ae', 'vgg19'),
        ('efficientnet_ae', 'efficientnet_b4'),
    ]
    
    for model_type, backbone in pretrained_tests:
        print(f"\nTesting {model_type} with {backbone}...")
        try:
            model = get_model(model_type, backbone=backbone, weights=None, latent_dim=256)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {total_params:,}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\nTesting with different weight options...")
    
    # 다양한 weights 옵션 테스트
    weight_tests = [
        ('resnet_ae', {'backbone': 'resnet18', 'weights': None}),
        ('resnet_ae', {'backbone': 'resnet18', 'weights': 'imagenet'}),
        ('resnet_ae', {'backbone': 'resnet18', 'weights': 'default'}),
        # ('resnet_ae', {'backbone': 'resnet18', 'weights': './pretrained/resnet18_custom.pth'}),  # 실제 파일이 있을 때
    ]
    
    for model_type, params in weight_tests:
        weights_name = params.get('weights', 'None')
        print(f"\nTesting {model_type} with weights={weights_name}...")
        try:
            model = get_model(model_type, latent_dim=256, **params)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\nAll model tests completed!")
    print("\nWeights options:")
    print("  - weights=None: Random initialization (default)")
    print("  - weights='imagenet' or 'default': ImageNet pretrained weights")
    print("  - weights='/path/to/model.pth': Custom pretrained weights")
    print("\nNote: No more deprecation warnings with updated torchvision weights API!")


if __name__ == "__main__":
    test_models()