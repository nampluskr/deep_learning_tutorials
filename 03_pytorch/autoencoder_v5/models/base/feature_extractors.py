"""
Pretrained feature extractors for anomaly detection models.

This module provides wrappers for popular pretrained models (ResNet, VGG, EfficientNet)
that can be used as encoders in various anomaly detection approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


class PretrainedEncoder(nn.Module):
    """Base class for pretrained feature extractors"""
    
    def __init__(self, backbone_name, pretrained=True, freeze_backbone=False, 
                 output_layers=None, latent_dim=512):
        """
        Args:
            backbone_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            output_layers: List of layer names to extract features from
            latent_dim: Dimension of final latent representation
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.output_layers = output_layers or []
        self.latent_dim = latent_dim
        
        # Will be set by subclasses
        self.backbone = None
        self.feature_layers = nn.ModuleDict()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _setup_feature_extraction(self):
        """Setup feature extraction from intermediate layers"""
        if not self.output_layers:
            return
            
        # This should be implemented by subclasses for specific architectures
        pass
    
    def forward(self, x):
        """Extract features and return latent representation"""
        features = self._extract_features(x)
        
        # Get final feature map for latent representation
        if isinstance(features, dict):
            final_features = list(features.values())[-1]
        else:
            final_features = features
            
        # Global average pooling + FC for latent representation
        pooled = self.adaptive_pool(final_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        if self.fc is not None:
            latent = self.fc(pooled)
        else:
            latent = pooled
            
        return latent, final_features
    
    def _extract_features(self, x):
        """Extract features from backbone - to be implemented by subclasses"""
        raise NotImplementedError


class ResNetEncoder(PretrainedEncoder):
    """ResNet-based feature extractor"""
    
    def __init__(self, arch='resnet50', pretrained=True, freeze_backbone=False,
                 output_layers=None, latent_dim=512):
        """
        Args:
            arch: ResNet architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            output_layers: Layer names to extract features from
            latent_dim: Final latent dimension
        """
        super().__init__(arch, pretrained, freeze_backbone, output_layers, latent_dim)
        
        # Load pretrained ResNet
        if arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif arch == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif arch == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        elif arch == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet architecture: {arch}")
        
        # Remove final classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Setup feature extraction layers
        self._setup_feature_extraction()
        
        # Final FC layer for latent representation
        self.fc = nn.Linear(backbone_dim, latent_dim)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _setup_feature_extraction(self):
        """Setup intermediate feature extraction for ResNet"""
        if not self.output_layers:
            return
            
        # ResNet layer mapping
        layer_mapping = {
            'conv1': 0,
            'layer1': 4,
            'layer2': 5, 
            'layer3': 6,
            'layer4': 7
        }
        
        for layer_name in self.output_layers:
            if layer_name in layer_mapping:
                self.feature_layers[layer_name] = self.backbone[layer_mapping[layer_name]]
    
    def _extract_features(self, x):
        """Extract features from ResNet backbone"""
        if not self.output_layers:
            # Simple forward pass
            return self.backbone(x)
        
        # Extract intermediate features
        features = {}
        current_x = x
        
        for i, layer in enumerate(self.backbone):
            current_x = layer(current_x)
            
            # Check if this layer should be saved
            for layer_name, layer_idx in [('conv1', 0), ('layer1', 4), ('layer2', 5), ('layer3', 6), ('layer4', 7)]:
                if i == layer_idx and layer_name in self.output_layers:
                    features[layer_name] = current_x
        
        return features if features else current_x


class VGGEncoder(PretrainedEncoder):
    """VGG-based feature extractor"""
    
    def __init__(self, arch='vgg16', pretrained=True, freeze_backbone=False,
                 output_layers=None, latent_dim=512):
        """
        Args:
            arch: VGG architecture ('vgg11', 'vgg13', 'vgg16', 'vgg19')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            output_layers: Layer indices to extract features from
            latent_dim: Final latent dimension
        """
        super().__init__(arch, pretrained, freeze_backbone, output_layers, latent_dim)
        
        # Load pretrained VGG
        if arch == 'vgg11':
            model = models.vgg11(pretrained=pretrained)
        elif arch == 'vgg13':
            model = models.vgg13(pretrained=pretrained)
        elif arch == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif arch == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown VGG architecture: {arch}")
        
        # Use only feature extraction part
        self.backbone = model.features
        backbone_dim = 512
        
        # Final FC layer for latent representation
        self.fc = nn.Linear(backbone_dim, latent_dim)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _extract_features(self, x):
        """Extract features from VGG backbone"""
        if not self.output_layers:
            return self.backbone(x)
        
        # Extract intermediate features
        features = {}
        current_x = x
        
        for i, layer in enumerate(self.backbone):
            current_x = layer(current_x)
            if i in self.output_layers:
                features[f'layer_{i}'] = current_x
        
        return features if features else current_x


class EfficientNetEncoder(PretrainedEncoder):
    """EfficientNet-based feature extractor"""
    
    def __init__(self, arch='efficientnet_b0', pretrained=True, freeze_backbone=False,
                 output_layers=None, latent_dim=512):
        """
        Args:
            arch: EfficientNet architecture ('efficientnet_b0' to 'efficientnet_b7')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone parameters  
            output_layers: Block names to extract features from
            latent_dim: Final latent dimension
        """
        super().__init__(arch, pretrained, freeze_backbone, output_layers, latent_dim)
        
        # Load pretrained EfficientNet
        try:
            if arch == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                backbone_dim = 1280
            elif arch == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                backbone_dim = 1280
            elif arch == 'efficientnet_b2':
                self.backbone = models.efficientnet_b2(pretrained=pretrained)
                backbone_dim = 1408
            elif arch == 'efficientnet_b3':
                self.backbone = models.efficientnet_b3(pretrained=pretrained)
                backbone_dim = 1536
            elif arch == 'efficientnet_b4':
                self.backbone = models.efficientnet_b4(pretrained=pretrained)
                backbone_dim = 1792
            elif arch == 'efficientnet_b5':
                self.backbone = models.efficientnet_b5(pretrained=pretrained)
                backbone_dim = 2048
            elif arch == 'efficientnet_b6':
                self.backbone = models.efficientnet_b6(pretrained=pretrained)
                backbone_dim = 2304
            elif arch == 'efficientnet_b7':
                self.backbone = models.efficientnet_b7(pretrained=pretrained)
                backbone_dim = 2560
            else:
                raise ValueError(f"Unknown EfficientNet architecture: {arch}")
        except AttributeError:
            raise ValueError(f"EfficientNet {arch} not available in this torchvision version")
        
        # Use only feature extraction part (remove classifier)
        self.backbone = self.backbone.features
        
        # Final FC layer for latent representation
        self.fc = nn.Linear(backbone_dim, latent_dim)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _extract_features(self, x):
        """Extract features from EfficientNet backbone"""
        if not self.output_layers:
            return self.backbone(x)
        
        # Extract intermediate features from blocks
        features = {}
        current_x = x
        
        for i, block in enumerate(self.backbone):
            current_x = block(current_x)
            block_name = f'block_{i}'
            if block_name in self.output_layers:
                features[block_name] = current_x
        
        return features if features else current_x


def get_pretrained_encoder(backbone_name, pretrained=True, freeze_backbone=False,
                          output_layers=None, latent_dim=512):
    """Factory function to get pretrained encoder"""
    
    # ResNet models
    resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if backbone_name in resnet_models:
        return ResNetEncoder(backbone_name, pretrained, freeze_backbone, output_layers, latent_dim)
    
    # VGG models  
    vgg_models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    if backbone_name in vgg_models:
        return VGGEncoder(backbone_name, pretrained, freeze_backbone, output_layers, latent_dim)
    
    # EfficientNet models
    efficientnet_models = [f'efficientnet_b{i}' for i in range(8)]
    if backbone_name in efficientnet_models:
        return EfficientNetEncoder(backbone_name, pretrained, freeze_backbone, output_layers, latent_dim)
    
    raise ValueError(f"Unknown backbone: {backbone_name}. "
                     f"Available: {resnet_models + vgg_models + efficientnet_models}")


if __name__ == "__main__":
    # Test pretrained encoders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 224, 224).to(device)
    
    # Test ResNet encoder
    print("Testing ResNet encoder:")
    resnet_encoder = get_pretrained_encoder('resnet50', pretrained=False).to(device)
    latent, features = resnet_encoder(x)
    print(f"Input: {x.shape} -> Latent: {latent.shape}, Features: {features.shape}")
    
    # Test VGG encoder
    print("\nTesting VGG encoder:")
    vgg_encoder = get_pretrained_encoder('vgg16', pretrained=False).to(device)
    latent, features = vgg_encoder(x)
    print(f"Input: {x.shape} -> Latent: {latent.shape}, Features: {features.shape}")
    
    # Test EfficientNet encoder  
    print("\nTesting EfficientNet encoder:")
    try:
        eff_encoder = get_pretrained_encoder('efficientnet_b0', pretrained=False).to(device)
        latent, features = eff_encoder(x)
        print(f"Input: {x.shape} -> Latent: {latent.shape}, Features: {features.shape}")
    except ValueError as e:
        print(f"EfficientNet test skipped: {e}")
    
    print("\nAll pretrained encoders working correctly!")