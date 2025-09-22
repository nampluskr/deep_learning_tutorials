import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional


# Backbone directory path (from existing code)
BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
}


def get_local_weight_path(backbone: str) -> str:
    """Get local weight path for backbone model."""
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


class EfficientADEncoder(nn.Module):
    """ResNet-based encoder for EfficientAD."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        
        # Load ResNet model
        if backbone == "resnet18":
            self.model = models.resnet18(weights=None)
            self.feature_dim = 512
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=None)
            self.feature_dim = 512
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Load pretrained weights
        if pretrained:
            weight_path = get_local_weight_path(backbone)
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                print(f" > Loaded pretrained weights for {backbone} from {weight_path}")
            else:
                print(f" > No local weight file found for {backbone}, using random init.")
        
        # Extract feature layers
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool
        )
        
        # Remove classifier
        self.model = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        features = self.features(x)
        features = torch.flatten(features, 1)
        return features


class AutoEncoderDecoder(nn.Module):
    """Decoder for EfficientAD autoencoder."""
    
    def __init__(self, feature_dim: int, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        
        # Calculate initial feature map size
        init_size = img_size // 32  # ResNet downsamples by 32x
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512 * init_size * init_size),
            nn.Unflatten(1, (512, init_size, init_size)),
            
            # Upsample blocks
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to image."""
        return self.decoder(x)


class StudentEncoder(nn.Module):
    """Student encoder for knowledge distillation."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Lightweight CNN encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(512, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract student features."""
        return self.encoder(x)


class EfficientAD(nn.Module):
    """EfficientAD model combining autoencoder and knowledge distillation."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, 
                 img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.backbone_name = backbone
        
        # Teacher encoder (frozen)
        self.teacher_encoder = EfficientADEncoder(backbone, pretrained)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        
        # Student encoder (trainable)
        self.student_encoder = StudentEncoder(self.teacher_encoder.feature_dim)
        
        # Autoencoder decoder (trainable)
        self.decoder = AutoEncoderDecoder(self.teacher_encoder.feature_dim, img_size)
        
        # Feature dimension
        self.feature_dim = self.teacher_encoder.feature_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            reconstructed: Reconstructed image
            teacher_features: Teacher encoder features
            student_features: Student encoder features
        """
        # Teacher features (frozen)
        with torch.no_grad():
            teacher_features = self.teacher_encoder(x)
        
        # Student features (trainable)
        student_features = self.student_encoder(x)
        
        # Reconstruction using teacher features
        reconstructed = self.decoder(teacher_features)
        
        return reconstructed, teacher_features, student_features
    
    def compute_anomaly_map(self, reconstructed: torch.Tensor, 
                           original: torch.Tensor,
                           teacher_features: torch.Tensor,
                           student_features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly map from reconstruction error and feature difference."""
        
        # Reconstruction error map
        recon_error = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        
        # Feature difference (knowledge distillation error)
        feature_diff = torch.mean((teacher_features - student_features) ** 2, dim=1)
        
        # Resize feature difference to match image size
        feature_diff = feature_diff.view(-1, 1, 1, 1)
        feature_diff = feature_diff.expand(-1, 1, self.img_size, self.img_size)
        
        # Combine reconstruction error and feature difference
        alpha = 0.5  # Weight for combining two error types
        anomaly_map = alpha * recon_error + (1 - alpha) * feature_diff
        
        return anomaly_map
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute image-level anomaly score."""
        # Use max pooling to get the highest anomaly score
        score = torch.amax(anomaly_map, dim=(-2, -1))
        return score
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict anomaly maps and scores."""
        self.eval()
        reconstructed, teacher_features, student_features = self.forward(images)
        anomaly_map = self.compute_anomaly_map(
            reconstructed, images, teacher_features, student_features
        )
        pred_score = self.compute_anomaly_score(anomaly_map)
        
        return {
            "anomaly_map": anomaly_map,
            "pred_score": pred_score,
            "reconstructed": reconstructed
        }


class EfficientADLoss(nn.Module):
    """Combined loss for EfficientAD training."""
    
    def __init__(self, recon_weight: float = 0.5, distill_weight: float = 0.5):
        super().__init__()
        self.recon_weight = recon_weight
        self.distill_weight = distill_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor,
                teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            reconstructed: Reconstructed images
            original: Original images
            teacher_features: Teacher encoder features
            student_features: Student encoder features
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(reconstructed, original)
        
        # Knowledge distillation loss
        distill_loss = self.mse_loss(student_features, teacher_features)
        
        # Combined loss
        total_loss = self.recon_weight * recon_loss + self.distill_weight * distill_loss
        
        return total_loss


class EfficientADMetric(nn.Module):
    """Metrics for EfficientAD evaluation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor,
                teacher_features: torch.Tensor, student_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics.
        
        Returns:
            Dictionary containing various metrics
        """
        metrics = {}
        
        # Reconstruction MSE
        recon_mse = F.mse_loss(reconstructed, original)
        metrics['recon_mse'] = recon_mse
        
        # Feature similarity (cosine similarity)
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)
        student_flat = student_features.view(student_features.size(0), -1)
        
        cosine_sim = F.cosine_similarity(teacher_flat, student_flat, dim=1).mean()
        metrics['feature_sim'] = cosine_sim
        
        # Feature MSE
        feature_mse = F.mse_loss(student_features, teacher_features)
        metrics['feature_mse'] = feature_mse
        
        return metrics
