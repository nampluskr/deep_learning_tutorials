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


def gat_backbone_path(backbone: str) -> str:
    """Get local weight path for backbone model."""
    if backbone in BACKBONE_WEIGHT_FILES:
        filename = BACKBONE_WEIGHT_FILES[backbone]
        return os.path.join(BACKBONE_DIR, filename)
    else:
        return os.path.join(BACKBONE_DIR, f"{backbone}.pth")


class ResNetFeatureExtractor(nn.Module):
    """ResNet feature extractor for STFPM."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        
        # Load ResNet model
        if backbone == "resnet18":
            self.model = models.resnet18(weights=None)
            self.feature_dims = [64, 128, 256, 512]
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=None)
            self.feature_dims = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=None)
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Load pretrained weights
        if pretrained:
            weight_path = gat_backbone_path(backbone)
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                print(f" > Loaded pretrained weights for {backbone} from {weight_path}")
            else:
                print(f" > No local weight file found for {backbone}, using random init.")
        
        # Extract feature layers
        self.layer1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1
        )
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
        # Remove classifier
        self.model = None
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []
        
        x = self.layer1(x)  # 1/4 resolution
        features.append(x)
        
        x = self.layer2(x)  # 1/8 resolution
        features.append(x)
        
        x = self.layer3(x)  # 1/16 resolution
        features.append(x)
        
        x = self.layer4(x)  # 1/32 resolution
        features.append(x)
        
        return features


class StudentNetwork(nn.Module):
    """Student network for STFPM."""
    
    def __init__(self, feature_dims: List[int]):
        super().__init__()
        self.feature_dims = feature_dims
        
        # Student networks for each feature level
        self.students = nn.ModuleList()
        for dim in feature_dims:
            student = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
            self.students.append(student)
    
    def forward(self, teacher_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate student features from teacher features."""
        student_features = []
        for i, feat in enumerate(teacher_features):
            student_feat = self.students[i](feat)
            student_features.append(student_feat)
        return student_features


class STFPM(nn.Module):
    """STFPM (Student-Teacher Feature Pyramid Matching) model."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        
        # Teacher network (frozen)
        self.teacher = ResNetFeatureExtractor(backbone, pretrained)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Student network (trainable)
        self.student = StudentNetwork(self.teacher.feature_dims)
        
        # Feature pyramid levels to use
        self.feature_levels = [0, 1, 2]  # Use first 3 levels
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass returning teacher and student features."""
        with torch.no_grad():
            teacher_features = self.teacher(x)
        
        student_features = self.student(teacher_features)
        
        return teacher_features, student_features
    
    def compute_anomaly_map(self, teacher_features: List[torch.Tensor], 
                           student_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute anomaly map from feature differences."""
        anomaly_maps = []
        
        for level in self.feature_levels:
            teacher_feat = teacher_features[level]
            student_feat = student_features[level]
            
            # Compute L2 distance
            diff = torch.pow(teacher_feat - student_feat, 2)
            anomaly_map = torch.mean(diff, dim=1, keepdim=True)
            
            # Upsample to input resolution
            if level == 0:
                target_size = (anomaly_map.size(2) * 4, anomaly_map.size(3) * 4)
            elif level == 1:
                target_size = (anomaly_map.size(2) * 8, anomaly_map.size(3) * 8)
            else:  # level == 2
                target_size = (anomaly_map.size(2) * 16, anomaly_map.size(3) * 16)
            
            anomaly_map = F.interpolate(
                anomaly_map, size=target_size, mode='bilinear', align_corners=False
            )
            anomaly_maps.append(anomaly_map)
        
        # Combine multi-scale anomaly maps
        # Resize all to the largest resolution
        max_h = max([amap.size(2) for amap in anomaly_maps])
        max_w = max([amap.size(3) for amap in anomaly_maps])
        
        resized_maps = []
        for amap in anomaly_maps:
            if amap.size(2) != max_h or amap.size(3) != max_w:
                amap = F.interpolate(amap, size=(max_h, max_w), mode='bilinear', align_corners=False)
            resized_maps.append(amap)
        
        # Average the maps
        final_anomaly_map = torch.mean(torch.stack(resized_maps), dim=0)
        return final_anomaly_map
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute image-level anomaly score from anomaly map."""
        # Use max pooling to get the highest anomaly score
        score = torch.amax(anomaly_map, dim=(-2, -1))
        return score
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict anomaly maps and scores."""
        self.eval()
        teacher_features, student_features = self.forward(images)
        anomaly_map = self.compute_anomaly_map(teacher_features, student_features)
        pred_score = self.compute_anomaly_score(anomaly_map)
        
        return {
            "anomaly_map": anomaly_map,
            "pred_score": pred_score
        }


class STFPMLoss(nn.Module):
    """Loss function for STFPM training."""
    
    def __init__(self, feature_levels: Optional[List[int]] = None):
        super().__init__()
        self.feature_levels = feature_levels or [0, 1, 2]
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, teacher_features: List[torch.Tensor], 
                student_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        total_loss = 0.0
        
        for level in self.feature_levels:
            teacher_feat = teacher_features[level]
            student_feat = student_features[level]
            
            # Normalize features
            teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
            student_norm = F.normalize(student_feat, p=2, dim=1)
            
            # Compute MSE loss
            loss = self.mse_loss(student_norm, teacher_norm)
            total_loss += loss
        
        return total_loss / len(self.feature_levels)


class STFPMMetric(nn.Module):
    """Metric for STFPM evaluation."""
    
    def __init__(self, feature_levels: Optional[List[int]] = None):
        super().__init__()
        self.feature_levels = feature_levels or [0, 1, 2]
    
    def forward(self, teacher_features: List[torch.Tensor], 
                student_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute cosine similarity between teacher and student features."""
        similarities = []
        
        for level in self.feature_levels:
            teacher_feat = teacher_features[level]
            student_feat = student_features[level]
            
            # Flatten features
            teacher_flat = teacher_feat.view(teacher_feat.size(0), -1)
            student_flat = student_feat.view(student_feat.size(0), -1)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(teacher_flat, student_flat, dim=1)
            similarities.append(similarity.mean())
        
        return torch.stack(similarities).mean()
