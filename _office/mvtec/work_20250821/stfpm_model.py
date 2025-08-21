import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class FeaturePyramidExtractor(nn.Module):
    """Feature pyramid extractor for STFPM"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3'], weights_path=None):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        
        # Load backbone without pretrained weights
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=None)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=None)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Load custom weights if provided
        if weights_path is not None:
            self._load_custom_weights(weights_path)
        
        # Remove classifier layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Register hooks for feature extraction
        self.features = {}
        self.hook_handles = []
        self._register_hooks()
    
    def _load_custom_weights(self, weights_path):
        """Load custom pretrained weights from file"""
        try:
            print(f"Loading custom weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print("Custom weights loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load custom weights from {weights_path}: {e}")
            print("Using random initialization instead")
    
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        for name, module in self.backbone.named_modules():
            if any(layer in name for layer in self.layers):
                handle = module.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)
    
    def forward(self, x):
        """Extract feature pyramid"""
        self.features.clear()
        _ = self.backbone(x)
        
        # Collect features from specified layers
        extracted_features = []
        for layer in self.layers:
            for name, feature in self.features.items():
                if layer in name:
                    extracted_features.append(feature)
                    break
        
        return extracted_features
    
    def __del__(self):
        """Clean up hooks"""
        for handle in self.hook_handles:
            handle.remove()


class TeacherNetwork(FeaturePyramidExtractor):
    """Teacher network with custom weights (frozen)"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3'], weights_path=None):
        super().__init__(backbone, layers, weights_path)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()


class StudentNetwork(FeaturePyramidExtractor):
    """Student network with random initialization (trainable)"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3']):
        super().__init__(backbone, layers, weights_path=None)  # No pretrained weights for student
        
        # Initialize weights randomly (default PyTorch initialization)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using default PyTorch initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class STFPM(nn.Module):
    """Student-Teacher Feature Pyramid Matching for Anomaly Detection"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3'], weights_path=None):
        super().__init__()
        self.model_type = "stfpm"
        self.backbone_name = backbone
        self.layers = layers
        
        # Teacher network (with optional custom weights, frozen)
        self.teacher = TeacherNetwork(backbone, layers, weights_path)
        
        # Student network (random init, trainable)  
        self.student = StudentNetwork(backbone, layers)
        
        # Feature dimensions for each layer
        self._get_feature_dimensions()
    
    def _get_feature_dimensions(self):
        """Get feature dimensions for each layer"""
        if self.backbone_name == 'resnet18':
            dim_map = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        elif self.backbone_name == 'resnet34':
            dim_map = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        elif self.backbone_name == 'resnet50':
            dim_map = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        self.feature_dims = [dim_map[layer] for layer in self.layers]
    
    def forward(self, x):
        """Forward pass through both teacher and student networks"""
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_features = self.teacher(x)
        
        # Student forward pass
        student_features = self.student(x)
        
        return {
            'teacher_features': teacher_features,
            'student_features': student_features,
            'input': x
        }
    
    def compute_loss(self, outputs, loss_fn_dict):
        """Compute feature matching loss between teacher and student"""
        teacher_features = outputs['teacher_features']
        student_features = outputs['student_features']
        
        losses = {}
        total_loss = 0
        
        # Feature matching loss at each scale
        for i, (teacher_feat, student_feat) in enumerate(zip(teacher_features, student_features)):
            # L2 loss between teacher and student features
            feat_loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += feat_loss
            losses[f'feat_loss_{i}'] = feat_loss
        
        losses['feature_matching'] = total_loss
        losses['total'] = total_loss
        return losses
    
    def compute_anomaly_scores(self, outputs):
        """Compute anomaly scores from teacher-student feature differences"""
        teacher_features = outputs['teacher_features']
        student_features = outputs['student_features']
        
        anomaly_maps = []
        
        # Compute feature differences at each scale
        for teacher_feat, student_feat in zip(teacher_features, student_features):
            # Compute squared differences
            diff = (teacher_feat - student_feat) ** 2
            
            # Aggregate across channels
            anomaly_map = torch.mean(diff, dim=1, keepdim=True)  # [B, 1, H, W]
            anomaly_maps.append(anomaly_map)
        
        # Resize all anomaly maps to the same size and combine
        target_size = anomaly_maps[0].shape[-2:]  # Use first layer size as target
        
        combined_map = None
        for i, anomaly_map in enumerate(anomaly_maps):
            if anomaly_map.shape[-2:] != target_size:
                anomaly_map = F.interpolate(anomaly_map, size=target_size, 
                                          mode='bilinear', align_corners=False)
            
            if combined_map is None:
                combined_map = anomaly_map
            else:
                combined_map += anomaly_map
        
        # Global average pooling to get image-level scores
        image_scores = F.adaptive_avg_pool2d(combined_map, (1, 1))
        image_scores = image_scores.squeeze()  # [B]
        
        return image_scores, combined_map
    
    def train(self, mode=True):
        """Set training mode (only student network should be trainable)"""
        super().train(mode)
        
        # Teacher is always in eval mode
        self.teacher.eval()
        
        # Only student should be trainable
        if mode:
            self.student.train()
        else:
            self.student.eval()
        
        return self


class STFPMLoss:
    """STFPM loss function"""
    def __init__(self, weights=None):
        self.weights = weights or [1.0, 1.0, 1.0]  # Weights for different scales
    
    def __call__(self, teacher_features, student_features):
        """Compute weighted feature matching loss"""
        total_loss = 0
        
        for i, (teacher_feat, student_feat) in enumerate(zip(teacher_features, student_features)):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            feat_loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += weight * feat_loss
        
        return total_loss


class STFPMEvaluator:
    """Evaluator for STFPM model"""
    def __init__(self, model):
        self.model = model
    
    def compute_pixel_scores(self, outputs):
        """Compute pixel-level anomaly scores"""
        _, anomaly_map = self.model.compute_anomaly_scores(outputs)
        return anomaly_map
    
    def compute_image_scores(self, outputs):
        """Compute image-level anomaly scores"""
        image_scores, _ = self.model.compute_anomaly_scores(outputs)
        return image_scores
