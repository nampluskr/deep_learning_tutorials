import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using ResNet for FastFlow"""
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'], weights_path=None):
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
        
        # Freeze parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        
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
        """Extract features from specified layers"""
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


class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flow"""
    def __init__(self, channels, hidden_dim=512, mask_type='checkerboard'):
        super().__init__()
        self.channels = channels
        self.mask_type = mask_type
        
        # Create mask
        if mask_type == 'checkerboard':
            self.register_buffer('mask', self._create_checkerboard_mask(channels))
        elif mask_type == 'channel':
            self.register_buffer('mask', self._create_channel_mask(channels))
        
        # Transformation networks
        self.scale_net = self._create_transform_net(channels // 2, channels // 2, hidden_dim)
        self.translation_net = self._create_transform_net(channels // 2, channels // 2, hidden_dim)
    
    def _create_checkerboard_mask(self, channels):
        """Create checkerboard mask for spatial coupling"""
        mask = torch.zeros(1, channels, 1, 1)
        mask[:, ::2] = 1
        return mask
    
    def _create_channel_mask(self, channels):
        """Create channel-wise mask"""
        mask = torch.zeros(1, channels, 1, 1)
        mask[:, :channels//2] = 1
        return mask
    
    def _create_transform_net(self, input_dim, output_dim, hidden_dim):
        """Create transformation network"""
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )
    
    def forward(self, x, reverse=False):
        """Forward pass through coupling layer"""
        if not reverse:
            return self._forward(x)
        else:
            return self._inverse(x)
    
    def _forward(self, x):
        """Forward transformation"""
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        scale = self.scale_net(x_masked)
        translation = self.translation_net(x_masked)
        
        # Apply affine transformation
        scale = torch.tanh(scale)  # Stabilize scaling
        y_unmasked = x_unmasked * torch.exp(scale) + translation
        
        y = x_masked + y_unmasked
        
        # Compute log determinant
        log_det = torch.sum(scale, dim=[1, 2, 3])
        
        return y, log_det
    
    def _inverse(self, y):
        """Inverse transformation"""
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)
        
        scale = self.scale_net(y_masked)
        translation = self.translation_net(y_masked)
        
        # Apply inverse affine transformation
        scale = torch.tanh(scale)
        x_unmasked = (y_unmasked - translation) * torch.exp(-scale)
        
        x = y_masked + x_unmasked
        
        # Compute log determinant (negative of forward)
        log_det = -torch.sum(scale, dim=[1, 2, 3])
        
        return x, log_det


class NormalizingFlow(nn.Module):
    """Normalizing Flow for FastFlow"""
    def __init__(self, input_dim, n_flows=8, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.n_flows = n_flows
        
        # Create coupling layers
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            mask_type = 'checkerboard' if i % 2 == 0 else 'channel'
            self.flows.append(CouplingLayer(input_dim, hidden_dim, mask_type))
    
    def forward(self, x, reverse=False):
        """Forward pass through normalizing flow"""
        log_det_total = 0
        
        if not reverse:
            # Forward pass
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_total += log_det
        else:
            # Reverse pass
            for flow in reversed(self.flows):
                x, log_det = flow(x, reverse=True)
                log_det_total += log_det
        
        return x, log_det_total
    
    def log_prob(self, x):
        """Compute log probability of input"""
        z, log_det = self.forward(x, reverse=False)
        
        # Assume standard normal prior
        log_prior = -0.5 * torch.sum(z ** 2, dim=[1, 2, 3]) - \
                    0.5 * np.prod(z.shape[1:]) * np.log(2 * np.pi)
        
        log_prob = log_prior + log_det
        return log_prob


class FastFlow(nn.Module):
    """FastFlow: Real-time Anomaly Detection via Normalizing Flow"""
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'], 
                 flow_steps=8, hidden_dim=512, weights_path=None):
        super().__init__()
        self.model_type = "fastflow"
        self.backbone_name = backbone
        self.layers = layers
        
        # Feature extractor
        self.feature_extractor = ResNetFeatureExtractor(backbone, layers, weights_path)
        
        # Get feature dimensions
        self._get_feature_dimensions()
        
        # Normalizing flows for each scale
        self.flows = nn.ModuleDict()
        for i, (layer, dim) in enumerate(zip(layers, self.feature_dims)):
            self.flows[f'flow_{i}'] = NormalizingFlow(dim, flow_steps, hidden_dim)
    
    def _get_feature_dimensions(self):
        """Get feature dimensions from backbone"""
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
        """Forward pass for FastFlow"""
        # Extract multi-scale features
        features_list = self.feature_extractor(x)
        
        # Apply normalizing flows
        log_probs = []
        flow_outputs = []
        
        for i, features in enumerate(features_list):
            flow = self.flows[f'flow_{i}']
            log_prob = flow.log_prob(features)
            log_probs.append(log_prob)
            flow_outputs.append(features)
        
        return {
            'log_probs': log_probs,
            'features': flow_outputs,
            'input': x
        }
    
    def compute_loss(self, outputs, loss_fn_dict):
        """Compute negative log-likelihood loss"""
        losses = {}
        
        # Initialize total_loss as tensor with gradient
        total_loss = torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
        # Negative log-likelihood loss
        for log_prob in outputs['log_probs']:
            nll = -torch.mean(log_prob)
            total_loss = total_loss + nll
        
        losses['nll'] = total_loss
        losses['total'] = total_loss
        
        return losses
    
    def compute_anomaly_scores(self, outputs):
        """Compute anomaly scores from log probabilities"""
        # Combine log probabilities from all scales
        total_log_prob = 0
        for log_prob in outputs['log_probs']:
            # Aggregate spatial dimensions
            log_prob_mean = torch.mean(log_prob.view(log_prob.size(0), -1), dim=1)
            total_log_prob += log_prob_mean
        
        # Negative log probability as anomaly score
        anomaly_scores = -total_log_prob
        return anomaly_scores


class FastFlowLoss:
    """FastFlow loss function"""
    def __init__(self):
        pass
    
    def __call__(self, outputs):
        """Compute negative log-likelihood loss"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize total_loss as tensor
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        if 'log_probs' in outputs and len(outputs['log_probs']) > 0:
            for log_prob in outputs['log_probs']:
                nll = -torch.mean(log_prob)
                total_loss = total_loss + nll
        else:
            # If no log_probs, return a small loss to avoid issues
            total_loss = torch.tensor(1e-6, requires_grad=True, device=device)
        
        return total_loss
