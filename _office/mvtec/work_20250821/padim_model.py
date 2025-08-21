import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis


class PaDiMFeatureExtractor(nn.Module):
    """Feature extractor using ResNet for PaDiM"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3'], weights_path=None):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        
        # Load backbone without pretrained weights
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            self.feature_dims = [64, 128, 256]
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            self.feature_dims = [256, 512, 1024]
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
        
        # Hook handles
        self.hook_handles = []
        self.features = {}
        
        # Register hooks
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
        """Extract features from multiple layers"""
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


class PaDiM(nn.Module):
    """PaDiM: Patch Distribution Modeling for Anomaly Detection"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3'], 
                 d_reduced=100, epsilon=0.04, weights_path=None):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self.d_reduced = d_reduced
        self.epsilon = epsilon
        self.model_type = "padim"
        
        # Feature extractor
        self.feature_extractor = PaDiMFeatureExtractor(backbone, layers, weights_path)
        
        # Patch embedding parameters (will be set during training)
        self.embedding_vectors = None
        self.mean_vectors = None
        self.covariance_matrices = None
        self.random_projection = None
        
        # Total feature dimension
        self.total_dim = sum(self.feature_extractor.feature_dims)
        
    def _generate_random_projection(self):
        """Generate random projection matrix for dimension reduction"""
        if self.d_reduced < self.total_dim:
            # Random sampling indices for dimension reduction
            indices = torch.randperm(self.total_dim)[:self.d_reduced]
            self.random_projection = indices
        else:
            self.random_projection = None
    
    def _extract_patch_embeddings(self, features_list):
        """Extract and concatenate patch embeddings from multi-scale features"""
        batch_size = features_list[0].size(0)
        
        # Resize all features to the same spatial size (use the largest)
        target_size = max([f.size(-1) for f in features_list])
        
        resized_features = []
        for features in features_list:
            if features.size(-1) != target_size:
                features = F.interpolate(features, size=(target_size, target_size), 
                                       mode='bilinear', align_corners=False)
            resized_features.append(features)
        
        # Concatenate features along channel dimension
        concatenated = torch.cat(resized_features, dim=1)  # [B, C_total, H, W]
        
        # Reshape to get patch embeddings
        patch_embeddings = concatenated.permute(0, 2, 3, 1)  # [B, H, W, C_total]
        patch_embeddings = patch_embeddings.reshape(-1, self.total_dim)  # [B*H*W, C_total]
        
        # Apply random projection for dimension reduction
        if self.random_projection is not None:
            patch_embeddings = patch_embeddings[:, self.random_projection]
        
        return patch_embeddings, (batch_size, target_size, target_size)
    
    def forward(self, x):
        """Forward pass for PaDiM"""
        # Extract multi-scale features
        features_list = self.feature_extractor(x)
        
        # Extract patch embeddings
        patch_embeddings, shape_info = self._extract_patch_embeddings(features_list)
        
        outputs = {
            'patch_embeddings': patch_embeddings,
            'shape_info': shape_info,
            'input': x
        }
        
        # If model is fitted and in eval mode, also compute anomaly scores
        if not self.training and self.mean_vectors is not None:
            try:
                anomaly_scores = self.compute_anomaly_scores(patch_embeddings, shape_info)
                outputs['anomaly_scores'] = anomaly_scores
            except Exception as e:
                print(f"Warning: Could not compute PaDiM anomaly scores: {e}")
        
        return outputs
    
    def fit_distribution(self, train_embeddings):
        """Fit Gaussian distribution to training patch embeddings"""
        # Convert to numpy for sklearn
        embeddings_np = train_embeddings.cpu().numpy()
        
        # Compute mean
        self.mean_vectors = np.mean(embeddings_np, axis=0)
        
        # Compute covariance with Ledoit-Wolf regularization
        cov_estimator = LedoitWolf()
        self.covariance_matrices = cov_estimator.fit(embeddings_np).covariance_
        
        # Add small epsilon to diagonal for numerical stability
        self.covariance_matrices += self.epsilon * np.eye(self.covariance_matrices.shape[0])
        
        # Compute inverse covariance for Mahalanobis distance
        self.inv_covariance = np.linalg.inv(self.covariance_matrices)
    
    def compute_anomaly_scores(self, patch_embeddings, shape_info):
        """Compute anomaly scores using Mahalanobis distance"""
        if self.mean_vectors is None or self.inv_covariance is None:
            raise RuntimeError("Model not fitted. Call fit_distribution first.")
        
        batch_size, height, width = shape_info
        embeddings_np = patch_embeddings.cpu().numpy()
        
        # Compute Mahalanobis distance for each patch
        scores = []
        for embedding in embeddings_np:
            score = mahalanobis(embedding, self.mean_vectors, self.inv_covariance)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Reshape to spatial dimensions
        scores = scores.reshape(batch_size, height, width)
        
        # Aggregate scores (max pooling or average)
        image_scores = np.max(scores.reshape(batch_size, -1), axis=1)
        
        return torch.tensor(image_scores, dtype=torch.float32)
    
    def compute_loss(self, outputs, loss_fn_dict):
        """PaDiM doesn't use traditional loss during training"""
        # Return dummy loss since PaDiM doesn't train with gradients
        device = next(self.parameters()).device
        losses = {
            'total': torch.tensor(0.0, requires_grad=True, device=device)
        }
        
        # Add any specific loss names from loss_fn_dict for compatibility
        for loss_name in loss_fn_dict.keys():
            if loss_name != 'total':
                losses[loss_name] = torch.tensor(0.0, requires_grad=True, device=device)
                
        return losses


class PaDiMTrainingWrapper:
    """Wrapper for PaDiM training process"""
    def __init__(self, model):
        self.model = model
        self.train_embeddings = []
        
    def collect_embeddings(self, data_loader):
        """Collect patch embeddings from training data"""
        self.model.eval()
        self.train_embeddings = []
        
        print("Collecting patch embeddings from training data...")
        with torch.no_grad():
            for data in data_loader:
                images = data['image']
                labels = data['label']
                
                # Use only normal samples
                normal_mask = labels == 0
                if not normal_mask.any():
                    continue
                
                normal_images = images[normal_mask]
                if torch.cuda.is_available():
                    normal_images = normal_images.cuda()
                
                # Forward pass
                outputs = self.model(normal_images)
                embeddings = outputs['patch_embeddings']
                
                self.train_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        self.train_embeddings = torch.cat(self.train_embeddings, dim=0)
        
        # Generate random projection if needed
        if self.model.random_projection is None:
            self.model._generate_random_projection()
        
        print(f"Collected {len(self.train_embeddings)} patch embeddings")
        
    def fit_distribution(self):
        """Fit Gaussian distribution to collected embeddings"""
        if len(self.train_embeddings) == 0:
            raise RuntimeError("No embeddings collected. Call collect_embeddings first.")
        
        print("Fitting Gaussian distribution...")
        self.model.fit_distribution(self.train_embeddings)
        print("Distribution fitting completed!")
        
    def train(self, train_loader):
        """Complete training process for PaDiM"""
        self.collect_embeddings(train_loader)
        self.fit_distribution()
        
        # Return dummy history for compatibility with other models
        return {
            'total': [0.0],
            'val_total': [0.0],
            'feature_mag': [0.0],
            'val_feature_mag': [0.0]
        }
