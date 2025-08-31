import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Tuple
import random
import numpy as np
from sklearn.mixture import GaussianMixture
from .model_base import TimmFeatureExtractor


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


class CutPasteAugmentation:
    """Cut-Paste augmentation for self-supervised anomaly detection."""
    
    def __init__(self, 
                 cut_size_ratio=(0.02, 0.15),
                 paste_number_range=(1, 4),
                 paste_prob=1.0,
                 rotation_angle_range=(-45, 45)):
        """
        Args:
            cut_size_ratio: Min and max ratio of cut patch size relative to image size
            paste_number_range: Min and max number of patches to paste
            paste_prob: Probability of applying cut-paste augmentation
            rotation_angle_range: Range of rotation angles for cut patches
        """
        self.cut_size_ratio = cut_size_ratio
        self.paste_number_range = paste_number_range
        self.paste_prob = paste_prob
        self.rotation_angle_range = rotation_angle_range
    
    def cut_patch(self, image, patch_size):
        """Cut a random patch from the image."""
        h, w = image.shape[-2:]
        
        # Random position for cutting
        cut_x = random.randint(0, max(1, w - patch_size[1]))
        cut_y = random.randint(0, max(1, h - patch_size[0]))
        
        # Extract patch
        patch = image[:, cut_y:cut_y + patch_size[0], cut_x:cut_x + patch_size[1]].clone()
        
        return patch, (cut_x, cut_y)
    
    def paste_patch(self, image, patch, avoid_area=None):
        """Paste patch onto image at random location."""
        h, w = image.shape[-2:]
        patch_h, patch_w = patch.shape[-2:]
        
        # Find valid paste location (avoiding the cut area)
        max_attempts = 20
        for _ in range(max_attempts):
            paste_x = random.randint(0, max(1, w - patch_w))
            paste_y = random.randint(0, max(1, h - patch_h))
            
            # Check if paste area overlaps with avoid area
            if avoid_area is not None:
                cut_x, cut_y = avoid_area
                if (paste_x < cut_x + patch_w and paste_x + patch_w > cut_x and
                    paste_y < cut_y + patch_h and paste_y + patch_h > cut_y):
                    continue  # Skip this location
            
            # Paste the patch
            result_image = image.clone()
            result_image[:, paste_y:paste_y + patch_h, paste_x:paste_x + patch_w] = patch
            
            return result_image, (paste_x, paste_y)
        
        # If no valid location found, paste at random location
        paste_x = random.randint(0, max(1, w - patch_w))
        paste_y = random.randint(0, max(1, h - patch_h))
        result_image = image.clone()
        result_image[:, paste_y:paste_y + patch_h, paste_x:paste_x + patch_w] = patch
        
        return result_image, (paste_x, paste_y)
    
    def rotate_patch(self, patch, angle):
        """Rotate patch by given angle."""
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Create rotation matrix
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Simple rotation using affine transformation
        # For simplicity, we'll just return the original patch
        # In practice, you might want to implement proper rotation
        return patch
    
    def apply_cutpaste(self, image):
        """Apply cut-paste augmentation to a single image."""
        if random.random() > self.paste_prob:
            return image, torch.tensor(0)  # No augmentation
        
        h, w = image.shape[-2:]
        result_image = image.clone()
        
        # Number of patches to paste
        num_patches = random.randint(*self.paste_number_range)
        
        for _ in range(num_patches):
            # Random patch size
            min_ratio, max_ratio = self.cut_size_ratio
            patch_ratio = random.uniform(min_ratio, max_ratio)
            patch_size = (
                int(h * patch_ratio * random.uniform(0.7, 1.3)),
                int(w * patch_ratio * random.uniform(0.7, 1.3))
            )
            patch_size = (min(patch_size[0], h), min(patch_size[1], w))
            
            # Cut patch from random location
            patch, cut_location = self.cut_patch(result_image, patch_size)
            
            # Optionally rotate patch
            if self.rotation_angle_range:
                angle = random.uniform(*self.rotation_angle_range)
                patch = self.rotate_patch(patch, angle)
            
            # Paste patch to different location
            result_image, _ = self.paste_patch(result_image, patch, avoid_area=cut_location)
        
        return result_image, torch.tensor(1)  # Augmented
    
    def __call__(self, images):
        """Apply cut-paste augmentation to batch of images."""
        batch_size = images.shape[0]
        augmented_images = []
        labels = []
        
        for i in range(batch_size):
            aug_img, label = self.apply_cutpaste(images[i])
            augmented_images.append(aug_img)
            labels.append(label)
        
        return torch.stack(augmented_images), torch.stack(labels)


class CutPasteClassifier(nn.Module):
    """Binary classifier for Cut-Paste augmented images."""
    
    def __init__(self, backbone="resnet18", pretrained=True, num_classes=2):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            layers=["layer4"],  # Use final layer for classification
            pre_trained=pretrained,
            requires_grad=True
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.feature_extractor(dummy_input)
            self.feature_dim = features["layer4"].shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # For feature extraction during inference
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, return_features=False):
        """Forward pass through classifier."""
        features = self.feature_extractor(x)
        layer4_features = features["layer4"]
        
        # Classification
        logits = self.classifier[1:](self.classifier[0](layer4_features))  # Skip pooling in classifier
        
        if return_features:
            # Return pooled features for density modeling
            pooled_features = self.feature_pool(layer4_features).flatten(1)
            return logits, pooled_features
        
        return logits
    
    def extract_features(self, x):
        """Extract features for density modeling."""
        with torch.no_grad():
            features = self.feature_extractor(x)
            layer4_features = features["layer4"]
            pooled_features = self.feature_pool(layer4_features).flatten(1)
            return pooled_features


class CutPasteModel(nn.Module):
    """CutPaste model for anomaly detection."""
    
    def __init__(self, 
                 backbone="resnet18", 
                 pretrained=True,
                 cut_size_ratio=(0.02, 0.15),
                 paste_number_range=(1, 4)):
        super().__init__()
        
        self.classifier = CutPasteClassifier(backbone=backbone, pretrained=pretrained)
        self.augmentation = CutPasteAugmentation(
            cut_size_ratio=cut_size_ratio,
            paste_number_range=paste_number_range
        )
        
        # GMM for density modeling (fitted during post-processing)
        self.gmm = None
        self.fitted = False
        
    def forward(self, images):
        """Forward pass - different behavior for training and inference."""
        if self.training:
            # Training mode: apply augmentation and classify
            augmented_images, labels = self.augmentation(images)
            
            # Create combined batch: [original_images, augmented_images]
            combined_images = torch.cat([images, augmented_images], dim=0)
            
            # Create combined labels: [0s for original, 1s for augmented]
            original_labels = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
            augmented_labels = labels.to(device=images.device, dtype=torch.long)
            combined_labels = torch.cat([original_labels, augmented_labels], dim=0)
            
            # Classification
            logits = self.classifier(combined_images)
            
            return {
                'logits': logits,
                'labels': combined_labels,
                'augmented_images': augmented_images,
                'original_images': images
            }
        else:
            # Inference mode: extract features and compute anomaly scores
            if not self.fitted:
                # If GMM not fitted, return zero scores
                batch_size = images.shape[0]
                return InferenceBatch(
                    pred_score=torch.zeros(batch_size, device=images.device),
                    anomaly_map=torch.zeros(batch_size, 1, *images.shape[-2:], device=images.device)
                )
            
            # Extract features
            features = self.classifier.extract_features(images)
            
            # Compute anomaly scores using GMM
            features_np = features.cpu().numpy()
            log_probs = self.gmm.score_samples(features_np)
            anomaly_scores = -log_probs  # Negative log-likelihood as anomaly score
            
            # Convert to tensor
            pred_scores = torch.from_numpy(anomaly_scores).float().to(images.device)
            
            # Create anomaly maps (uniform for now, could be improved with patch-based analysis)
            anomaly_maps = pred_scores.view(-1, 1, 1, 1).expand(-1, 1, *images.shape[-2:])
            
            return InferenceBatch(
                pred_score=pred_scores,
                anomaly_map=anomaly_maps
            )
    
    def fit_gmm(self, normal_images, n_components=1):
        """Fit GMM to normal image features."""
        self.eval()
        
        all_features = []
        with torch.no_grad():
            # Extract features from normal images
            for i in range(0, len(normal_images), 32):  # Process in batches
                batch = normal_images[i:i+32]
                if isinstance(batch, list):
                    batch = torch.stack(batch)
                features = self.classifier.extract_features(batch)
                all_features.append(features.cpu().numpy())
        
        # Combine all features
        features_np = np.concatenate(all_features, axis=0)
        
        # Fit GMM
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.gmm.fit(features_np)
        self.fitted = True
        
        return features_np


class CutPasteLoss(nn.Module):
    """Binary cross-entropy loss for CutPaste classification."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets=None):
        """Compute classification loss."""
        if isinstance(outputs, dict):
            logits = outputs['logits']
            labels = outputs['labels']
        else:
            logits = outputs
            labels = targets
            
        return self.criterion(logits, labels)


if __name__ == "__main__":
    # Test CutPaste model
    model = CutPasteModel(backbone="resnet18")
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Training mode
    model.train()
    train_output = model(x)
    print("Training output keys:", train_output.keys())
    print("Logits shape:", train_output['logits'].shape)
    print("Labels shape:", train_output['labels'].shape)
    
    # Test loss
    loss_fn = CutPasteLoss()
    loss = loss_fn(train_output)
    print(f"Training loss: {loss.item():.4f}")
    
    # Inference mode (without fitted GMM)
    model.eval()
    with torch.no_grad():
        inference_output = model(x)
        print(f"Inference pred_score shape: {inference_output.pred_score.shape}")
        print(f"Inference anomaly_map shape: {inference_output.anomaly_map.shape}")