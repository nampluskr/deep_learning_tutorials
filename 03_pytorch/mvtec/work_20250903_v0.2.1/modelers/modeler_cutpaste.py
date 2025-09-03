import torch
from torch import optim
import numpy as np
from tqdm import tqdm

from .modeler_base import BaseModeler


class CutPasteModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)
        self._gmm_fitted = False

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        
        # Forward pass - model handles augmentation and creates training pairs
        outputs = self.model(inputs['image'])
        
        # CutPaste training: classify original vs cut-paste augmented
        loss = self.loss_fn(outputs)
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        
        # Calculate classification metrics
        with torch.no_grad():
            logits = outputs['logits']
            labels = outputs['labels']
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            results['accuracy'] = accuracy.item()
            
            # Separate accuracy for normal (0) and augmented (1) samples
            normal_mask = labels == 0
            augmented_mask = labels == 1
            
            if normal_mask.any():
                normal_correct = (predictions[normal_mask] == labels[normal_mask]).sum().item()
                normal_total = normal_mask.sum().item()
                results['neg_samples'] = normal_total
                results['correct_neg'] = normal_correct
            else:
                results['neg_samples'] = 0
                results['correct_neg'] = 0
            
            if augmented_mask.any():
                aug_correct = (predictions[augmented_mask] == labels[augmented_mask]).sum().item()
                aug_total = augmented_mask.sum().item()
                results['pos_samples'] = aug_total
                results['correct_pos'] = aug_correct
            else:
                results['pos_samples'] = 0
                results['correct_pos'] = 0
            
            # Confidence metrics
            probs = torch.softmax(logits, dim=1)
            normal_confidence = probs[normal_mask, 0].mean().item() if normal_mask.any() else 0.0
            aug_confidence = probs[augmented_mask, 1].mean().item() if augmented_mask.any() else 0.0
            
            results.update({
                'normal_confidence': normal_confidence,
                'aug_confidence': aug_confidence,
            })
            
            # Additional metrics from self.metrics
            for metric_name, metric_fn in self.metrics.items():
                if metric_name == 'accuracy' and hasattr(metric_fn, '__call__'):
                    # Skip redundant accuracy calculation
                    continue
                else:
                    try:
                        # Apply metric if applicable
                        results[metric_name] = 0.0  # Placeholder
                    except:
                        results[metric_name] = 0.0

        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        if not self._gmm_fitted:
            # During classification training phase, just return classification metrics
            outputs = self.model(inputs['image'])
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
                labels = outputs['labels']
                
                # Compute validation accuracy
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                
                results = {
                    'loss': 0.0,  # No validation loss during classification
                    'accuracy': accuracy.item(),
                }
            else:
                results = {'loss': 0.0, 'accuracy': 0.0}
        else:
            # After GMM fitting, evaluate anomaly detection performance
            inference_output = self.model(inputs['image'])
            
            if hasattr(inference_output, 'pred_score'):
                scores = inference_output.pred_score
                labels = inputs['label']
                
                # Score distribution analysis
                normal_mask = labels == 0
                anomaly_mask = labels == 1

                normal_scores = scores[normal_mask] if normal_mask.any() else torch.tensor([0.0])
                anomaly_scores = scores[anomaly_mask] if anomaly_mask.any() else torch.tensor([0.0])

                results = {
                    'loss': 0.0,
                    'score_mean': scores.mean().item(),
                    'score_std': scores.std().item(),
                    'normal_mean': normal_scores.mean().item(),
                    'anomaly_mean': anomaly_scores.mean().item(),
                    'separation': (anomaly_scores.mean() - normal_scores.mean()).item() if anomaly_mask.any() and normal_mask.any() else 0.0,
                }
            else:
                results = {'loss': 0.0}

        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        if not self._gmm_fitted:
            # If GMM not fitted yet, return zero scores
            batch_size = inputs['image'].shape[0]
            return torch.zeros(batch_size, device=self.device)

        predictions = self.model(inputs['image'])
        
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback
            return torch.zeros(inputs['image'].shape[0], device=self.device)

    def compute_anomaly_scores(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            if not self._gmm_fitted:
                # Return zero scores if GMM not fitted
                batch_size = inputs['image'].shape[0]
                image_size = inputs['image'].shape[-2:]
                return {
                    'anomaly_maps': torch.zeros(batch_size, 1, *image_size, device=self.device),
                    'pred_scores': torch.zeros(batch_size, device=self.device)
                }

            predictions = self.model(inputs['image'])
            
            if hasattr(predictions, 'anomaly_map') and hasattr(predictions, 'pred_score'):
                return {
                    'anomaly_maps': predictions.anomaly_map,
                    'pred_scores': predictions.pred_score
                }
            else:
                # Fallback computation
                batch_size = inputs['image'].shape[0]
                image_size = inputs['image'].shape[-2:]
                return {
                    'anomaly_maps': torch.zeros(batch_size, 1, *image_size, device=self.device),
                    'pred_scores': torch.zeros(batch_size, device=self.device)
                }

    def fit_density_model(self, train_loader):
        """Fit GMM to normal image features (Stage 2 of CutPaste)."""
        self.log(" > Extracting features from normal images...")
        
        self.model.eval()
        normal_images = []
        
        # Collect normal images only
        with torch.no_grad():
            for inputs in tqdm(train_loader, desc="Collecting normal images", leave=False):
                inputs = self.to_device(inputs)
                images = inputs['image']
                labels = inputs.get('label', torch.zeros(images.shape[0]))  # Assume all training images are normal
                
                # Only use normal images (label == 0)
                normal_mask = labels == 0
                if normal_mask.any():
                    normal_imgs = images[normal_mask]
                    normal_images.extend([img for img in normal_imgs])
        
        if len(normal_images) == 0:
            self.log("Warning: No normal images found for GMM fitting")
            return
        
        self.log(f" > Fitting GMM on {len(normal_images)} normal images...")
        
        # Convert to tensor for batch processing
        normal_images_tensor = torch.stack(normal_images[:min(len(normal_images), 1000)])  # Limit for memory
        
        # Fit GMM
        features = self.model.fit_gmm(normal_images_tensor, n_components=1)
        self._gmm_fitted = True
        
        self.log(f" > GMM fitted successfully on features of shape: {features.shape}")

    def configure_optimizers(self):
        """Configure optimizer for classification training."""
        return optim.AdamW(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
        )

    @property
    def learning_type(self):
        return "one_class"

    @property
    def trainer_arguments(self):
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }

    def get_classification_stats(self):
        """Get classification training statistics."""
        if hasattr(self.model, 'classifier'):
            num_params = sum(p.numel() for p in self.model.classifier.parameters())
            trainable_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
            
            return {
                'classifier_params': num_params,
                'trainable_params': trainable_params,
                'backbone': getattr(self.model.classifier.feature_extractor, 'backbone', 'unknown'),
            }
        return {}

    def get_density_stats(self):
        """Get GMM density model statistics."""
        if not self._gmm_fitted or not hasattr(self.model, 'gmm') or self.model.gmm is None:
            return {'gmm_fitted': False}
        
        gmm = self.model.gmm
        return {
            'gmm_fitted': True,
            'n_components': gmm.n_components,
            'n_features': gmm.means_.shape[1],
            'aic': gmm.aic(gmm.means_),  # Approximate AIC
            'bic': gmm.bic(gmm.means_),  # Approximate BIC
            'converged': gmm.converged_,
        }

    def get_augmentation_stats(self):
        """Get cut-paste augmentation statistics."""
        if hasattr(self.model, 'augmentation'):
            aug = self.model.augmentation
            return {
                'cut_size_ratio': aug.cut_size_ratio,
                'paste_number_range': aug.paste_number_range,
                'paste_prob': aug.paste_prob,
                'rotation_angle_range': aug.rotation_angle_range,
            }
        return {}

    def set_augmentation_params(self, **params):
        """Set cut-paste augmentation parameters."""
        if hasattr(self.model, 'augmentation'):
            aug = self.model.augmentation
            for key, value in params.items():
                if hasattr(aug, key):
                    setattr(aug, key, value)

    def predict_classification(self, inputs):
        """Predict classification scores (for evaluation during training)."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            # Apply augmentation to create test pairs
            images = inputs['image']
            augmented_images, aug_labels = self.model.augmentation(images)
            
            # Create test batch: [original, augmented]
            test_images = torch.cat([images, augmented_images], dim=0)
            logits = self.model.classifier(test_images)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            return predictions

    def extract_features(self, data_loader):
        """Extract features from trained classifier."""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for inputs in tqdm(data_loader, desc="Extracting features", leave=False):
                inputs = self.to_device(inputs)
                features = self.model.classifier.extract_features(inputs['image'])
                labels = inputs.get('label', torch.zeros(features.shape[0]))
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        
        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        
        return features_tensor, labels_tensor

    def get_feature_extractor(self):
        """Get feature extractor for analysis."""
        if hasattr(self.model, 'classifier'):
            return self.model.classifier
        return None

    def is_gmm_fitted(self):
        """Check if GMM is fitted."""
        return self._gmm_fitted

    def reset_gmm(self):
        """Reset GMM fitting state."""
        self._gmm_fitted = False
        if hasattr(self.model, 'gmm'):
            self.model.gmm = None
            self.model.fitted = False

    def evaluate_cut_paste_quality(self, data_loader, num_samples=100):
        """Evaluate cut-paste augmentation quality."""
        self.model.eval()
        
        augmentation_results = []
        with torch.no_grad():
            for i, inputs in enumerate(data_loader):
                if i * inputs['image'].shape[0] >= num_samples:
                    break
                
                inputs = self.to_device(inputs)
                images = inputs['image']
                
                # Apply augmentation
                aug_images, labels = self.model.augmentation(images)
                
                # Calculate augmentation statistics
                for j in range(images.shape[0]):
                    if j >= num_samples:
                        break
                    
                    original = images[j]
                    augmented = aug_images[j]
                    label = labels[j].item()
                    
                    if label == 1:  # Only analyze augmented images
                        # Calculate difference metrics
                        mse = torch.mean((original - augmented) ** 2).item()
                        mae = torch.mean(torch.abs(original - augmented)).item()
                        
                        augmentation_results.append({
                            'mse': mse,
                            'mae': mae,
                            'augmented': True
                        })
        
        if augmentation_results:
            avg_mse = np.mean([r['mse'] for r in augmentation_results])
            avg_mae = np.mean([r['mae'] for r in augmentation_results])
            
            return {
                'num_evaluated': len(augmentation_results),
                'avg_mse': avg_mse,
                'avg_mae': avg_mae,
                'augmentation_rate': len(augmentation_results) / num_samples
            }
        else:
            return {'num_evaluated': 0, 'augmentation_rate': 0.0}

    def log(self, message):
        """Unified logging interface."""
        print(message)