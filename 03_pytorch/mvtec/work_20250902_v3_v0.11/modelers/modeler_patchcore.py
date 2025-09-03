"""PatchCore modeler implementation wrapping the PatchCore model."""

import torch
from torch import optim

from .modeler_base import BaseModeler


class PatchcoreModeler(BaseModeler):
    """PatchCore Modeler for memory-based anomaly detection."""

    def __init__(self, model, loss_fn=None, metrics=None, device=None, coreset_sampling_ratio=0.1):
        super().__init__(model, loss_fn, metrics, device)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self._fitted = False

    def train_step(self, inputs, optimizer):
        """Training step for PatchCore model - collect embeddings."""
        self.model.train()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            embeddings = self.model(inputs['image'])

        # Memory bank progress monitoring
        current_batches = len(self.model.embedding_store)
        total_samples = sum(emb.shape[0] for emb in self.model.embedding_store) if self.model.embedding_store else 0

        # Latest embedding statistics
        if self.model.embedding_store:
            latest_embedding = self.model.embedding_store[-1]
            embedding_mean = latest_embedding.mean().item()
            embedding_std = latest_embedding.std().item()
            embedding_dim = latest_embedding.shape[1]
        else:
            embedding_mean = 0.0
            embedding_std = 0.0
            embedding_dim = 0

        results = {
            'loss': 0.0,  # dummy loss for memory-based training
            'memory_batches': current_batches,
            'total_samples': total_samples,
            'embedding_mean': embedding_mean,
            'embedding_std': embedding_std,
            'embedding_dim': embedding_dim,
        }
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        """Validation step for PatchCore model."""
        if not self._fitted:
            # Cannot validate without fitting first
            return {'loss': 0.0, 'separation': 0.0}

        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])

        if hasattr(predictions, 'pred_score'):
            scores = predictions.pred_score
            labels = inputs['label']

            # Normal vs Anomaly score distribution analysis
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

            # Additional PatchCore-specific metrics
            results['score_range'] = (scores.max() - scores.min()).item()
            results['score_median'] = scores.median().item()
            results['memory_bank_size'] = self.model.memory_bank.shape[0] if self.model.memory_bank.numel() > 0 else 0

            return results

        return {'loss': 0.0}

    @torch.no_grad()
    def predict_step(self, inputs):
        """Prediction step for PatchCore model."""
        # Ensure model is fitted before prediction
        if not self._fitted:
            self.fit()

        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])

        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Training mode output - convert to scores (shouldn't happen if fit() was called)
            return torch.zeros(inputs['image'].shape[0], device=self.device)

    def compute_anomaly_scores(self, inputs):
        """Compute anomaly scores and maps for PatchCore model."""
        # Ensure model is fitted before computing scores
        if not self._fitted:
            self.fit()

        self.model.eval()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            predictions = self.model(inputs['image'])

            if hasattr(predictions, 'anomaly_map'):
                return {
                    'anomaly_maps': predictions.anomaly_map,
                    'pred_scores': predictions.pred_score
                }
            else:
                # Training mode output - shouldn't happen if fitted
                return {
                    'anomaly_maps': torch.zeros(inputs['image'].shape[0], 1, *inputs['image'].shape[-2:], device=self.device),
                    'pred_scores': torch.zeros(inputs['image'].shape[0], device=self.device)
                }

    def fit(self):
        """Fit PatchCore by applying coreset subsampling."""
        if hasattr(self.model, 'subsample_embedding'):
            print(f" > Applying coreset subsampling with ratio {self.coreset_sampling_ratio}...")
            print(f" > Embedding store size: {len(self.model.embedding_store)} batches")

            # Calculate total samples before coreset subsampling
            if self.model.embedding_store:
                total_samples_before = sum(emb.shape[0] for emb in self.model.embedding_store)
                print(f" > Total samples before coreset: {total_samples_before}")

                # Perform coreset subsampling
                self.model.subsample_embedding(self.coreset_sampling_ratio)
                self._fitted = True

                # Post-coreset statistics
                memory_bank_size = self.model.memory_bank.shape[0] if self.model.memory_bank.numel() > 0 else 0
                memory_bank_dim = self.model.memory_bank.shape[1] if self.model.memory_bank.numel() > 0 else 0
                
                print(f" > Memory bank size after coreset: {memory_bank_size}")
                print(f" > Memory bank dimension: {memory_bank_dim}")
                print(f" > Coreset reduction: {total_samples_before} -> {memory_bank_size} "
                      f"({memory_bank_size/total_samples_before*100:.1f}%)")
            else:
                print(f" > Warning: No embeddings collected during training")

    def configure_optimizers(self):
        """PatchCore doesn't require optimization during training."""
        # Return a dummy optimizer for compatibility with trainer
        return optim.AdamW([torch.tensor(0.0, requires_grad=True)], lr=1e-3)

    @property
    def learning_type(self):
        """PatchCore uses one-class learning."""
        return "one_class"

    @property
    def trainer_arguments(self):
        """PatchCore trainer arguments."""
        return {
            "max_epochs": 1,  # PatchCore only needs one pass through training data
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }

    def get_memory_stats(self):
        """Get memory bank statistics."""
        if hasattr(self.model, 'memory_bank') and self.model.memory_bank.numel() > 0:
            return {
                'memory_bank_size': self.model.memory_bank.shape[0],
                'memory_bank_dim': self.model.memory_bank.shape[1],
                'is_fitted': self._fitted,
                'coreset_ratio': self.coreset_sampling_ratio,
                'backbone': self.model.backbone,
                'layers': list(self.model.layers),
                'num_neighbors': self.model.num_neighbors,
            }
        else:
            return {
                'memory_bank_size': 0,
                'memory_bank_dim': 0,
                'is_fitted': self._fitted,
                'coreset_ratio': self.coreset_sampling_ratio,
            }

    def get_nearest_neighbor_stats(self, inputs, k=5):
        """Get statistics about nearest neighbor distances."""
        if not self._fitted:
            return {'error': 'Model not fitted yet'}
        
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            # Get embeddings
            features = self.model.feature_extractor(inputs['image'])
            features = {layer: self.model.feature_pooler(feature) for layer, feature in features.items()}
            embedding = self.model.generate_embedding(features)
            embedding = self.model.reshape_embedding(embedding)
            
            # Get k nearest neighbors
            distances, _ = self.model.nearest_neighbors(embedding, n_neighbors=min(k, self.model.memory_bank.shape[0]))
            
            # Compute statistics
            stats = {
                'mean_distance': distances.mean().item(),
                'std_distance': distances.std().item(),
                'min_distance': distances.min().item(),
                'max_distance': distances.max().item(),
                'median_distance': distances.median().item(),
                'k_neighbors': min(k, self.model.memory_bank.shape[0]),
            }
            
            return stats