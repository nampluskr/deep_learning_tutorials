import torch
from torch import optim

from .modeler_base import BaseModeler


class PadimModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)
        self._fitted = False

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            predictions = self.model(inputs['image'])

        # Memory bank progress monitoring
        current_batches = len(self.model.memory_bank)
        total_samples = sum(emb.shape[0] for emb in self.model.memory_bank) if self.model.memory_bank else 0

        # Latest embedding statistics
        if self.model.memory_bank:
            latest_embedding = self.model.memory_bank[-1]
            embedding_mean = latest_embedding.mean().item()
            embedding_std = latest_embedding.std().item()
        else:
            embedding_mean = 0.0
            embedding_std = 0.0

        results = {
            'loss': 0.0,  # dummy loss for memory-based training
            'memory_batches': current_batches,
            'total_samples': total_samples,
            'embedding_mean': embedding_mean,
            'embedding_std': embedding_std,
        }
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        if not self._fitted:
            self.fit()

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

            return results

        return {'loss': 0.0}

    @torch.no_grad()
    def predict_step(self, inputs):
        # Ensure model is fitted before prediction
        if not self._fitted:
            self.fit()

        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])

        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Training mode output - convert to scores
            embeddings = predictions
            # For PaDiM, we need fitted gaussian to compute scores
            # This shouldn't happen if fit() was called properly
            return torch.zeros(inputs['image'].shape[0], device=self.device)

    def compute_anomaly_scores(self, inputs):
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
                # Training mode output - shouldn't happen
                return {
                    'anomaly_maps': torch.zeros(inputs['image'].shape[0], 1, *inputs['image'].shape[-2:], device=self.device),
                    'pred_scores': torch.zeros(inputs['image'].shape[0], device=self.device)
                }

    def fit(self):
        """Fit Gaussian distribution to collected embeddings."""
        if hasattr(self.model, 'fit'):
            print(f" > Fitting Gaussian distribution...")
            print(f" > Memory bank size: {len(self.model.memory_bank)} batches")

            # Calculate total samples
            total_samples = sum(emb.shape[0] for emb in self.model.memory_bank)
            print(f" > Total samples: {total_samples}")

            # Perform fitting
            self.model.fit()
            self._fitted = True

            # Post-fitting Gaussian parameter statistics
            if hasattr(self.model.gaussian, 'mean') and self.model.gaussian.mean is not None:
                mean_shape = self.model.gaussian.mean.shape
                inv_cov_shape = self.model.gaussian.inv_covariance.shape

                mean_avg = self.model.gaussian.mean.mean().item()
                mean_std = self.model.gaussian.mean.std().item()

                print(f" > Gaussian mean shape: {mean_shape}")
                print(f" > Gaussian inv_cov shape: {inv_cov_shape}")
                print(f" > Mean statistics - avg: {mean_avg:.4f}, std: {mean_std:.4f}")

    def configure_optimizers(self):
        """PaDiM doesn't require optimization during training"""
        # Return a dummy optimizer for compatibility
        return optim.Adam([torch.tensor(0.0, requires_grad=True)], lr=1e-3)

    @property
    def learning_type(self):
        return "one_class"

    @property
    def trainer_arguments(self):
        return {
            "max_epochs": 1,  # PaDiM only needs one pass through training data
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }