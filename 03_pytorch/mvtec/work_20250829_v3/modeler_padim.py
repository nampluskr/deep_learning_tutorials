import torch
from torch import optim

from modeler_base import BaseModeler


class PadimModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)
        self._fitted = False

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        # PaDiM doesn't require gradient computation during training
        with torch.no_grad():
            predictions = self.model(inputs['image'])
        
        # Return dummy loss for compatibility
        dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        results = {'loss': 0.0}
        # No metrics computed during training for PaDiM
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        # Ensure model is fitted before validation
        if not self._fitted:
            self.fit()
            
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])
        
        # PaDiM validation uses inference mode (returns InferenceBatch)
        if hasattr(predictions, 'pred_score'):
            # No loss computation during validation for PaDiM
            results = {'loss': 0.0}
            
            # Compute metrics if available (using anomaly map and original image)
            if self.metrics:
                # For image-level metrics, we can use pred_scores
                pred_scores = predictions.pred_score
                for metric_name, metric_fn in self.metrics.items():
                    if 'map' in metric_name.lower():
                        # For anomaly map metrics
                        metric_value = metric_fn(predictions.anomaly_map, inputs['image'])
                    else:
                        # For score-based metrics, need ground truth scores
                        # Skip metrics that require ground truth during validation
                        continue
                    results[metric_name] = float(metric_value)
            
            return results
        else:
            # Training mode output - shouldn't happen in validation
            results = {'loss': 0.0}
            return results

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
            self.model.fit()
            self._fitted = True

    def configure_optimizers(self):
        # PaDiM doesn't require optimization during training
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