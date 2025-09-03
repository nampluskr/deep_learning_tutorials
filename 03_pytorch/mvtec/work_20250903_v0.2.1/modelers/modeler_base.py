import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class MemoryModelMixin:
    """Mixin for memory-based models lifecycle management."""
    
    def __init__(self):
        self._fitted = False
        
    def _ensure_fitted(self):
        """Ensure the model is fitted (lazy fitting)."""
        if not self._fitted and hasattr(self, 'fit'):
            self.fit()
            self._fitted = True
    
    def is_fitted(self):
        """Check if the model is fitted."""
        return self._fitted


class BaseModeler(ABC):
    """Base class for all modelers with Template Method pattern."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None, **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move components to device
        self.model = self.model.to(self.device)
        if self.loss_fn:
            self.loss_fn = self.loss_fn.to(self.device)
        
        for metric_name, metric_fn in self.metrics.items():
            if hasattr(metric_fn, 'to'):
                self.metrics[metric_name] = metric_fn.to(self.device)

    def to_device(self, inputs):
        """Move inputs to the appropriate device."""
        device_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                device_inputs[key] = value.to(self.device, non_blocking=True)
            else:
                device_inputs[key] = value
        return device_inputs

    def get_metric_names(self):
        """Get list of metric names."""
        return list(self.metrics.keys())

    # ========================================================================
    # Template Methods - Provide common workflow, call hooks for customization
    # ========================================================================

    def train_step(self, inputs, optimizer):
        """Template method for training step with backpropagation."""
        self._prepare_training()
        inputs = self.to_device(inputs)
        optimizer.zero_grad()
        
        # Hook: Compute loss (model-specific logic)
        loss = self._compute_loss(inputs)
        
        loss.backward()
        optimizer.step()
        
        # Hook: Collect results and metrics
        results = self._collect_training_results(inputs, loss)
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        """Template method for validation step (no backprop, for early stopping)."""
        self._prepare_validation()
        inputs = self.to_device(inputs)
        
        # Hook: Compute loss (model-specific logic)
        loss = self._compute_loss(inputs)
        
        # Hook: Collect results and metrics
        results = self._collect_validation_results(inputs, loss)
        return results

    @torch.no_grad()
    def evaluate_step(self, inputs):
        """Template method for evaluation step (anomaly detection evaluation)."""
        self._prepare_evaluation()
        inputs = self.to_device(inputs)
        
        # Hook: Compute predictions (model-specific logic)
        predictions = self._compute_predictions(inputs)
        
        # Hook: Collect evaluation results
        results = self._collect_evaluation_results(inputs, predictions)
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        """Template method for prediction step (inference only)."""
        self._prepare_prediction()
        inputs = self.to_device(inputs)
        
        # Hook: Compute predictions and return scores only
        return self._compute_prediction_scores(inputs)

    def compute_anomaly_scores(self, inputs):
        """Compute comprehensive anomaly scores and maps."""
        self._prepare_evaluation()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            # Hook: Get detailed anomaly information
            return self._compute_detailed_anomaly_scores(inputs)

    # ========================================================================
    # Hook Methods - Abstract methods for model-specific customization
    # ========================================================================

    def _prepare_training(self):
        """Hook: Prepare model for training (default: set train mode)."""
        self.model.train()

    def _prepare_validation(self):
        """Hook: Prepare model for validation (default: set train mode for consistent loss)."""
        self.model.train()  # Keep in training mode for consistent loss computation

    def _prepare_evaluation(self):
        """Hook: Prepare model for evaluation (default: set eval mode)."""
        self.model.eval()

    def _prepare_prediction(self):
        """Hook: Prepare model for prediction (default: set eval mode)."""
        self.model.eval()

    @abstractmethod
    def _compute_loss(self, inputs):
        """Hook: Compute loss for training/validation (model-specific)."""
        pass

    @abstractmethod
    def _compute_predictions(self, inputs):
        """Hook: Compute predictions for evaluation (model-specific)."""
        pass

    @abstractmethod
    def _compute_prediction_scores(self, inputs):
        """Hook: Compute prediction scores for inference (model-specific)."""
        pass

    def _compute_detailed_anomaly_scores(self, inputs):
        """Hook: Compute detailed anomaly scores and maps (default implementation)."""
        predictions = self._compute_predictions(inputs)
        
        if hasattr(predictions, 'anomaly_map') and hasattr(predictions, 'pred_score'):
            return {
                'anomaly_maps': predictions.anomaly_map,
                'pred_scores': predictions.pred_score
            }
        else:
            # Fallback for models that don't return InferenceBatch
            scores = self._compute_prediction_scores(inputs)
            batch_size = scores.shape[0]
            image_size = inputs['image'].shape[-2:]
            
            return {
                'anomaly_maps': torch.zeros(batch_size, 1, *image_size, device=self.device),
                'pred_scores': scores
            }

    def _collect_training_results(self, inputs, loss):
        """Hook: Collect training results and metrics (default implementation)."""
        results = {'loss': loss.item()}
        
        # Compute metrics if available
        if hasattr(self, '_compute_training_metrics'):
            metrics = self._compute_training_metrics(inputs, loss)
            results.update(metrics)
        
        return results

    def _collect_validation_results(self, inputs, loss):
        """Hook: Collect validation results and metrics (default implementation)."""
        results = {'loss': loss.item()}
        
        # Compute metrics if available
        if hasattr(self, '_compute_validation_metrics'):
            metrics = self._compute_validation_metrics(inputs, loss)
            results.update(metrics)
        
        return results

    def _collect_evaluation_results(self, inputs, predictions):
        """Hook: Collect evaluation results (default implementation)."""
        # Handle InferenceBatch format
        if hasattr(predictions, 'pred_score'):
            scores = predictions.pred_score
            labels = inputs['label']
            
            # Score distribution analysis
            normal_mask = labels == 0
            anomaly_mask = labels == 1

            normal_scores = scores[normal_mask] if normal_mask.any() else torch.tensor([0.0])
            anomaly_scores = scores[anomaly_mask] if anomaly_mask.any() else torch.tensor([0.0])

            results = {
                'pred_scores': scores,
                'anomaly_maps': predictions.anomaly_map if hasattr(predictions, 'anomaly_map') else None,
                'score_mean': scores.mean().item(),
                'score_std': scores.std().item(),
                'normal_mean': normal_scores.mean().item(),
                'anomaly_mean': anomaly_scores.mean().item(),
                'separation': (anomaly_scores.mean() - normal_scores.mean()).item() if anomaly_mask.any() and normal_mask.any() else 0.0,
            }
        else:
            # Fallback for non-InferenceBatch predictions
            scores = self._compute_prediction_scores(inputs)
            results = {
                'pred_scores': scores,
                'anomaly_maps': None,
                'score_mean': scores.mean().item(),
                'score_std': scores.std().item(),
                'separation': 0.0,
            }

        return results

    def compute_image_scores(self, anomaly_maps):
        """Compute image-level scores from anomaly maps."""
        if anomaly_maps is None:
            return torch.zeros(1, device=self.device)
            
        batch_size = anomaly_maps.shape[0]
        flattened = anomaly_maps.view(batch_size, -1)
        image_scores = torch.max(flattened, dim=1)[0]
        return image_scores

    # ========================================================================
    # Properties - Abstract properties for model characteristics
    # ========================================================================

    @property
    @abstractmethod
    def learning_type(self):
        """Get the learning type of the model (e.g., 'one_class', 'supervised')."""
        pass

    @property
    def trainer_arguments(self):
        """Get trainer-specific arguments (default: empty dict)."""
        return {}

    # ========================================================================
    # Model persistence
    # ========================================================================

    def save_model(self, path):
        """Save model state."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def get_model(self):
        """Get the underlying model."""
        return self.model


class MemoryBasedModeler(BaseModeler, MemoryModelMixin):
    """Base class for memory-based modelers (PaDiM, PatchCore)."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None, **kwargs):
        BaseModeler.__init__(self, model, loss_fn, metrics, device, **kwargs)
        MemoryModelMixin.__init__(self)

    def validate_step(self, inputs):
        """Override for memory models - ensure fitting before validation."""
        self._ensure_fitted()
        return super().validate_step(inputs)

    def evaluate_step(self, inputs):
        """Override for memory models - ensure fitting before evaluation."""
        self._ensure_fitted()
        return super().evaluate_step(inputs)

    def predict_step(self, inputs):
        """Override for memory models - ensure fitting before prediction."""
        self._ensure_fitted()
        return super().predict_step(inputs)

    def compute_anomaly_scores(self, inputs):
        """Override for memory models - ensure fitting before computing scores."""
        self._ensure_fitted()
        return super().compute_anomaly_scores(inputs)


if __name__ == "__main__":
    pass