import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModeler(ABC):
    """Base class for all anomaly detection modelers with unified interface."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        """Initialize modeler with model, loss function, and metrics."""
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
        """Get list of metric names for this modeler."""
        return list(self.metrics.keys())

    @abstractmethod
    def forward(self, inputs):
        """Core forward pass - model specific implementation."""
        pass

    def training_step(self, inputs, optimizer):
        """Training step with backpropagation."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        
        # Use abstract forward method
        outputs = self.forward(inputs)
        
        # Compute loss using model-specific outputs
        loss = self.compute_loss(outputs, inputs)
        loss.backward()
        optimizer.step()

        # Compute training metrics
        train_metrics = {}
        with torch.no_grad():
            train_metrics = self.compute_train_metrics(outputs, inputs)

        results = {'loss': loss.item()}
        results.update(train_metrics)
        return results

    def validation_step(self, inputs):
        """Validation step: same as training_step but without backpropagation."""
        self.model.train()  # Keep in train mode for consistent computation
        inputs = self.to_device(inputs)

        with torch.no_grad():
            # Same forward computation as training
            outputs = self.forward(inputs)
            
            # Same loss computation as training
            val_loss = self.compute_loss(outputs, inputs)
            
            # Same metrics computation as training
            val_metrics = self.compute_train_metrics(outputs, inputs)

        # Return results with val_ prefix for consistency
        results = {'val_loss': val_loss.item()}
        for metric_name, metric_value in val_metrics.items():
            results[f'val_{metric_name}'] = metric_value
        
        return results

    def compute_anomaly_maps(self, inputs):
        """Compute pixel-level anomaly heatmaps."""
        self.model.eval()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            outputs = self.forward(inputs)
            anomaly_maps = self.generate_anomaly_maps(outputs, inputs)
        
        return anomaly_maps

    def compute_anomaly_scores(self, inputs):
        """Compute image-level anomaly scores."""
        anomaly_maps = self.compute_anomaly_maps(inputs)
        
        # Default: max pooling over spatial dimensions
        batch_size = anomaly_maps.shape[0]
        flattened = anomaly_maps.view(batch_size, -1)
        anomaly_scores = torch.max(flattened, dim=1)[0]
        
        return anomaly_scores

    @abstractmethod
    def compute_loss(self, outputs, inputs):
        """Compute loss from model outputs and inputs."""
        pass

    def compute_train_metrics(self, outputs, inputs):
        """Compute training metrics from model outputs."""
        train_metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            try:
                metric_value = self.evaluate_metric(metric_fn, metric_name, outputs, inputs)
                train_metrics[metric_name] = float(metric_value)
            except Exception:
                train_metrics[metric_name] = 0.0
        return train_metrics

    @abstractmethod
    def generate_anomaly_maps(self, outputs, inputs):
        """Generate pixel-level anomaly maps from model outputs."""
        pass

    def evaluate_metric(self, metric_fn, metric_name, outputs, inputs):
        """Evaluate a specific metric based on its type."""
        # Default implementation - subclasses can override
        return 0.0

    def configure_optimizers(self):
        """Configure optimizer for training."""
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state dict."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    @property
    @abstractmethod
    def learning_type(self):
        """Return learning type (one_class, supervised, etc.)."""
        pass

    @property
    def trainer_arguments(self):
        """Return additional arguments for trainer configuration."""
        return {}


if __name__ == "__main__":
    pass