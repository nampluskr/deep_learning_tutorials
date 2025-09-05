from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim


# ===================================================================
# Base Modeler
# ===================================================================

class BaseModeler(ABC):
    """Base modeler for anomaly detection models with unified PyTorch interface."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        """Initialize modeler with model and training components."""
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
        """Move inputs to appropriate device."""
        device_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                device_inputs[key] = value.to(self.device, non_blocking=True)
            else:
                device_inputs[key] = value
        return device_inputs

    def get_metric_names(self):
        """Return list of available metric names."""
        return list(self.metrics.keys())

    @abstractmethod
    def train_step(self, inputs, optimizer):
        """Execute training step with backpropagation."""
        pass

    @abstractmethod  
    def validation_step(self, inputs):
        """Execute validation step without backpropagation."""
        pass



    @abstractmethod
    def predict_step(self, inputs):
        """Execute deployment-optimized prediction step."""
        pass
        
    @abstractmethod
    def test_step(self, inputs):
        """Execute evaluation-focused test step."""
        pass

    def save_model(self, path):
        """Save model state to disk."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state from disk."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def get_model(self):
        """Return the underlying model."""
        return self.model


# ===================================================================
# Autoencoder Modeler
# ===================================================================

class AEModeler(BaseModeler):
    """Modeler for autoencoder-based anomaly detection models."""
    
    def __init__(self, model, loss_fn, metrics=None, device=None):
        """Initialize AE modeler with reconstruction components."""
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        """Training step for autoencoder models."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        predictions = self.model(inputs['image'])
        
        # Training mode: (reconstructed, latent, features)
        reconstructed, latent, features = predictions
        loss = self.loss_fn(reconstructed, inputs['image'])
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(reconstructed, inputs['image'])
                results[metric_name] = float(metric_value)
        return results

    def validation_step(self, inputs):
        """Validation step for autoencoder models."""
        self.model.train()  # Keep training mode for validation
        inputs = self.to_device(inputs)

        with torch.no_grad():
            # Training mode: (reconstructed, latent, features)
            reconstructed, latent, features = self.model(inputs['image'])
            loss = self.loss_fn(reconstructed, inputs['image'])

            results = {'loss': loss.item()}
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(reconstructed, inputs['image'])
                results[metric_name] = float(metric_value)
        return results

    def predict_step(self, inputs):
        """Deployment-optimized prediction step for autoencoders."""
        self.model.eval()  # Use inference mode
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            predictions = self.model(inputs['image'])
            
            if isinstance(predictions, dict) and 'pred_score' in predictions:
                # Model returns inference outputs directly
                return {
                    'pred_scores': predictions['pred_score'],
                    'anomaly_maps': predictions.get('anomaly_map', None)
                }
            else:
                # Legacy: compute from reconstruction outputs
                reconstructed, _, _ = predictions
                anomaly_maps = torch.mean((inputs['image'] - reconstructed)**2, dim=1, keepdim=True)
                pred_scores = torch.amax(anomaly_maps, dim=(-2, -1))
                return {
                    'pred_scores': pred_scores,
                    'anomaly_maps': anomaly_maps
                }

    def test_step(self, inputs):
        """Evaluation-focused test step for autoencoders."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            predictions = self.model(inputs['image'])
            
            if isinstance(predictions, dict) and 'pred_score' in predictions:
                # Model returns inference outputs directly
                results = {
                    'pred_scores': predictions['pred_score'],
                    'anomaly_maps': predictions.get('anomaly_map', None),
                    'gt_labels': inputs['label']
                }
            else:
                # Legacy: compute from reconstruction outputs
                reconstructed, latent, features = predictions
                anomaly_maps = torch.mean((inputs['image'] - reconstructed)**2, dim=1, keepdim=True)
                pred_scores = torch.amax(anomaly_maps, dim=(-2, -1))
                results = {
                    'pred_scores': pred_scores,
                    'anomaly_maps': anomaly_maps,
                    'gt_labels': inputs['label']
                }
                
        return results


# ===================================================================
# STFPM Modeler
# ===================================================================

class STFPMModeler(BaseModeler):
    """Modeler for STFPM (Student-Teacher Feature Pyramid Matching) model."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        """Initialize STFPM modeler with distillation components."""
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        """Training step for STFPM model."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        teacher_features, student_features = self.model(inputs['image'])
        loss = self.loss_fn(teacher_features, student_features)
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                if metric_name == "feature_sim":
                    similarities = []
                    for layer in teacher_features:
                        layer_sim = metric_fn(teacher_features[layer], student_features[layer])
                        similarities.append(layer_sim)
                    results[metric_name] = sum(similarities) / len(similarities) if similarities else 0.0
                else:
                    results[metric_name] = 0.0

        return results

    def validation_step(self, inputs):
        """Validation step for STFPM model."""
        self.model.train()  # Keep training mode for validation
        inputs = self.to_device(inputs)

        with torch.no_grad():
            teacher_features, student_features = self.model(inputs['image'])
            loss = self.loss_fn(teacher_features, student_features)

            results = {'loss': loss.item()}
            for metric_name, metric_fn in self.metrics.items():
                if metric_name == "feature_sim":
                    similarities = []
                    for layer in teacher_features:
                        layer_sim = metric_fn(teacher_features[layer], student_features[layer])
                        similarities.append(layer_sim)
                    results[metric_name] = sum(similarities) / len(similarities) if similarities else 0.0
                else:
                    results[metric_name] = 0.0

        return results

    def predict_step(self, inputs):
        """Deployment-optimized prediction step for STFPM."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            model_outputs = self.model(inputs['image'])
            return self._compute_anomaly_outputs(inputs, model_outputs)
    
    def _compute_anomaly_outputs(self, inputs, model_outputs):
        """Compute anomaly outputs with fallback strategies."""
        # Strategy 1: Direct model inference outputs
        if isinstance(model_outputs, dict) and 'pred_score' in model_outputs:
            return {
                'pred_scores': model_outputs['pred_score'],
                'anomaly_maps': model_outputs.get('anomaly_map', None)
            }
        
        # Strategy 2: Use model's compute_anomaly_map method
        if hasattr(self.model, 'compute_anomaly_map'):
            try:
                anomaly_map = self._call_compute_anomaly_map(inputs, model_outputs)
                pred_scores = torch.amax(anomaly_map, dim=(-2, -1))
                return {
                    'pred_scores': pred_scores,
                    'anomaly_maps': anomaly_map
                }
            except Exception:
                pass
        
        # Strategy 3: Fallback computation
        return self._fallback_anomaly_computation(inputs, model_outputs)
    
    def _call_compute_anomaly_map(self, inputs, model_outputs):
        """Call model's compute_anomaly_map for STFPM model."""
        if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
            teacher_features, student_features = model_outputs
            return self.model.compute_anomaly_map(
                teacher_features, student_features, inputs['image'].shape[-2:]
            )
        else:
            raise ValueError("Unexpected STFPM model outputs for compute_anomaly_map")
    
    def _fallback_anomaly_computation(self, inputs, model_outputs):
        """Fallback computation for STFPM model."""
        if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
            teacher_features, student_features = model_outputs
            
            # Fallback 1: Use AnomalyMapGenerator if available
            if hasattr(self.model, 'anomaly_map_generator'):
                anomaly_map = self.model.anomaly_map_generator(
                    teacher_features=teacher_features,
                    student_features=student_features,
                    image_size=inputs['image'].shape[-2:],
                )
                pred_scores = torch.amax(anomaly_map, dim=(-2, -1))
                return {
                    'pred_scores': pred_scores,
                    'anomaly_maps': anomaly_map
                }
            else:
                # Fallback 2: Simple zero output
                batch_size = next(iter(teacher_features.values())).shape[0]
                image_size = inputs['image'].shape[-2:]
                anomaly_maps = torch.zeros(batch_size, 1, *image_size, device=self.device)
                pred_scores = torch.zeros(batch_size, device=self.device)
                return {
                    'pred_scores': pred_scores,
                    'anomaly_maps': anomaly_maps
                }
        else:
            raise ValueError("Cannot compute fallback anomaly for unexpected model outputs")

    def test_step(self, inputs):
        """Evaluation-focused test step for STFPM."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            model_outputs = self.model(inputs['image'])
            results = self._compute_anomaly_outputs(inputs, model_outputs)
            results['gt_labels'] = inputs['label']
            return results


if __name__ == "__main__":
    pass