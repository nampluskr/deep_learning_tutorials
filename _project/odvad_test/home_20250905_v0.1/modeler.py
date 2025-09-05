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
# VAE Modeler
# ===================================================================

class VAEModeler(BaseModeler):
    """Modeler for variational autoencoder-based anomaly detection models."""
    
    def __init__(self, model, loss_fn, metrics=None, device=None):
        """Initialize VAE modeler with reconstruction and KL divergence components."""
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        """Training step for VAE models."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        predictions = self.model(inputs['image'])
        
        # VAE training mode: (reconstructed, mu, log_var, features)
        reconstructed, mu, log_var, features = predictions
        
        # VAE loss expects (reconstructed, original, mu, log_var)
        loss_dict = self.loss_fn(reconstructed, inputs['image'], mu, log_var)
        if isinstance(loss_dict, dict):
            loss = loss_dict['total_loss']
        else:
            loss = loss_dict
        
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        
        # Add VAE-specific loss components if available
        if isinstance(loss_dict, dict):
            if 'recon_loss' in loss_dict:
                results['recon_loss'] = loss_dict['recon_loss'].item()
            if 'kl_loss' in loss_dict:
                results['kl_loss'] = loss_dict['kl_loss'].item()
        
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(reconstructed, inputs['image'])
                results[metric_name] = float(metric_value)
        return results

    def validation_step(self, inputs):
        """Validation step for VAE models."""
        self.model.train()  # Keep training mode for validation
        inputs = self.to_device(inputs)

        with torch.no_grad():
            # VAE training mode: (reconstructed, mu, log_var, features)
            reconstructed, mu, log_var, features = self.model(inputs['image'])
            
            # VAE loss calculation
            loss_dict = self.loss_fn(reconstructed, inputs['image'], mu, log_var)
            if isinstance(loss_dict, dict):
                loss = loss_dict['total_loss']
            else:
                loss = loss_dict

            results = {'loss': loss.item()}
            
            # Add VAE-specific loss components if available
            if isinstance(loss_dict, dict):
                if 'recon_loss' in loss_dict:
                    results['recon_loss'] = loss_dict['recon_loss'].item()
                if 'kl_loss' in loss_dict:
                    results['kl_loss'] = loss_dict['kl_loss'].item()
            
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(reconstructed, inputs['image'])
                results[metric_name] = float(metric_value)
        return results

    def predict_step(self, inputs):
        """Deployment-optimized prediction step for VAE."""
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
                # In eval mode, VAE uses mu (mean) for deterministic output
                # Should return {'pred_score': ..., 'anomaly_map': ...}
                # If model doesn't return dict, compute anomaly map manually
                if hasattr(self.model, 'compute_anomaly_map'):
                    # Use model's method if available
                    reconstructed = predictions.get('reconstructed', predictions[0] if isinstance(predictions, tuple) else None)
                    if reconstructed is not None:
                        anomaly_maps = self.model.compute_anomaly_map(inputs['image'], reconstructed)
                        pred_scores = self.model.compute_anomaly_score(anomaly_maps)
                    else:
                        # Fallback computation
                        anomaly_maps = torch.zeros_like(inputs['image'][:, :1])
                        pred_scores = torch.zeros(inputs['image'].shape[0], device=self.device)
                else:
                    # Manual computation as fallback
                    anomaly_maps = torch.zeros_like(inputs['image'][:, :1])
                    pred_scores = torch.zeros(inputs['image'].shape[0], device=self.device)
                
                return {
                    'pred_scores': pred_scores,
                    'anomaly_maps': anomaly_maps
                }

    def test_step(self, inputs):
        """Evaluation-focused test step for VAE."""
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
                # Compute anomaly outputs from model predictions
                if hasattr(self.model, 'compute_anomaly_map'):
                    reconstructed = predictions.get('reconstructed', predictions[0] if isinstance(predictions, tuple) else None)
                    if reconstructed is not None:
                        anomaly_maps = self.model.compute_anomaly_map(inputs['image'], reconstructed)
                        pred_scores = self.model.compute_anomaly_score(anomaly_maps)
                    else:
                        anomaly_maps = torch.zeros_like(inputs['image'][:, :1])
                        pred_scores = torch.zeros(inputs['image'].shape[0], device=self.device)
                else:
                    anomaly_maps = torch.zeros_like(inputs['image'][:, :1])
                    pred_scores = torch.zeros(inputs['image'].shape[0], device=self.device)
                
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


# ===================================================================
# FastFlow Modeler
# ===================================================================

class FastFlowModeler(BaseModeler):
    """Modeler for FastFlow normalizing flow-based anomaly detection."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        """Execute training step with negative log-likelihood optimization."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        
        # Flow training: (hidden_variables, jacobians)
        hidden_variables, jacobians = self.model(inputs['image'])
        loss = self.loss_fn(hidden_variables, jacobians)
        
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        
        # Add flow-specific metrics if available
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                if metric_name == "likelihood":
                    total_likelihood = 0
                    for hidden_var, jacobian in zip(hidden_variables, jacobians):
                        # FastFlowLoss와 일치하는 계산
                        neg_log_likelihood = 0.5 * torch.sum(hidden_var**2, dim=(1, 2, 3)) - jacobian
                        likelihood = -torch.mean(neg_log_likelihood)  # 음의 부호로 likelihood 변환
                        total_likelihood += likelihood
                    results[metric_name] = (total_likelihood / len(hidden_variables)).item()

        return results

    def validation_step(self, inputs):
        """Validation step for flow models."""
        self.model.train()  # Keep training mode for validation
        inputs = self.to_device(inputs)

        with torch.no_grad():
            hidden_variables, jacobians = self.model(inputs['image'])
            loss = self.loss_fn(hidden_variables, jacobians)

            results = {'loss': loss.item()}
            
        for metric_name, metric_fn in self.metrics.items():
            if metric_name == "likelihood":
                total_likelihood = 0
                for hidden_var, jacobian in zip(hidden_variables, jacobians):
                    # FastFlowLoss와 일치하는 계산
                    neg_log_likelihood = 0.5 * torch.sum(hidden_var**2, dim=(1, 2, 3)) - jacobian
                    likelihood = -torch.mean(neg_log_likelihood)  # 음의 부호로 likelihood 변환
                    total_likelihood += likelihood
                results[metric_name] = (total_likelihood / len(hidden_variables)).item()

        return results

    def predict_step(self, inputs):
        """Deployment-optimized prediction step for FastFlow."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            # Inference mode: {'pred_score': ..., 'anomaly_map': ...}
            predictions = self.model(inputs['image'])
            return {
                'pred_scores': predictions['pred_score'],
                'anomaly_maps': predictions['anomaly_map']
            }

    def test_step(self, inputs):
        """Evaluation-focused test step for FastFlow."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            predictions = self.model(inputs['image'])
            return {
                'pred_scores': predictions['pred_score'],
                'anomaly_maps': predictions['anomaly_map'],
                'gt_labels': inputs['label']
            }


# ===================================================================
# DRAEM Modeler
# ===================================================================


# modeler.py에서 DRAEMModeler 수정

class DRAEMModeler(BaseModeler):
    """Modeler for DRAEM anomaly detection with synthetic anomaly training."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)
        # Note: PerlinAnomalyGenerator is built into the model

    def train_step(self, inputs, optimizer):
        """Training with synthetic anomaly generation."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        
        # DRAEM training: model handles synthetic anomaly generation internally
        model_outputs = self.model(inputs['image'])
        
        # Extract training outputs
        input_image = model_outputs['input_image']
        reconstruction = model_outputs['reconstruction']
        anomaly_mask = model_outputs['anomaly_mask']
        prediction = model_outputs['prediction']
        
        # DRAEM loss calculation
        loss = self.loss_fn(input_image, reconstruction, anomaly_mask, prediction)
        
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        
        # DRAEM 학습 중 메트릭 계산 (synthetic anomaly 사용)
        with torch.no_grad():
            # 생성된 synthetic anomaly mask를 ground truth로 사용
            gt_labels = anomaly_mask.flatten()  # [B*H*W]
            pred_scores = torch.softmax(prediction, dim=1)[:, 1, :, :].flatten()  # [B*H*W]
            
            # 픽셀 레벨에서 AUROC/AUPR 계산
            for metric_name, metric_fn in self.metrics.items():
                if metric_name in ['auroc', 'aupr']:
                    try:
                        # 정상과 이상 픽셀이 모두 있는 경우만 계산
                        if len(torch.unique(gt_labels)) > 1:
                            metric_value = metric_fn(gt_labels, pred_scores)
                            results[metric_name] = float(metric_value)
                        else:
                            results[metric_name] = 0.0
                    except:
                        results[metric_name] = 0.0
                else:
                    results[metric_name] = 0.0

        return results

    def validation_step(self, inputs):
        """Validation step for DRAEM."""
        self.model.train()  # Keep training mode for validation (synthetic anomaly generation)
        inputs = self.to_device(inputs)

        with torch.no_grad():
            # Generate synthetic anomalies for validation
            model_outputs = self.model(inputs['image'])
            
            # Extract training outputs
            input_image = model_outputs['input_image']
            reconstruction = model_outputs['reconstruction']
            anomaly_mask = model_outputs['anomaly_mask']
            prediction = model_outputs['prediction']
            
            loss = self.loss_fn(input_image, reconstruction, anomaly_mask, prediction)

            results = {'loss': loss.item()}
            
            # Validation 중 메트릭 계산 (synthetic anomaly 사용)
            gt_labels = anomaly_mask.flatten()
            pred_scores = torch.softmax(prediction, dim=1)[:, 1, :, :].flatten()
            
            for metric_name, metric_fn in self.metrics.items():
                if metric_name in ['auroc', 'aupr']:
                    try:
                        if len(torch.unique(gt_labels)) > 1:
                            metric_value = metric_fn(gt_labels, pred_scores)
                            results[metric_name] = float(metric_value)
                        else:
                            results[metric_name] = 0.0
                    except:
                        results[metric_name] = 0.0
                else:
                    results[metric_name] = 0.0

        return results

    def predict_step(self, inputs):
        """Deployment-optimized prediction step for DRAEM."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            predictions = self.model(inputs['image'])
            return {
                'pred_scores': predictions['pred_score'],
                'anomaly_maps': predictions['anomaly_map']
            }

    def test_step(self, inputs):
        """Evaluation-focused test step for DRAEM."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            predictions = self.model(inputs['image'])
            return {
                'pred_scores': predictions['pred_score'],
                'anomaly_maps': predictions['anomaly_map'],
                'gt_labels': inputs['label']
            }
if __name__ == "__main__":
    pass