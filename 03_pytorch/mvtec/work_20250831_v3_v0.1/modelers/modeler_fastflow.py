"""FastFlow modeler implementation wrapping the FastFlow model."""

import torch
from torch import optim

from .modeler_base import BaseModeler


class FastflowModeler(BaseModeler):
    """FastFlow Modeler for normalizing flow-based anomaly detection."""

    def __init__(self, model, loss_fn, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        """Training step for FastFlow model."""
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        
        # Forward pass through normalizing flows
        hidden_variables, jacobians = self.model(inputs['image'])
        loss = self.loss_fn(hidden_variables, jacobians)
        
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}

        # Compute flow-specific metrics
        with torch.no_grad():
            # Log-likelihood computation for monitoring
            log_likelihood = 0.0
            for hidden_variable, jacobian in zip(hidden_variables, jacobians):
                log_likelihood -= torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
            
            results['log_likelihood'] = log_likelihood.item()

            # Jacobian statistics for monitoring flow stability
            total_jacobian = torch.stack(jacobians).mean()
            results['jacobian_mean'] = total_jacobian.item()

            # Hidden variable statistics
            hidden_var_mean = torch.mean(torch.stack([torch.mean(hv) for hv in hidden_variables]))
            hidden_var_std = torch.mean(torch.stack([torch.std(hv) for hv in hidden_variables]))
            results['hidden_var_mean'] = hidden_var_mean.item()
            results['hidden_var_std'] = hidden_var_std.item()

        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        """Validation step for FastFlow model."""
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])

        # For validation, we get InferenceBatch(pred_score, anomaly_map)
        if hasattr(predictions, 'pred_score'):
            scores = predictions.pred_score
            labels = inputs['label']

            # Compute score distribution analysis
            normal_mask = labels == 0
            anomaly_mask = labels == 1

            normal_scores = scores[normal_mask] if normal_mask.any() else torch.tensor([0.0])
            anomaly_scores = scores[anomaly_mask] if anomaly_mask.any() else torch.tensor([0.0])

            results = {
                'loss': 0.0,  # No loss computation in validation for flow models
                'score_mean': scores.mean().item(),
                'score_std': scores.std().item(),
                'normal_mean': normal_scores.mean().item(),
                'anomaly_mean': anomaly_scores.mean().item(),
                'separation': (anomaly_scores.mean() - normal_scores.mean()).item() if anomaly_mask.any() and normal_mask.any() else 0.0,
            }

            # Additional flow-specific validation metrics
            results['score_range'] = (scores.max() - scores.min()).item()
            results['score_median'] = scores.median().item()

            return results
        else:
            # Training mode output - shouldn't happen in validation
            return {'loss': 0.0}

    @torch.no_grad()
    def predict_step(self, inputs):
        """Prediction step for FastFlow model."""
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])

        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback: if somehow training mode output
            hidden_variables, jacobians = predictions
            # Compute negative log-likelihood as anomaly score
            scores = torch.zeros(inputs['image'].shape[0], device=self.device)
            for hidden_variable, jacobian in zip(hidden_variables, jacobians):
                scores += 0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian
            return scores

    def compute_anomaly_scores(self, inputs):
        """Compute anomaly scores and maps for FastFlow model."""
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
                # Fallback: compute from hidden variables
                hidden_variables, jacobians = predictions
                
                # Use the anomaly map generator from the model
                if hasattr(self.model, 'anomaly_map_generator'):
                    anomaly_maps = self.model.anomaly_map_generator(hidden_variables)
                    pred_scores = torch.amax(anomaly_maps, dim=(-2, -1))
                else:
                    # Simple fallback
                    batch_size = inputs['image'].shape[0]
                    image_size = inputs['image'].shape[-2:]
                    anomaly_maps = torch.zeros(batch_size, 1, *image_size, device=self.device)
                    pred_scores = torch.zeros(batch_size, device=self.device)

                return {
                    'anomaly_maps': anomaly_maps,
                    'pred_scores': pred_scores
                }

    def configure_optimizers(self):
        """Configure FastFlow optimizers."""
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    @property
    def learning_type(self):
        """FastFlow uses one-class learning."""
        return "one_class"

    @property
    def trainer_arguments(self):
        """FastFlow trainer arguments."""
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }

    def get_flow_stats(self):
        """Get flow-specific statistics if model has been used."""
        if hasattr(self.model, 'fast_flow_blocks'):
            num_blocks = len(self.model.fast_flow_blocks)
            total_params = sum(p.numel() for block in self.model.fast_flow_blocks for p in block.parameters())
            return {
                'num_flow_blocks': num_blocks,
                'flow_params': total_params,
                'backbone': getattr(self.model, 'backbone', 'unknown') if hasattr(self.model, 'backbone') else 'unknown'
            }
        return {'num_flow_blocks': 0, 'flow_params': 0, 'backbone': 'unknown'}