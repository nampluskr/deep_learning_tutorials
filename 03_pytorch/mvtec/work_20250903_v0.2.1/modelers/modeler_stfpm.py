import torch
from .modeler_base import BaseModeler


class STFPMModeler(BaseModeler):
    """STFPM Modeler for teacher-student feature pyramid matching using Template Method pattern."""
    
    def __init__(self, model, loss_fn=None, metrics=None, device=None, **kwargs):
        super().__init__(model, loss_fn, metrics, device, **kwargs)

    # ========================================================================
    # Hook Methods Implementation - Required by BaseModeler Template Methods
    # ========================================================================

    def _compute_loss(self, inputs):
        """Hook: Compute STFPM loss from teacher-student features."""
        predictions = self.model(inputs['image'])
        teacher_features, student_features = predictions
        return self.loss_fn(teacher_features, student_features)

    def _compute_predictions(self, inputs):
        """Hook: Compute predictions for evaluation mode."""
        return self.model(inputs['image'])

    def _compute_prediction_scores(self, inputs):
        """Hook: Compute prediction scores for inference."""
        predictions = self.model(inputs['image'])
        
        # Return scores from InferenceBatch
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback: compute from teacher-student feature differences
            teacher_features, student_features = predictions
            total_diff = 0
            for layer in teacher_features:
                diff = torch.mean((teacher_features[layer] - student_features[layer]) ** 2, dim=[1, 2, 3])
                total_diff += diff
            return total_diff

    def _collect_training_results(self, inputs, loss):
        """Hook: Collect STFPM training results and feature similarity metrics."""
        results = {'loss': loss.item()}
        
        # Calculate feature similarity metrics
        with torch.no_grad():
            # Get teacher-student features for metric calculation
            predictions = self.model(inputs['image'])
            teacher_features, student_features = predictions
            
            for metric_name, metric_fn in self.metrics.items():
                if metric_name == "feature_sim":
                    # Compute feature similarity for each layer and average
                    similarities = []
                    for layer in teacher_features:
                        layer_sim = metric_fn(teacher_features[layer], student_features[layer])
                        similarities.append(layer_sim)
                    results[metric_name] = sum(similarities) / len(similarities) if similarities else 0.0
                else:
                    results[metric_name] = 0.0

        return results

    def _collect_validation_results(self, inputs, loss):
        """Hook: Collect STFPM validation results and feature similarity metrics."""
        # Same as training results for STFPM
        return self._collect_training_results(inputs, loss)

    # ========================================================================
    # Properties - Required by BaseModeler
    # ========================================================================

    @property
    def learning_type(self):
        """STFPM uses one-class learning."""
        return "one_class"

    @property
    def trainer_arguments(self):
        """Get trainer arguments specific to STFPM."""
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }

    # ========================================================================
    # Optional: Custom methods for advanced functionality
    # ========================================================================

    def _compute_detailed_anomaly_scores(self, inputs):
        """Override: Compute detailed anomaly scores using STFPM-specific logic."""
        predictions = self.model(inputs['image'])

        if hasattr(predictions, 'anomaly_map') and hasattr(predictions, 'pred_score'):
            return {
                'anomaly_maps': predictions.anomaly_map,
                'pred_scores': predictions.pred_score
            }
        else:
            # Fallback: compute from teacher-student feature differences
            teacher_features, student_features = predictions

            # Use AnomalyMapGenerator from model if available
            if hasattr(self.model, 'anomaly_map_generator'):
                anomaly_map = self.model.anomaly_map_generator(
                    teacher_features=teacher_features,
                    student_features=student_features,
                    image_size=inputs['image'].shape[-2:],
                )
                pred_scores = torch.amax(anomaly_map, dim=(-2, -1))

                return {
                    'anomaly_maps': anomaly_map,
                    'pred_scores': pred_scores
                }
            else:
                # Simple fallback
                batch_size = next(iter(teacher_features.values())).shape[0]
                image_size = inputs['image'].shape[-2:]

                anomaly_maps = torch.zeros(batch_size, 1, *image_size, device=self.device)
                pred_scores = torch.zeros(batch_size, device=self.device)

                return {
                    'anomaly_maps': anomaly_maps,
                    'pred_scores': pred_scores
                }