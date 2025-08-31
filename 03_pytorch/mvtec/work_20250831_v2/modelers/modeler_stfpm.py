import torch
from torch import optim

from .modeler_base import BaseModeler


class STFPMModeler(BaseModeler):
    def __init__(self, model, loss_fn, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        predictions = self.model(inputs['image'])
        
        # Training mode: (teacher_features, student_features)
        teacher_features, student_features = predictions
        loss = self.loss_fn(teacher_features, student_features)
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        
        # STFPM doesn't use traditional metrics during training
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = 0.0  # Placeholder
            
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])
        
        # Inference mode: InferenceBatch(pred_score, anomaly_map)
        if hasattr(predictions, 'pred_score'):
            scores = predictions.pred_score
            labels = inputs['label']

            # Normal vs Anomaly score distribution analysis (like PaDiM)
            normal_mask = labels == 0
            anomaly_mask = labels == 1

            normal_scores = scores[normal_mask] if normal_mask.any() else torch.tensor([0.0])
            anomaly_scores = scores[anomaly_mask] if anomaly_mask.any() else torch.tensor([0.0])

            results = {
                'loss': 0.0,  # No loss for inference mode
                'score_mean': scores.mean().item(),
                'score_std': scores.std().item(),
                'normal_mean': normal_scores.mean().item(),
                'anomaly_mean': anomaly_scores.mean().item(),
                'separation': (anomaly_scores.mean() - normal_scores.mean()).item() if anomaly_mask.any() and normal_mask.any() else 0.0,
            }
        else:
            # Training mode: (teacher_features, student_features)
            teacher_features, student_features = predictions
            loss = self.loss_fn(teacher_features, student_features)
            
            results = {'loss': loss.item()}
            for metric_name, metric_fn in self.metrics.items():
                results[metric_name] = 0.0  # STFPM doesn't use reconstruction metrics
        
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])
        
        # Inference mode: InferenceBatch(pred_score, anomaly_map)
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback: if training mode output
            teacher_features, student_features = predictions
            # Compute feature difference as anomaly score
            total_diff = 0
            for layer in teacher_features:
                diff = torch.mean((teacher_features[layer] - student_features[layer]) ** 2, dim=[1, 2, 3])
                total_diff += diff
            return total_diff

    def compute_anomaly_scores(self, inputs):
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
                # Fallback: compute from teacher-student feature differences
                teacher_features, student_features = predictions
                
                # Use AnomalyMapGenerator from model to compute maps
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

    def configure_optimizers(self):
        # STFPM optimizes only the student model parameters
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
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