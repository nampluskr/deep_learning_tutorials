import torch
import torch.nn.functional as F
from torch import optim

from .modeler_base import BaseModeler


class STFPMModeler(BaseModeler):
    """STFPM (Student-Teacher Feature Pyramid Matching) modeler implementation."""
    
    def __init__(self, model, loss_fn, metrics=None, device=None):
        """Initialize STFPM modeler with teacher-student model."""
        super().__init__(model, loss_fn, metrics, device)

    def forward(self, inputs):
        """Core forward pass returning teacher and student features."""
        images = inputs['image']
        
        # STFPM model always returns teacher-student features in training mode
        # Force training mode to get consistent feature extraction
        was_training = self.model.training
        self.model.train()
        
        try:
            outputs = self.model(images)
            
            # Always expect training mode output: (teacher_features, student_features)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                teacher_features, student_features = outputs
                return {
                    'teacher_features': teacher_features,
                    'student_features': student_features,
                    'image_size': images.shape[-2:]
                }
            else:
                # Fallback: handle unexpected output format
                raise ValueError(f"Unexpected STFPM output format: {type(outputs)}")
        finally:
            # Restore original training state if needed
            if not was_training:
                self.model.eval()

    def compute_loss(self, outputs, inputs):
        """Compute STFPM loss from teacher-student feature differences."""
        teacher_features = outputs['teacher_features']
        student_features = outputs['student_features']
        return self.loss_fn(teacher_features, student_features)

    def generate_anomaly_maps(self, outputs, inputs):
        """Generate pixel-level anomaly maps from teacher-student feature differences."""
        teacher_features = outputs['teacher_features']
        student_features = outputs['student_features']
        image_size = outputs['image_size']
        
        # Use model's anomaly map generator if available
        if hasattr(self.model, 'anomaly_map_generator'):
            anomaly_map = self.model.anomaly_map_generator(
                teacher_features=teacher_features,
                student_features=student_features,
                image_size=image_size,
            )
            return anomaly_map
        else:
            # Fallback: compute anomaly map manually
            return self._compute_anomaly_map_fallback(teacher_features, student_features, image_size)

    def _compute_anomaly_map_fallback(self, teacher_features, student_features, image_size):
        """Fallback method to compute anomaly maps from feature differences."""
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.ones(batch_size, 1, image_size[0], image_size[1], device=self.device)
        
        for layer in teacher_features:
            # Normalize features
            norm_teacher = F.normalize(teacher_features[layer])
            norm_student = F.normalize(student_features[layer])
            
            # Compute layer-wise anomaly map
            layer_map = 0.5 * torch.norm(norm_teacher - norm_student, p=2, dim=1, keepdim=True) ** 2
            layer_map = F.interpolate(layer_map, size=image_size, mode="bilinear", align_corners=False)
            
            # Multiply anomaly maps (as in original STFPM)
            anomaly_map *= layer_map
        
        return anomaly_map

    def evaluate_metric(self, metric_fn, metric_name, outputs, inputs):
        """Evaluate STFPM-specific metrics."""
        if metric_name == "feature_sim" and 'teacher_features' in outputs and 'student_features' in outputs:
            teacher_features = outputs['teacher_features']
            student_features = outputs['student_features']
            
            # Compute feature similarity for each layer and average
            similarities = []
            for layer in teacher_features:
                layer_sim = metric_fn(teacher_features[layer], student_features[layer])
                similarities.append(layer_sim)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
        else:
            return 0.0

    @property
    def learning_type(self):
        """STFPM uses one-class learning."""
        return "one_class"

    @property
    def trainer_arguments(self):
        """Return trainer-specific arguments for STFPM."""
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }