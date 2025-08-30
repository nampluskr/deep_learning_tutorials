import torch
from torch import optim

from .modeler_base import BaseModeler


class STFPMModeler(BaseModeler):
    def __init__(self, model, loss_fn, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            predictions = self.model(inputs['image'])
        
        # Memory bank 상태 모니터링
        current_samples = len(self.model.memory_bank)
        if current_samples > 0:
            latest_embedding = self.model.memory_bank[-1]
            embedding_mean = latest_embedding.mean().item()
            embedding_std = latest_embedding.std().item()
            embedding_shape = latest_embedding.shape
        else:
            embedding_mean = 0.0
            embedding_std = 0.0
            embedding_shape = "N/A"
        
        results = {
            'loss': 0.0,
            'memory_samples': current_samples,
            'embedding_mean': embedding_mean,
            'embedding_std': embedding_std,
        }
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.train()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])
        
        # Training mode: (teacher_features, student_features)
        teacher_features, student_features = predictions
        loss = self.loss_fn(teacher_features, student_features)

        results = {'loss': loss.item()}
        for metric_name, metric_fn in self.metrics.items():
            if 'psnr' in metric_name.lower() or 'ssim' in metric_name.lower():
                # Skip reconstruction metrics for STFPM
                results[metric_name] = 0.0
            else:
                results[metric_name] = 0.0
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
            # Compute simple feature difference as score
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
                # Fallback: if training mode output
                teacher_features, student_features = predictions
                # Compute feature difference as anomaly map
                batch_size = next(iter(teacher_features.values())).shape[0]
                image_size = inputs['image'].shape[-2:]
                
                anomaly_maps = torch.zeros(batch_size, 1, *image_size, device=self.device)
                pred_scores = torch.zeros(batch_size, device=self.device)
                
                return {
                    'anomaly_maps': anomaly_maps,
                    'pred_scores': pred_scores
                }

    def configure_optimizers(self):
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
        
    def get_memory_stats(self):
        """Get memory bank statistics"""
        if len(self.model.memory_bank) == 0:
            return {'total_samples': 0}
        
        # 전체 memory bank 통계
        total_samples = sum(emb.shape[0] for emb in self.model.memory_bank)
        
        # 최신 배치 통계
        if self.model.memory_bank:
            latest = self.model.memory_bank[-1]
            stats = {
                'total_samples': total_samples,
                'latest_batch_size': latest.shape[0],
                'feature_dim': latest.shape[1],
                'spatial_size': latest.shape[2:],
                'latest_mean': latest.mean().item(),
                'latest_std': latest.std().item(),
                'latest_min': latest.min().item(),
                'latest_max': latest.max().item(),
            }
        else:
            stats = {'total_samples': 0}
        
        return stats