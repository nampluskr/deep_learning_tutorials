import torch
from torch import optim

from .modeler_base import BaseModeler


class AEModeler(BaseModeler):
    def __init__(self, model, loss_fn, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        reconstructed, latent, features = self.model(inputs['image'])
        loss = self.loss_fn(reconstructed, inputs['image'])
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(reconstructed, inputs['image'])
                results[metric_name] = float(metric_value)
        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        reconstructed, latent, features = self.model(inputs['image'])
        loss = self.loss_fn(reconstructed, inputs['image'])

        results = {'loss': loss.item()}
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(reconstructed, inputs['image'])
            results[metric_name] = float(metric_value)
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])
        
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            reconstructed, latent, features = predictions
            scores = torch.mean((inputs['image'] - reconstructed)**2, dim=[1, 2, 3])
            return scores

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
                reconstructed, latent, features = predictions
                anomaly_maps = torch.mean((inputs['image'] - reconstructed)**2, dim=1, keepdim=True)
                pred_scores = torch.mean((inputs['image'] - reconstructed)**2, dim=[1, 2, 3])
                return {
                    'anomaly_maps': anomaly_maps,
                    'pred_scores': pred_scores
                }

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    @property
    def learning_type(self):
        return "one_class"

    @property
    def trainer_arguments(self):
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }


if __name__ == "__main__":
    pass