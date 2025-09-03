import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModeler(ABC):
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
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
        device_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                device_inputs[key] = value.to(self.device, non_blocking=True)
            else:
                device_inputs[key] = value
        return device_inputs

    def get_metric_names(self):
        return list(self.metrics.keys())

    @abstractmethod
    def train_step(self, inputs, optimizer):
        pass

    @abstractmethod
    def validate_step(self, inputs):
        pass

    @abstractmethod
    def predict_step(self, inputs):
        pass

    def compute_image_scores(self, anomaly_maps):
        batch_size = anomaly_maps.shape[0]
        flattened = anomaly_maps.view(batch_size, -1)
        image_scores = torch.max(flattened, dim=1)[0]
        return image_scores

    def compute_anomaly_scores(self, inputs):
        with torch.no_grad():
            self.model.eval()
            inputs = self.to_device(inputs)
            
            predictions = self.model(inputs['image'])
            
            if hasattr(predictions, 'anomaly_map'):
                return {
                    'anomaly_maps': predictions.anomaly_map,
                    'pred_scores': predictions.pred_score
                }
            else:
                anomaly_maps = predictions
                pred_scores = self.compute_image_scores(anomaly_maps)
                return {
                    'anomaly_maps': anomaly_maps,
                    'pred_scores': pred_scores
                }

    @property
    @abstractmethod
    def learning_type(self):
        pass

    @property
    def trainer_arguments(self):
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def get_model(self):
        return self.model


if __name__ == "__main__":
    pass