from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim


# ===================================================================
# Base Modeler
# ===================================================================

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

    @abstractmethod
    def compute_anomaly_scores(self, inputs):
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def get_model(self):
        return self.model


# ===================================================================
# Autoencoer Modeler
# ===================================================================

class AEModeler(BaseModeler):
    def __init__(self, model, loss_fn, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
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

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.train()
        inputs = self.to_device(inputs)

        # Training mode: (reconstructed, latent, features)
        reconstructed, latent, features = self.model(inputs['image'])
        loss = self.loss_fn(reconstructed, inputs['image'])

        results = {'loss': loss.item()}
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(reconstructed, inputs['image'])
                results[metric_name] = float(metric_value)
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()  # Use inference mode for prediction
        inputs = self.to_device(inputs)
        predictions = self.model(inputs['image'])
        
        if isinstance(predictions, dict) and 'pred_score' in predictions:
            return predictions['pred_score']
        else:
            reconstructed, _, _ = predictions
            scores = torch.mean((inputs['image'] - reconstructed)**2, dim=[1, 2, 3])
            return scores

    @torch.no_grad()
    def compute_anomaly_scores(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)
        predictions = self.model(inputs['image'])
        
        if isinstance(predictions, dict) and 'anomaly_map' in predictions:
            return dict(anomaly_maps=anomaly_maps, pred_scores=pred_scores)
        else:
            reconstructed, latent, features = predictions
            anomaly_maps = torch.mean((inputs['image'] - reconstructed)**2, dim=1, keepdim=True)
            pred_scores = torch.mean((inputs['image'] - reconstructed)**2, dim=[1, 2, 3])
            return dict(anomaly_maps=anomaly_maps, pred_scores=pred_scores)


# ===================================================================
# STFPM Modeler
# ===================================================================

class STFPMModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        super().__init__(model, loss_fn, metrics, device)

    def train_step(self, inputs, optimizer):
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

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.train()
        inputs = self.to_device(inputs)

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

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)
        predictions = self.model(inputs['image'])

        if isinstance(predictions, dict) and 'pred_score' in predictions:
            return predictions['pred_score']
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
                return pred_scores
            else:
                # Simple fallback
                batch_size = next(iter(teacher_features.values())).shape[0]
                image_size = inputs['image'].shape[-2:]

                anomaly_maps = torch.zeros(batch_size, 1, *image_size, device=self.device)
                pred_scores = torch.zeros(batch_size, device=self.device)
                return pred_scores

    @torch.no_grad()
    def compute_anomaly_scores(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)
        predictions = self.model(inputs['image'])

        if isinstance(predictions, dict) and 'pred_score' in predictions:
            return dict(anomaly_maps=predictions['anomaly_map'], 
                        pred_scores=predictions['pred_score'])
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
                return dict(anomaly_maps=anomaly_maps, pred_scores=pred_scores)
            else:
                # Simple fallback
                batch_size = next(iter(teacher_features.values())).shape[0]
                image_size = inputs['image'].shape[-2:]

                anomaly_maps = torch.zeros(batch_size, 1, *image_size, device=self.device)
                pred_scores = torch.zeros(batch_size, device=self.device)
                return dict(anomaly_maps=anomaly_maps, pred_scores=pred_scores)


if __name__ == "__main__":
    pass
