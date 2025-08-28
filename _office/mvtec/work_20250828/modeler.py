import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModeler:
    def __init__(self, model, loss_fn, metrics={}, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        for metric_name, metric_fn in self.metrics.items():
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



class AEModeler(BaseModeler):
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
        for metric_name, metric_fn in self.meterics.items():
            metric_value = metric_fn(reconstructed, inputs['image'])
            results[metric_name] = float(metric_value)
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        reconstructed, latent, features = self.model(inputs['image'])
        scores = torch.mean((reconstructed - inputs['image'])**2, dim=[1, 2, 3])
        return scores
