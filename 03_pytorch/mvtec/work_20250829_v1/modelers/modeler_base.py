# modelers/modeler_base.py
import torch

class BaseModeler:
    """Base wrapper for anomaly detection models"""

    def __init__(self, model, loss_fn=None, device="cuda"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, batch):
        self.model.train()
        images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
        outputs = self.model(images)
        loss = self.loss_fn(outputs, images) if self.loss_fn else None
        return loss

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            outputs = self.model(images)
            loss = self.loss_fn(outputs, images) if self.loss_fn else None
        return loss

    @torch.no_grad()
    def predict_step(self, inputs):
        raise NotImplementedError("Must be implemented in subclasses")
