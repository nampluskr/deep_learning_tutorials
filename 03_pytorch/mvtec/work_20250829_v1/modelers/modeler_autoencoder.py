import torch
from models.model_autoencoder import AutoencoderLoss, compute_anomaly_map
from modelers.modeler_base import BaseModeler

class AutoencoderModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, device="cuda"):
        super().__init__(model, loss_fn or AutoencoderLoss(), device)

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        recons = self.model(images)
        loss = self.loss_fn(recons, images)
        return {"loss": loss}

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            recons = self.model(images)
            loss = self.loss_fn(recons, images)
        return {"val_loss": loss}

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        images = inputs["image"].to(self.device)
        recons = self.model(images)
        anomaly_map, score = compute_anomaly_map(images, recons)
        return {"reconstruction": recons, "anomaly_map": anomaly_map, "score": score}
