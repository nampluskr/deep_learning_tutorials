import torch
from models.model_vae import VAELoss, compute_anomaly_map
from modelers.modeler_base import BaseModeler

class VAEModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, device="cuda"):
        super().__init__(model, loss_fn or VAELoss(), device)

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        recon, mu, logvar = self.model(images)
        loss = self.loss_fn(recon, images, mu, logvar)
        return {"loss": loss}

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            recon, mu, logvar = self.model(images)
            loss = self.loss_fn(recon, images, mu, logvar)
        return {"val_loss": loss}

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        images = inputs["image"].to(self.device)
        recon, mu, logvar = self.model(images)
        anomaly_map, score = compute_anomaly_map(images, recon)
        return {"reconstruction": recon, "anomaly_map": anomaly_map, "score": score}
