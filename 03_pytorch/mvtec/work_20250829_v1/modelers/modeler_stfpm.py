import torch
from models.model_stfpm import STFPMModel, STFPMLoss, compute_anomaly_map
from modelers.modeler_base import BaseModeler

class STFPMModeler(BaseModeler):
    def __init__(self, model: STFPMModel, loss_fn=None, device="cuda"):
        super().__init__(model, loss_fn or STFPMLoss(), device)

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        outputs = self.model(images)  # (t_feats, s_feats)
        loss = self.loss_fn(outputs)
        return {"loss": loss}

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            outputs = self.model(images)
            loss = self.loss_fn(outputs)
        return {"val_loss": loss}

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        images = inputs["image"].to(self.device)
        t_feats, s_feats = self.model(images)
        anomaly_map = compute_anomaly_map(t_feats, s_feats, out_size=images.shape[-2:])
        score = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
        return {"anomaly_map": anomaly_map, "score": score}
