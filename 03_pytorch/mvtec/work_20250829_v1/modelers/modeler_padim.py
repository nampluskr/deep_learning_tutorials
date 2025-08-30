import torch
from models.model_padim import PadimModel, PadimLoss, compute_anomaly_map
from modelers.modeler_base import BaseModeler

class PadimModeler(BaseModeler):
    def __init__(self, model: PadimModel, loss_fn=None, device="cuda"):
        super().__init__(model, loss_fn or PadimLoss(), device)

    def train_step(self, batch):
        # PaDiM은 fit 단계에서 분포 추정만 수행 → train_step 없음
        return {"loss": torch.tensor(0.0)}

    def validate_step(self, batch):
        # Validation도 없음
        return {"val_loss": torch.tensor(0.0)}

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        images = inputs["image"].to(self.device)
        feats = self.model(images)
        scores = self.model.compute_anomaly_scores(feats)
        anomaly_map = compute_anomaly_map(scores, out_size=images.shape[-2:])
        return {"anomaly_map": anomaly_map, "score": scores}
