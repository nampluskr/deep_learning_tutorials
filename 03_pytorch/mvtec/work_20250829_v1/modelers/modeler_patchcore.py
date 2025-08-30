import torch
from models.model_patchcore import PatchcoreModel, PatchcoreLoss, compute_anomaly_map
from modelers.modeler_base import BaseModeler

class PatchcoreModeler(BaseModeler):
    def __init__(self, model: PatchcoreModel, loss_fn=None, device="cuda"):
        super().__init__(model, loss_fn or PatchcoreLoss(), device)

    def train_step(self, batch):
        # PatchCore는 메모리 뱅크 구축만 필요
        return {"loss": torch.tensor(0.0)}

    def validate_step(self, batch):
        return {"val_loss": torch.tensor(0.0)}

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        images = inputs["image"].to(self.device)
        feats = self.model(images)
        scores = self.model.compute_anomaly_scores(feats)
        b = images.size(0)
        side = int((scores.numel() // b) ** 0.5)
        patch_scores = scores.view(b, side, side)
        anomaly_map = compute_anomaly_map(patch_scores, image_size=images.shape[-2:])
        return {"anomaly_map": anomaly_map, "score": scores}
