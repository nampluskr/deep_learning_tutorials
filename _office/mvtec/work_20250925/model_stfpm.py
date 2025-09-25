from collections.abc import Sequence
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from feature_extractor import TimmFeatureExtractor
from trainer import BaseTrainer


class AnomalyMapGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)

    @staticmethod
    def compute_layer_map(teacher_features, student_features, image_size):
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)
        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, 
            p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map, size=image_size, align_corners=False, mode="bilinear")

    def compute_anomaly_map(self, teacher_features, student_features, image_size):
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.ones(batch_size, 1, image_size[0], image_size[1])
        for layer in teacher_features:
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer], 
                image_size)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map
        return anomaly_map

    def forward(self, **kwargs: dict[str, torch.Tensor]):
        if not ("teacher_features" in kwargs and "student_features" in kwargs):
            msg = f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}"
            raise ValueError(msg)

        teacher_features: dict[str, torch.Tensor] = kwargs["teacher_features"]
        student_features: dict[str, torch.Tensor] = kwargs["student_features"]
        image_size: tuple[int, int] | torch.Size = kwargs["image_size"]
        return self.compute_anomaly_map(teacher_features, student_features, image_size)


###########################################################
# Student-Teacher Model
###########################################################

class STFPM(nn.Module):
    def __init__(self, backbone="resnet50", layers=["layer1", "layer2", "layer3"]):
        super().__init__()
        self.tiler=None
        self.backbone = backbone

        self.teacher_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=True, 
            layers=layers).eval()
        self.student_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=False, 
            layers=layers, requires_grad=True)

        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False
        self.anomaly_map_generator = AnomalyMapGenerator()

    def forward(self, images):
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, torch.Tensor] = self.teacher_model(images)
        student_features: dict[str, torch.Tensor] = self.student_model(images)

        if self.tiler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)
        return teacher_features, student_features

    def predict(self, images):
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, torch.Tensor] = self.teacher_model(images)
        student_features: dict[str, torch.Tensor] = self.student_model(images)

        if self.tiler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)

        anomaly_map = self.anomaly_map_generator(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=images.shape[-2:],
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


#############################################################
# Loss for STFPM Model
#############################################################

class STFPMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats, student_feats):
        height, width = teacher_feats.shape[2:]
        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        return (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

    def forward(self, teacher_features, student_features):
        layer_losses: list[torch.Tensor] = []
        for layer in teacher_features:
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)
        return torch.stack(layer_losses).sum()


#############################################################
# Trainer for STFPM Model
#############################################################

class STFPMTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.SGD(params=model.student_model.parameters(),
                lr=0.4, momentum=0.9, dampening=0.0, weight_decay=0.001)
        self.loss_fn = loss_fn or STFPMLoss()
        self.metrics = metrics or {}

    def run_epoch(self, loader, mode='train', desc=""):
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0

        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                batch_size = images.size(0)
                num_images += batch_size

                teacher_features, student_features = self.model(images)
                loss = self.loss_fn(teacher_features, student_features)

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                results["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        metric_value = metric_fn(teacher_features, student_features)
                        results[name] += metric_value.item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})

        return {name: value / num_images for name, value in results.items()}


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model = STFPMModel(
        backbone="resnet50",
        layers=["layer1", "layer2", "layer3"]).to(device)
    x = torch.randn(8, 3, 256, 256).to(device)
    teacher_features, student_features= model(x)

    print()
    for name, feat in teacher_features.items():
        print(f"Teacher {name}: {feat.shape}")
    
    print()
    for name, feat in student_features.items():
        print(f"Student {name}: {feat.shape}")

    loss_fn = STFPMLoss()
    loss = loss_fn(teacher_features, student_features)
    print()
    print(f"Loss={loss.item():.4f}")

    predictions = model.predict(x)
    print()
    print(f"pred_socre:  {predictions['pred_score'].shape}")
    print(f"anoamly_map: {predictions['anomaly_map'].shape}")
