import torch
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple

from .model_base import TimmFeatureExtractor


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


class AnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)

    @staticmethod
    def compute_layer_map(teacher_features, student_features, image_size):
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map, size=image_size, align_corners=False, mode="bilinear")

    def compute_anomaly_map(self, teacher_features, student_features, image_size):
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.ones(batch_size, 1, image_size[0], image_size[1])
        for layer in teacher_features:
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer], image_size)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, **kwargs):
        if not ("teacher_features" in kwargs and "student_features" in kwargs):
            msg = f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}"
            raise ValueError(msg)

        teacher_features = kwargs["teacher_features"]
        student_features = kwargs["student_features"]
        image_size = kwargs["image_size"]

        return self.compute_anomaly_map(teacher_features, student_features, image_size)


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
        layer_losses = []
        for layer in teacher_features:
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        return torch.stack(layer_losses).sum()


class STFPMModel(nn.Module):
    def __init__(self, layers, backbone="resnet18"):
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.teacher_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=False, layers=layers).eval()
        self.student_model = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=False,  # 변경
            layers=layers,
            requires_grad=True,
        )

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator()

    def forward(self, images):
        output_size = images.shape[-2:]
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features = self.teacher_model(images)
        student_features = self.student_model(images)

        if self.tiler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)

        if self.training:
            return teacher_features, student_features

        anomaly_map = self.anomaly_map_generator(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)