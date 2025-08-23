import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeaturePyramidExtractor(nn.Module):
    """Feature pyramid extractor for STFPM"""
    def __init__(self, backbone='resnet18', layers=['layer1', 'layer2', 'layer3'], weights_path=None):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers

        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=None)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=None)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if weights_path is not None:
            self._load_custom_weights(weights_path)

        # backbone 전체를 사용, 후처리는 hook으로 feature 추출
        self.features = {}
        self.hook_handles = []
        self._register_hooks()

    def _load_custom_weights(self, weights_path):
        try:
            print(f"Loading custom weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print("Custom weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        for name, module in self.backbone.named_modules():
            if any(layer in name for layer in self.layers):
                handle = module.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)

    def forward(self, x):
        self.features.clear()
        _ = self.backbone(x)
        extracted = []
        for layer in self.layers:
            for name, feat in self.features.items():
                if layer in name:
                    extracted.append(feat)
                    break
        return extracted

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()


class TeacherNetwork(FeaturePyramidExtractor):
    """Frozen teacher"""
    def __init__(self, backbone='resnet18', layers=['layer1','layer2','layer3'], weights_path=None):
        super().__init__(backbone, layers, weights_path)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


class StudentNetwork(FeaturePyramidExtractor):
    """Trainable student"""
    def __init__(self, backbone='resnet18', layers=['layer1','layer2','layer3']):
        super().__init__(backbone, layers, weights_path=None)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class STFPM(nn.Module):
    """STFPM Student-Teacher Feature Pyramid Matching"""
    def __init__(self, backbone='resnet18', layers=['layer1','layer2','layer3'], weights_path=None):
        super().__init__()
        self.model_type = "stfpm"
        self.teacher = TeacherNetwork(backbone, layers, weights_path)
        self.student = StudentNetwork(backbone, layers)

    def forward(self, batch_data):
        x = batch_data["input"]
        with torch.no_grad():
            teacher_feats = self.teacher(x)
        student_feats = self.student(x)
        return {"teacher_features": teacher_feats,
                "student_features": student_feats,
                "input": x}

    def compute_loss(self, outputs):
        teacher_feats = outputs["teacher_features"]
        student_feats = outputs["student_features"]
        losses = []
        for t, s in zip(teacher_feats, student_feats):
            losses.append(F.mse_loss(s, t))
        return torch.stack(losses).mean()

    def compute_anomaly_scores(self, outputs):
        teacher_feats = outputs["teacher_features"]
        student_feats = outputs["student_features"]
        maps = []
        for t, s in zip(teacher_feats, student_feats):
            diff = (t - s) ** 2
            amap = diff.mean(dim=1, keepdim=True)
            amap = F.interpolate(amap, size=teacher_feats[0].shape[-2:],
                                 mode='bilinear', align_corners=False)
            maps.append(amap)
        combined_map = torch.stack(maps).mean(dim=0)
        image_score = F.adaptive_avg_pool2d(combined_map, (1,1)).squeeze()
        return image_score, combined_map

    def train_step(self, data, optimizer, loss_fn, metrics, device):
        inputs = data["image"].to(device)
        outputs = self.forward({"input": inputs})
        loss = self.compute_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            image_score, _ = self.compute_anomaly_scores(outputs)
            results["anomaly_score"] = image_score.mean().item()
        return results

    @torch.no_grad()
    def validate_step(self, data, loss_fn, metrics, device):
        inputs = data["image"].to(device)
        outputs = self.forward({"input": inputs})
        loss = self.compute_loss(outputs)

        results = {"loss": loss.item()}
        image_score, _ = self.compute_anomaly_scores(outputs)
        results["anomaly_score"] = image_score.mean().item()
        return results


# === Loss / Metric factory functions ===

def stfpm_loss(pred, target=None):
    """Feature matching loss wrapper"""
    return pred["model"].compute_loss(pred)


def stfpm_anomaly_score_metric(pred, target=None):
    """Image-level anomaly score"""
    if isinstance(pred, dict) and "teacher_features" in pred:
        image_score, _ = pred["model"].compute_anomaly_scores(pred)
        return image_score.mean().item()
    return 0.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STFPM(backbone="resnet18", layers=["layer1","layer2","layer3"]).to(device)
    dummy = {"input": torch.randn(2,3,256,256).to(device)}
    outputs = model(dummy)
    loss = model.compute_loss(outputs)
    print("STFPM test forward:", loss.item())
