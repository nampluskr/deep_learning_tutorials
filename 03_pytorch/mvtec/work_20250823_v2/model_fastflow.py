import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor using ResNet for FastFlow"""
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'], weights_path=None):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers

        # Load backbone without pretrained weights
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=None)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=None)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Load custom weights if provided
        if weights_path is not None:
            self._load_custom_weights(weights_path)

        # NOTE: backbone 전체를 유지해야 hook 이 layer2, layer3에 걸림
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # ❌ 제거

        # Freeze parameters (optional: fine-tuning 시 주석 처리 가능)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Register hooks for feature extraction
        self.features = {}
        self.hook_handles = []
        self._register_hooks()

    def _load_custom_weights(self, weights_path):
        """Load custom pretrained weights from file"""
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
            print(f"Warning: Could not load custom weights from {weights_path}: {e}")
            print("Using random initialization instead")

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        for name, module in self.backbone.named_modules():
            if any(layer in name for layer in self.layers):
                handle = module.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)

    def forward(self, x):
        """Extract features from specified layers"""
        self.features.clear()
        _ = self.backbone(x)

        # Collect features from specified layers
        extracted_features = []
        for layer in self.layers:
            for name, feature in self.features.items():
                if layer in name:
                    extracted_features.append(feature)
                    break

        return extracted_features

    def __del__(self):
        """Clean up hooks"""
        for handle in self.hook_handles:
            handle.remove()


class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flow"""
    def __init__(self, channels, hidden_dim=512, mask_type='checkerboard'):
        super().__init__()
        self.channels = channels
        self.mask_type = mask_type
        self.hidden_dim = hidden_dim
        self.register_buffer('mask', None)

        # 수정: 채널을 절반으로 split하지 않고 full channel 사용
        self.scale_net = self._create_transform_net(channels, channels, hidden_dim)
        self.translation_net = self._create_transform_net(channels, channels, hidden_dim)

    def _create_checkerboard_mask(self, channels, height, width, device):
        mask = torch.zeros(1, 1, height, width, device=device)
        for i in range(height):
            for j in range(width):
                mask[0, 0, i, j] = (i + j) % 2
        mask = mask.repeat(1, channels, 1, 1)
        return mask

    def _create_channel_mask(self, channels, device):
        mask = torch.zeros(1, channels, 1, 1, device=device)
        mask[:, :channels // 2] = 1
        return mask

    def _create_transform_net(self, input_dim, output_dim, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, x, reverse=False):
        b, c, h, w = x.shape
        device = x.device

        if self.mask_type == 'checkerboard':
            if self.mask is None or self.mask.shape[-2:] != (h, w):
                self.mask = self._create_checkerboard_mask(self.channels, h, w, device)
        elif self.mask_type == 'channel':
            if self.mask is None or self.mask.shape[1] != self.channels:
                self.mask = self._create_channel_mask(self.channels, device)

        if not reverse:
            return self._forward(x)
        else:
            return self._inverse(x)

    def _forward(self, x):
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)

        scale = self.scale_net(x_masked)
        translation = self.translation_net(x_masked)

        scale = torch.tanh(scale)
        y_unmasked = x_unmasked * torch.exp(scale) + translation

        y = x_masked + y_unmasked
        log_det = torch.sum(scale, dim=[1, 2, 3])
        return y, log_det

    def _inverse(self, y):
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)

        scale = self.scale_net(y_masked)
        translation = self.translation_net(y_masked)

        scale = torch.tanh(scale)
        x_unmasked = (y_unmasked - translation) * torch.exp(-scale)

        x = y_masked + x_unmasked
        log_det = -torch.sum(scale, dim=[1, 2, 3])
        return x, log_det


class NormalizingFlow(nn.Module):
    """Normalizing Flow for FastFlow"""
    def __init__(self, input_dim, n_flows=8, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.n_flows = n_flows
        self.flows = nn.ModuleList()

        for i in range(n_flows):
            mask_type = 'checkerboard' if i % 2 == 0 else 'channel'
            self.flows.append(CouplingLayer(input_dim, hidden_dim, mask_type))

    def forward(self, x, reverse=False):
        log_det_total = 0
        if not reverse:
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_total += log_det
        else:
            for flow in reversed(self.flows):
                x, log_det = flow(x, reverse=True)
                log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x):
        z, log_det = self.forward(x)
        num_dims = z[0].numel()
        log_prior = -0.5 * torch.sum(z ** 2, dim=[1, 2, 3]) \
                    - 0.5 * num_dims * torch.log(torch.tensor(2 * math.pi, device=z.device))
        log_probs = log_prior + log_det
        return log_probs


class FastFlow(nn.Module):
    """FastFlow: Real-time Anomaly Detection via Normalizing Flow"""
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'],
                 flow_steps=8, hidden_dim=512, weights_path=None):
        super().__init__()
        self.model_type = "fastflow"
        self.backbone_name = backbone
        self.layers = layers

        self.feature_extractor = ResNetFeatureExtractor(backbone, layers, weights_path)
        self._get_feature_dimensions()

        self.flows = nn.ModuleDict()
        for i, (layer, dim) in enumerate(zip(layers, self.feature_dims)):
            self.flows[f'flow_{i}'] = NormalizingFlow(dim, flow_steps, hidden_dim)

    def _get_feature_dimensions(self):
        if self.backbone_name in ['resnet18', 'resnet34']:
            dim_map = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        elif self.backbone_name == 'resnet50':
            dim_map = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        self.feature_dims = [dim_map[layer] for layer in self.layers]

    def forward(self, batch_data):
        x = batch_data['input']
        features_list = self.feature_extractor(x)

        log_probs = []
        flow_outputs = []
        for i, features in enumerate(features_list):
            flow = self.flows[f'flow_{i}']
            log_prob = flow.log_prob(features)
            log_probs.append(log_prob)
            flow_outputs.append(features)

        return {
            'log_probs': log_probs,
            'features': flow_outputs,
            'input': x
        }

    def compute_anomaly_scores(self, outputs):
        total_log_prob = 0
        for log_prob in outputs['log_probs']:
            log_prob_mean = torch.mean(log_prob.view(log_prob.size(0), -1), dim=1)
            total_log_prob += log_prob_mean
        anomaly_scores = -total_log_prob
        return anomaly_scores

    def train_step(self, data, optimizer, loss_fn, metrics, device):
        inputs = data["image"].to(device)
        optimizer.zero_grad()

        outputs = self.forward({"input": inputs})
        loss = loss_fn(outputs)
        loss.backward()
        optimizer.step()

        results = {"loss": loss.item()}
        with torch.no_grad():
            for metric_name, metric_fn in metrics.items():
                results[metric_name] = float(metric_fn(outputs))
        return results

    @torch.no_grad()
    def validate_step(self, data, loss_fn, metrics, device):
        inputs = data["image"].to(device)
        outputs = self.forward({"input": inputs})
        loss = loss_fn(outputs)

        results = {"loss": loss.item()}
        for metric_name, metric_fn in metrics.items():
            results[metric_name] = float(metric_fn(outputs))
        return results


# Loss Functions
def fastflow_loss(pred, target=None):
    if isinstance(pred, dict) and 'log_probs' in pred:
        log_probs = pred['log_probs']
        if not log_probs:  # safety guard
            device = pred['input'].device
            return torch.tensor(1e-6, device=device)

        # Compute NLL for each flow scale
        losses = []
        for log_prob in log_probs:
            # log_prob: (batch,) or (batch, H, W)
            nll = -torch.mean(log_prob)   # mean over batch
            losses.append(nll)

        # Take mean over all flows to stabilize scale
        total_loss = torch.stack(losses).mean()
        return total_loss

    else:
        device = pred.device if torch.is_tensor(pred) else torch.device('cpu')
        return torch.tensor(1e-6, device=device)


def fastflow_log_prob_metric(pred, target=None):
    if isinstance(pred, dict) and 'log_probs' in pred and pred['log_probs']:
        total_log_prob = 0
        count = 0
        for log_prob in pred['log_probs']:
            total_log_prob += torch.mean(log_prob)
            count += 1
        return (total_log_prob / count).item() if count > 0 else 0.0
    return 0.0


def fastflow_anomaly_score_metric(pred, target=None):
    if isinstance(pred, dict) and 'log_probs' in pred and pred['log_probs']:
        total_log_prob = 0
        count = 0
        for log_prob in pred['log_probs']:
            total_log_prob += torch.mean(log_prob)
            count += 1
        avg_log_prob = total_log_prob / count if count > 0 else 0.0
        return (-avg_log_prob).item()
    return 1.0


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastFlow(backbone='resnet18', layers=['layer2', 'layer3'], flow_steps=4, hidden_dim=256).to(device)

    batch_data = {'input': torch.randn(2, 3, 256, 256).to(device)}
    outputs = model(batch_data)
    print("Output keys:", outputs.keys())
    loss = fastflow_loss(outputs)
    print("Loss:", loss.item())
