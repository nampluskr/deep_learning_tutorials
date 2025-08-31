from __future__ import annotations

import torch
from torch import nn, linspace
import torch.nn.functional as F
import os
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from pytorch_msssim import ssim
# import lpips


BACKBONE_DIR = os.path.abspath(os.path.join("..", "..", "backbones"))

def set_backbone_dir(path):
    """Set global backbone directory for metrics"""
    global BACKBONE_DIR
    BACKBONE_DIR = path

# Available metric classes
AVAILABLE_METRICS = {
    'auroc': 'AUROCMetric',
    'auc': 'AUROCMetric',
    'aupr': 'AUPRMetric',
    'ap': 'AUPRMetric',
    'accuracy': 'AccuracyMetric',
    'precision': 'PrecisionMetric',
    'recall': 'RecallMetric',
    'f1': 'F1Metric',
    'threshold': 'OptimalThresholdMetric',
    'psnr': 'PSNRMetric',
    'ssim': 'SSIMMetric',
    'lpips': 'LPIPSMetric',
}


def get_metric(name, **params):
    """Factory function to create metric instances"""
    name = name.lower()
    if name not in AVAILABLE_METRICS:
        available_names = list(AVAILABLE_METRICS.keys())
        raise ValueError(f"Unknown metric: {name}. Available metrics: {available_names}")

    class_name = AVAILABLE_METRICS[name]
    metric_class = globals()[class_name]
    return metric_class(**params)


def thresholds_between_min_and_max(preds, num_thresholds=100, device=None):
    return linspace(start=preds.min(), end=preds.max(), steps=num_thresholds, device=device)


def thresholds_between_0_and_1(num_thresholds=100, device=None):
    return linspace(start=0, end=1, steps=num_thresholds, device=device)


class AUROCMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, scores):
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return roc_auc_score(labels_np, scores_np)


class AUPRMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, scores):
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return average_precision_score(labels_np, scores_np)


class AccuracyMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return accuracy_score(labels_np, predictions_np)


class PrecisionMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return precision_score(labels_np, predictions_np, zero_division=0)


class RecallMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return recall_score(labels_np, predictions_np, zero_division=0)


class F1Metric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return f1_score(labels_np, predictions_np, zero_division=0)


class PSNRMetric(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, preds, targets):
        mse = F.mse_loss(preds, targets, reduction='mean')
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(torch.tensor(self.max_val) / torch.sqrt(mse))
        return psnr.item()


class SSIMMetric(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        ssim_value = ssim(preds, targets, data_range=self.data_range, size_average=True)
        return ssim_value.item()


# class LPIPSMetric(nn.Module):
#     def __init__(self, net='alex', version='0.1'):
#         super().__init__()
#         self.net = net
#         self.version = version
        
#         # LPIPS 내부적으로 여전히 다운로드 시도할 수 있음
#         # 환경변수로 torch hub 비활성화 필요
#         import os
#         os.environ['TORCH_HOME'] = BACKBONE_DIR  # torch hub 경로 변경
        
#         # Create LPIPS model without pretrained weights
#         self.loss_fn = lpips.LPIPS(net=net, version=version, pretrained=False, verbose=False)
        
#         # Load local weights directly
#         local_weights_path = os.path.join(BACKBONE_DIR, f"lpips_{net}.pth")
#         state_dict = torch.load(local_weights_path, map_location='cpu')
#         self.loss_fn.load_state_dict(state_dict)

#     def forward(self, preds, targets):
#         preds_normalized = preds * 2.0 - 1.0
#         targets_normalized = targets * 2.0 - 1.0

#         distance = self.loss_fn(preds_normalized, targets_normalized)
#         return distance.mean().item()


class OptimalThresholdMetric(nn.Module):
    def __init__(self, method="f1", percentile=95):
        super().__init__()
        self.method = method
        self.percentile = percentile

    def forward(self, labels, scores):
        from sklearn.metrics import roc_curve

        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()

        if self.method == "roc":
            fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
            optimal_idx = (tpr - fpr).argmax()
            return thresholds[optimal_idx]

        elif self.method == "f1":
            thresholds_tensor = thresholds_between_min_and_max(scores, num_thresholds=100)
            best_f1 = 0
            best_threshold = 0.5

            f1_metric = F1Metric()
            for threshold in thresholds_tensor:
                preds = (scores >= threshold).float()
                f1 = f1_metric(labels, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold.item()

            return best_threshold

        elif self.method == "percentile":
            # Use normal samples only (label == 0)
            normal_mask = labels_np == 0
            if normal_mask.sum() == 0:
                return 0.5  # fallback if no normal samples

            normal_scores = scores_np[normal_mask]
            threshold = float(torch.quantile(torch.tensor(normal_scores), self.percentile / 100.0))
            return threshold

        else:
            return 0.5