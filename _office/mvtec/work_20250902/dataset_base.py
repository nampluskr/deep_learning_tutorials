"""Base metrics used across all anomaly detection models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve
)


class AUROCMetric(nn.Module):
    """Area Under ROC Curve metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, labels, scores):
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return roc_auc_score(labels_np, scores_np)


class AUPRMetric(nn.Module):
    """Area Under Precision-Recall Curve metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, labels, scores):
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return average_precision_score(labels_np, scores_np)


class AccuracyMetric(nn.Module):
    """Classification accuracy metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return accuracy_score(labels_np, predictions_np)


class PrecisionMetric(nn.Module):
    """Precision metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return precision_score(labels_np, predictions_np, zero_division=0)


class RecallMetric(nn.Module):
    """Recall metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return recall_score(labels_np, predictions_np, zero_division=0)


class F1Metric(nn.Module):
    """F1-score metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, labels, predictions):
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        return f1_score(labels_np, predictions_np, zero_division=0)


class OptimalThresholdMetric(nn.Module):
    """Optimal threshold finding metric."""
    
    def __init__(self, method="f1", percentile=95):
        super().__init__()
        self.method = method
        self.percentile = percentile

    def forward(self, labels, scores):
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()

        if self.method == "roc":
            fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
            optimal_idx = (tpr - fpr).argmax()
            return thresholds[optimal_idx]

        elif self.method == "f1":
            thresholds = torch.linspace(scores.min(), scores.max(), 100)
            best_f1 = 0
            best_threshold = 0.5

            f1_metric = F1Metric()
            for threshold in thresholds:
                preds = (scores >= threshold).float()
                f1 = f1_metric(labels, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold.item()

            return best_threshold

        elif self.method == "percentile":
            normal_mask = labels_np == 0
            if normal_mask.sum() == 0:
                return 0.5
            normal_scores = scores_np[normal_mask]
            threshold = float(torch.quantile(torch.tensor(normal_scores), self.percentile / 100.0))
            return threshold

        else:
            return 0.5
