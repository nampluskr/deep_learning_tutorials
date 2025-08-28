import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score)


class Evaluator:
    """Evaluator wrapper for anomaly detection models"""
    def __init__(self, modeler):
        self.modeler = modeler

    @torch.no_grad()
    def predict(self, test_loader):
        all_scores, all_labels = [], []

        with tqdm(test_loader, desc="Evaluate", leave=False, ascii=True) as pbar:
            for inputs in pbar:
                scores = self.modeler.predict_step(inputs)
                labels = inputs["label"]

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        return {
            "score": torch.cat(all_scores, dim=0).numpy(),
            "label": torch.cat(all_labels, dim=0).numpy(),
        }


def evaluate_anomaly_detection(scores, labels, threshold):
    """Evaluate anomaly detection performance"""
    try:
        # Convert to numpy arrays
        scores = np.array(scores)
        labels = np.array(labels)

        # Compute AUROC and AUPR
        auroc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)

        # Compute metrics at optimal threshold
        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        results = {
            'auroc': auroc,
            'aupr': aupr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold
        }

        print(f"\n=== Evaluation Results ===")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AUPR: {aupr:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        return results

    except Exception as e:
        print(f"Error evaluating: {e}")
        return {}


def compute_threshold(scores, labels, method="percentile", percentile=95):
    """Find optimal threshold for anomaly detection"""
    if method == "percentile":
        threshold = np.percentile(scores, percentile)

    elif method == "roc":
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

    elif method == "f1":
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_f1 = 0
        best_threshold = thresholds[0]

        for thresh in thresholds:
            pred = (scores > thresh).astype(int)
            f1 = f1_score(labels, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        threshold = best_threshold
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return threshold


if __name__ == "__main__":
    pass
