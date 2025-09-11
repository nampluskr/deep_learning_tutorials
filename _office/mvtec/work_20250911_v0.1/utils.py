import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, confusion_matrix,
)
import torch


def show_history(history, save_path=None):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, len(history), figsize=(4*len(history), 3))
    if len(history) == 1:
        axes = [axes]

    for idx, (model_name, losses) in enumerate(history.items()):
        ax = axes[idx]
        ax.plot(losses, color='blue', linewidth=2)

        ax.set_title(f"{model_name} Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to: {save_path}")

    plt.show()


def show_roc_curve(scores, labels, title="ROC Curve"):
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    
    if fpr is None or tpr is None:
        print("Cannot plot ROC curve: insufficient data")
        return

    plt.figure(figsize=(5, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


def show_evaluation(scores, labels):
    results = {}
    results['auroc'] = roc_auc_score(labels, scores)
    results['aupr'] = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    results["threshold"] = best_threshold

    predictions = (scores >= best_threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    results['tn'] = tn
    results['fp'] = fp
    results['fn'] = fn
    results['tp'] = tp

    results["accuracy"] = accuracy_score(labels, predictions)
    results["precision"] = precision_score(labels, predictions)
    results["recall"] = recall_score(labels, predictions)
    results["f1"] = f1_score(labels, predictions)

    result_lines = [
        "-" * 60,
        f"EXPERIMENT RESULTS",
        "-" * 60,
        f" > AUROC:             {results['auroc']:.4f}",
        f" > AUPR:              {results['aupr']:.4f}",
        f" > Threshold:         {results['threshold']:.3e}",
         "-" * 60,
        f" > Accuracy:          {results['accuracy']:.4f}",
        f" > Precision:         {results['precision']:.4f}",
        f" > Recall:            {results['recall']:.4f}",
        f" > F1-Score:          {results['f1']:.4f}",
        "-" * 60,
        f"                   Predicted",
        f"   Actual    Normal  Anomaly",
        f"   Normal    {results['tn']:6d}  {results['fp']:7d}",
        f"   Anomaly   {results['fn']:6d}  {results['tp']:7d}",
    ]
    for line in result_lines:
        print(line)

def show_statistics(scores, labels):
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    print(f"\nDETAILED SCORE STATISTICS:")
    print("-"*40)
    if len(normal_scores) > 0:
        print(f"Normal  - Count: {len(normal_scores):4d}")
        print(f"          Mean:  {normal_scores.mean():.6f} ± {normal_scores.std():.6f}")
        print(f"          Range: [{normal_scores.min():.6f}, {normal_scores.max():.6f}]")

    if len(anomaly_scores) > 0:
        print(f"\nAnomaly - Count: {len(anomaly_scores):4d}")
        print(f"          Mean:  {anomaly_scores.mean():.6f} ± {anomaly_scores.std():.6f}")
        print(f"          Range: [{anomaly_scores.min():.6f}, {anomaly_scores.max():.6f}]")

    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        pooled_std = np.sqrt(((len(normal_scores) - 1) * normal_scores.var() +
                             (len(anomaly_scores) - 1) * anomaly_scores.var()) /
                             (len(normal_scores) + len(anomaly_scores) - 2))
        cohens_d = (anomaly_scores.mean() - normal_scores.mean()) / pooled_std

        print(f"\nSEPARATION ANALYSIS:")
        print("-"*40)
        print(f"Cohen's d (effect size): {cohens_d:.4f}")

        if cohens_d > 0.8:   print(" > EXCELLENT separation (Cohen's d > 0.8)")
        elif cohens_d > 0.5: print(" > GOOD separation (Cohen's d > 0.5)")
        elif cohens_d > 0.2: print(" > FAIR separation (Cohen's d > 0.2)")
        else:                print(" > POOR separation (Cohen's d ≤ 0.2)")


def show_distribution(scores, labels, bins=50, save_path=None):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    # Overall distribution
    sns.histplot(scores, bins=bins, kde=True, color="steelblue", edgecolor="black", ax=ax1)
    ax1.set_title("Anomaly Score Distribution - Overall")
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    # Separated by class
    if len(anomaly_scores) > 0:
        sns.histplot(anomaly_scores, bins=bins, kde=True, color="red",
            label=f"Anomaly (n={len(anomaly_scores)})", alpha=0.6,
            edgecolor="black", ax=ax2)

    if len(normal_scores) > 0:
        sns.histplot(normal_scores, bins=bins, kde=True, color="green",
            label=f"Normal (n={len(normal_scores)})", alpha=0.6, 
            edgecolor="black", ax=ax2)

    ax2.set_title("Anomaly Score Distribution - By Class")
    ax2.set_xlabel("Anomaly Score")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to: {save_path}")

    plt.show()
