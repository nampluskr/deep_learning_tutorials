"""
Evaluation utilities for anomaly detection performance assessment
Contains functions for anomaly score computation and performance metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
from pytorch_msssim import ssim


def load_trained_model(model_path, device='cuda'):
    """Load trained model from checkpoint"""
    from models import get_model

    checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    config = checkpoint.get('config', None)
    model_type = checkpoint.get('model_type', 'vanilla_ae')

    if config:
        # Use config from checkpoint
        model = get_model(
            model_type,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            latent_dim=config.latent_dim
        )
    else:
        # Use default parameters if config not available
        model = get_model(model_type)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    print(f"Model type: {model_type}")

    return model, config

@torch.no_grad()
def compute_anomaly_scores(model, data_loader, method="mse"):
    """Compute anomaly scores for given dataset"""
    device = next(model.parameters()).device

    model.eval()
    all_scores = []
    all_labels = []
    all_images = []
    all_preds = []

    for batch in data_loader:
        images = batch['image'].to(device)
        labels = batch['label']
        reconstructions, _, _ = model(images)

        if method == "mse":
            scores = torch.mean((images - reconstructions)**2, dim=[1,2,3])
        elif method == "ssim":
            scores = 1 - ssim(images, reconstructions, size_average=False)
        elif method == "l1":
            scores = torch.mean(torch.abs(images - reconstructions), dim=[1,2,3])
        elif method == "combined":
            mse_scores = torch.mean((images - reconstructions)**2, dim=[1,2,3])
            ssim_scores = 1 - ssim(images, reconstructions, size_average=False)
            scores = 0.5 * mse_scores + 0.5 * ssim_scores
        else:
            raise ValueError(f"Unknown method: {method}")

        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(images.cpu().numpy())
        all_preds.extend(reconstructions.cpu().numpy())

    return {
        'scores': np.array(all_scores),
        'labels': np.array(all_labels),
        'images': np.array(all_images),
        'reconstructions': np.array(all_preds)
    }


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


def evaluate_anomaly_detection(model, test_loader, method="mse", threshold_method="percentile", percentile=95):
    """Comprehensive anomaly detection evaluation"""
    device = next(model.parameters()).device

    # Compute anomaly scores
    results = compute_anomaly_scores(model, test_loader, method)
    scores = results['scores']
    labels = results['labels']

    # Make predictions
    threshold = compute_threshold(scores, labels, threshold_method, percentile)
    preds = (scores > threshold).astype(int)

    # Compute metrics
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'auroc': auroc,
        'aupr': aupr,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'threshold': threshold,
        'confusion_matrix': cm,
        'scores': scores,
        'labels': labels,
        'preds': preds,
        'images': results['images'],
        'reconstructions': results['reconstructions'],
        'method': method,
        'threshold_method': threshold_method
    }


def evaluate_model(model, test_loader, method="mse", percentile=95):
    """Evaluate model performance on test dataset (simplified version)"""
    return evaluate_anomaly_detection(
        model, test_loader, method=method,
        threshold_method="percentile", percentile=percentile
    )


def show_results(results):
    """Display evaluation results"""
    print("\n" + "="*50)
    print("ANOMALY DETECTION EVALUATION RESULTS")
    print("="*50)

    print(f"Method: {results['method'].upper()}")
    print(f"Threshold: {results['threshold']:.6f}")
    print(f"Threshold Method: {results['threshold_method']}")

    print("\n" + "-"*30)
    print("PERFORMANCE METRICS")
    print("-"*30)
    print(f"AUROC:       {results['auroc']:.4f}")
    print(f"AUPR:        {results['aupr']:.4f}")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"F1 Score:    {results['f1_score']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")

    print("\n" + "-"*30)
    print("CONFUSION MATRIX")
    print("-"*30)
    cm = results['confusion_matrix']
    print(f"TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    print("\n" + "="*50)


def save_results(results, save_path):
    """Save evaluation results to file"""
    # Prepare results for JSON serialization
    json_results = {
        'auroc': float(results['auroc']),
        'aupr': float(results['aupr']),
        'accuracy': float(results['accuracy']),
        'f1_score': float(results['f1_score']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'specificity': float(results['specificity']),
        'threshold': float(results['threshold']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'method': results['method'],
        'threshold_method': results['threshold_method'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=4)

    print(f"Results saved to: {save_path}")


def plot_roc_curve(y_true, y_scores, save_path=None, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    plt.show()


def plot_precision_recall_curve(y_true, y_scores, save_path=None, title="Precision-Recall Curve"):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUPR = {aupr:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")

    plt.show()


def plot_anomaly_score_distribution(scores, labels, save_path=None, title="Anomaly Score Distribution"):
    """Plot distribution of anomaly scores for normal and anomalous samples"""
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)

    # Add statistics
    plt.axvline(normal_scores.mean(), color='blue', linestyle='--', alpha=0.8,
                label=f'Normal Mean: {normal_scores.mean():.4f}')
    plt.axvline(anomaly_scores.mean(), color='red', linestyle='--', alpha=0.8,
                label=f'Anomaly Mean: {anomaly_scores.mean():.4f}')

    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(cm, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


def plot_reconstruction_samples(images, reconstructions, scores, labels,
                               num_samples=8, save_path=None):
    """Plot original vs reconstructed images with anomaly scores"""
    # Select samples
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]

    # Get samples
    normal_samples = min(num_samples // 2, len(normal_indices))
    anomaly_samples = min(num_samples // 2, len(anomaly_indices))

    selected_indices = []
    if normal_samples > 0:
        selected_indices.extend(normal_indices[:normal_samples])
    if anomaly_samples > 0:
        selected_indices.extend(anomaly_indices[:anomaly_samples])

    fig, axes = plt.subplots(3, len(selected_indices), figsize=(2*len(selected_indices), 6))
    if len(selected_indices) == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(selected_indices):
        # Original image
        img = images[idx].transpose(1, 2, 0)
        if img.shape[2] == 3:
            img = np.clip(img, 0, 1)
        else:
            img = img[:, :, 0]
        axes[0, i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[0, i].set_title(f'Original\n{"Anomaly" if labels[idx] == 1 else "Normal"}')
        axes[0, i].axis('off')

        # Reconstructed image
        recon = reconstructions[idx].transpose(1, 2, 0)
        if recon.shape[2] == 3:
            recon = np.clip(recon, 0, 1)
        else:
            recon = recon[:, :, 0]
        axes[1, i].imshow(recon, cmap='gray' if len(recon.shape) == 2 else None)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

        # Difference
        diff = np.abs(img - recon)
        if len(diff.shape) == 3 and diff.shape[2] == 3:
            diff = np.mean(diff, axis=2)
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'Diff\nScore: {scores[idx]:.4f}')
        axes[2, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction samples saved to: {save_path}")

    plt.show()


def create_evaluation_report(results, save_dir=None, model_name="model"):
    """Create comprehensive evaluation report with all plots"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("Creating evaluation report...")

    # Extract data
    scores = results['scores']
    labels = results['labels']
    images = results['images']
    reconstructions = results['reconstructions']
    cm = results['confusion_matrix']

    # Plot ROC curve
    roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png") if save_dir else None
    plot_roc_curve(labels, scores, roc_path)

    # Plot PR curve
    pr_path = os.path.join(save_dir, f"{model_name}_pr_curve.png") if save_dir else None
    plot_precision_recall_curve(labels, scores, pr_path)

    # Plot score distribution
    dist_path = os.path.join(save_dir, f"{model_name}_score_distribution.png") if save_dir else None
    plot_anomaly_score_distribution(scores, labels, dist_path)

    # Plot confusion matrix
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png") if save_dir else None
    plot_confusion_matrix(cm, cm_path)

    # Plot reconstruction samples
    recon_path = os.path.join(save_dir, f"{model_name}_reconstruction_samples.png") if save_dir else None
    plot_reconstruction_samples(images, reconstructions, scores, labels, save_path=recon_path)

    # Save results
    if save_dir:
        results_path = os.path.join(save_dir, f"{model_name}_results.json")
        save_results(results, results_path)

    print("Evaluation report created successfully!")


# Jupyter notebook helper functions
def quick_evaluate(model_path, test_loader, method="mse", device='cuda'):
    """Quick evaluation function for Jupyter notebooks"""
    # Load model
    model, config = load_trained_model(model_path, device)

    # Evaluate
    results = evaluate_anomaly_detection(model, test_loader, config, method=method)

    # Show results
    show_results(results)

    # Create plots
    create_evaluation_report(results, model_name=f"quick_eval_{method}")

    return results


def compare_methods(model_path, test_loader, methods=["mse", "ssim", "combined"], device='cuda'):
    """Compare different anomaly detection methods"""
    # Load model
    model, config = load_trained_model(model_path, device)

    results_dict = {}

    print("Comparing anomaly detection methods...")
    for method in methods:
        print(f"\nEvaluating method: {method}")
        results = evaluate_anomaly_detection(model, test_loader, config, method=method)
        results_dict[method] = results

    # Create comparison table
    comparison_df = pd.DataFrame({
        method: {
            'AUROC': results['auroc'],
            'AUPR': results['aupr'],
            'Accuracy': results['accuracy'],
            'F1-Score': results['f1_score'],
            'Precision': results['precision'],
            'Recall': results['recall']
        }
        for method, results in results_dict.items()
    })

    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    print(comparison_df.round(4))

    return results_dict, comparison_df
