import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import (roc_auc_score, 
    average_precision_score, f1_score, accuracy_score)
import sys
from tqdm import tqdm
from time import time

from metrics_functional import get_loss_fn, get_metrics, compute_reconstruction_error


def set_seed(seed=42, device='cpu'):
    """Set random seeds for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_to_unit_range(images):
    """Normalize images to [0, 1] range regardless of input range"""
    # Handle batch dimension
    batch_size = images.size(0)
    images_flat = images.view(batch_size, -1)
    
    # Per-sample normalization to [0, 1]
    min_vals = images_flat.min(dim=1, keepdim=True)[0]
    max_vals = images_flat.max(dim=1, keepdim=True)[0]
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals > 1e-8, range_vals, torch.ones_like(range_vals))
    
    normalized = (images_flat - min_vals) / range_vals
    return normalized.view_as(images)


def denormalize_from_imagenet(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize images from ImageNet normalization back to [0, 1] range"""
    device = images.device
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    # Denormalize: x = x_norm * std + mean
    denorm_images = images * std + mean

    # Clamp to [0, 1] range to ensure valid pixel values
    return torch.clamp(denorm_images, 0.0, 1.0)


def prepare_images_for_loss(images, model_output, normalize_input=True):
    """Prepare images for consistent loss calculation"""
    # Model output is always in [0, 1] range due to sigmoid
    # Need to ensure target images are also in [0, 1] range
    
    if normalize_input:
        # Check if images are ImageNet normalized (contain negative values)
        if images.min() < 0:
            # Images are ImageNet normalized, denormalize them
            target_images = denormalize_from_imagenet(images)
        else:
            # Images might be in [0, 1] range already or need normalization
            if images.max() > 1.0:
                # Images are in [0, 255] range, normalize to [0, 1]
                target_images = images / 255.0
            else:
                # Images are already in [0, 1] range
                target_images = images
    else:
        # No normalization, use as is but ensure [0, 1] range
        target_images = torch.clamp(images, 0.0, 1.0)
    
    return target_images


def train_model(model, train_loader, config, valid_loader=None):
    """Main training loop for autoencoder model"""
    loss_fn = get_loss_fn(config.loss_type)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    metrics = get_metrics(['ssim', 'psnr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    best_loss = float('inf')
    for epoch in range(1, config.num_epochs + 1):
        start_time = time()
        
        # Training phase
        train_results = train_epoch(
            model, train_loader, loss_fn, optimizer, 
            metrics=metrics, normalize_input=config.normalize
        )
        train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

        if valid_loader is not None:
            # Validation phase
            valid_results = evaluate_epoch(
                model, valid_loader, loss_fn, 
                metrics=metrics, normalize_input=config.normalize
            )
            valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

            elapsed_time = time() - start_time
            print(f" [{epoch:2d}/{config.num_epochs}] "
                  f"{train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")

            # Update learning rate based on validation loss
            old_lr = scheduler.get_last_lr()[0]
            scheduler.step(valid_results["loss"])
            new_lr = scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                print(f" > Learning rate: {old_lr:.4e} -> {new_lr:.4e}")

            # Track best model
            if valid_results["loss"] < best_loss:
                best_loss = valid_results["loss"]
        else:
            elapsed_time = time() - start_time
            print(f" [{epoch:2d}/{config.num_epochs}] {train_info} "
                  f"({elapsed_time:.1f}s)")


def train_epoch(model, data_loader, loss_fn, optimizer, metrics={}, normalize_input=True):
    """Train model for one epoch using only normal data"""
    device = next(model.parameters()).device
    model.train()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    num_valid_batches = 0
    
    with tqdm(data_loader, desc="Train", file=sys.stdout, ascii=True,
              dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Use only normal data (labels == 0) for training
            normal_mask = (labels == 0)
            if not normal_mask.any():
                num_batches += 1
                continue

            normal_images = images[normal_mask]

            optimizer.zero_grad()
            reconstructed, latent, features = model(normal_images)

            # Prepare target images for consistent loss calculation
            target_images = prepare_images_for_loss(
                normal_images, reconstructed, normalize_input=normalize_input
            )

            loss = loss_fn(reconstructed, target_images)
            loss.backward()
            optimizer.step()

            # Update metrics
            results["loss"] += loss.item()
            with torch.no_grad():
                for metric_name, metric_fn in metrics.items():
                    try:
                        metric_value = metric_fn(reconstructed, target_images)
                        results[metric_name] += metric_value.item()
                    except Exception as e:
                        print(f"Warning: Error computing {metric_name}: {e}")
                        results[metric_name] += 0.0

            num_batches += 1
            num_valid_batches += 1
            
            if num_valid_batches > 0:
                progress_info = {f'{key}': f'{value/num_valid_batches:.3f}'
                                for key, value in results.items()}
                progress_bar.set_postfix(progress_info)

    # Average over valid batches only
    if num_valid_batches > 0:
        return {key: value / num_valid_batches for key, value in results.items()}
    else:
        return {key: 0.0 for key in results.keys()}


@torch.no_grad()
def evaluate_epoch(model, data_loader, loss_fn, metrics={}, normalize_input=True):
    """Evaluate model for one epoch using only normal data"""
    device = next(model.parameters()).device
    model.eval()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    num_valid_batches = 0
    
    with tqdm(data_loader, desc="Evaluation", file=sys.stdout, ascii=True,
              dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Use only normal data (labels == 0) for validation loss
            normal_mask = (labels == 0)
            if not normal_mask.any():
                num_batches += 1
                continue

            normal_images = images[normal_mask]
            reconstructed, latent, features = model(normal_images)

            # Prepare target images for consistent loss calculation
            target_images = prepare_images_for_loss(
                normal_images, reconstructed, normalize_input=normalize_input
            )

            loss = loss_fn(reconstructed, target_images)

            results["loss"] += loss.item()
            for metric_name, metric_fn in metrics.items():
                try:
                    metric_value = metric_fn(reconstructed, target_images)
                    results[metric_name] += metric_value.item()
                except Exception as e:
                    print(f"Warning: Error computing {metric_name} in evaluation: {e}")
                    results[metric_name] += 0.0

            num_batches += 1
            num_valid_batches += 1
            
            if num_valid_batches > 0:
                progress_info = {f'{key}': f'{value/num_valid_batches:.3f}'
                                for key, value in results.items()}
                progress_bar.set_postfix(progress_info)

    # Average over valid batches only
    if num_valid_batches > 0:
        return {key: value / num_valid_batches for key, value in results.items()}
    else:
        return {key: 0.0 for key in results.keys()}


@torch.no_grad()
def compute_anomaly_scores(model, data_loader, method='mse', normalize_input=True):
    """Compute anomaly scores for all samples in the data loader"""
    device = next(model.parameters()).device
    model.eval()

    anomaly_scores = []
    labels = []
    defect_types = []

    with tqdm(data_loader, desc="Computing anomaly scores", file=sys.stdout, ascii=True,
              dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            batch_labels = batch['label'].cpu().numpy()
            batch_defect_types = batch['defect_type']

            reconstructed, _, _ = model(images)

            # Prepare target images for consistent comparison
            target_images = prepare_images_for_loss(
                images, reconstructed, normalize_input=normalize_input
            )

            # Compute per-sample reconstruction error using specified method
            batch_scores = compute_reconstruction_error(
                reconstructed, target_images, method=method, reduction='none'
            )

            anomaly_scores.extend(batch_scores.cpu().numpy())
            labels.extend(batch_labels)
            defect_types.extend(batch_defect_types)

    return np.array(anomaly_scores), np.array(labels), defect_types


def evaluate_anomaly_detection(model, test_loader, method='mse', percentile=95, normalize_input=True):
    """Evaluate anomaly detection performance on test data"""
    # Compute anomaly scores and threshold
    scores, labels, defect_types = compute_anomaly_scores(
        model, test_loader, method, normalize_input=normalize_input
    )
    
    if len(scores) == 0:
        print("Warning: No samples found for anomaly detection evaluation")
        return {
            'auroc': 0.0, 'aupr': 0.0, 'f1_score': 0.0, 'accuracy': 0.0,
            'threshold': 0.0, 'percentile': percentile, 'method': method,
            'total_samples': 0, 'normal_samples': 0, 'anomaly_samples': 0,
            'defect_types': []
        }
    
    # Compute threshold based on normal samples only
    normal_scores = scores[labels == 0]
    if len(normal_scores) > 0:
        threshold = np.percentile(normal_scores, percentile)
    else:
        threshold = np.percentile(scores, percentile)
    
    predictions = (scores > threshold).astype(int)

    # Compute metrics
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError as e:
        print(f"Warning: Cannot compute AUROC: {e}")
        auroc = 0.0

    try:
        aupr = average_precision_score(labels, scores)
    except ValueError as e:
        print(f"Warning: Cannot compute AUPR: {e}")
        aupr = 0.0

    try:
        f1 = f1_score(labels, predictions)
    except ValueError as e:
        print(f"Warning: Cannot compute F1 score: {e}")
        f1 = 0.0

    try:
        acc = accuracy_score(labels, predictions)
    except ValueError as e:
        print(f"Warning: Cannot compute accuracy: {e}")
        acc = 0.0

    return {
        'auroc': auroc,
        'aupr': aupr,
        'f1_score': f1,
        'accuracy': acc,
        'threshold': threshold,
        'percentile': percentile,
        'method': method,
        'total_samples': len(scores),
        'normal_samples': np.sum(labels == 0),
        'anomaly_samples': np.sum(labels == 1),
        'defect_types': list(set(defect_types))
    }


def compute_detailed_metrics(model, test_loader, methods=['mse', 'l1', 'ssim'], 
                           percentiles=[90, 95, 99], normalize_input=True):
    """Compute detailed metrics across different methods and thresholds"""
    results = {}
    
    for method in methods:
        print(f"\n > Evaluating with method: {method}")
        method_results = {}
        
        # Compute scores once per method
        scores, labels, defect_types = compute_anomaly_scores(
            model, test_loader, method, normalize_input=normalize_input
        )
        
        for percentile in percentiles:
            metrics = evaluate_anomaly_detection_from_scores(
                scores, labels, defect_types, method, percentile
            )
            method_results[f"p{percentile}"] = metrics
        
        results[method] = method_results
    
    return results


def evaluate_anomaly_detection_from_scores(scores, labels, defect_types, method, percentile):
    """Evaluate anomaly detection from pre-computed scores"""
    if len(scores) == 0:
        return {
            'auroc': 0.0, 'aupr': 0.0, 'f1_score': 0.0, 'accuracy': 0.0,
            'threshold': 0.0, 'percentile': percentile, 'method': method,
            'total_samples': 0, 'normal_samples': 0, 'anomaly_samples': 0,
            'defect_types': []
        }
    
    # Compute threshold based on normal samples only
    normal_scores = scores[labels == 0]
    if len(normal_scores) > 0:
        threshold = np.percentile(normal_scores, percentile)
    else:
        threshold = np.percentile(scores, percentile)
    
    predictions = (scores > threshold).astype(int)

    # Compute metrics with error handling
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.0

    try:
        aupr = average_precision_score(labels, scores)
    except ValueError:
        aupr = 0.0

    try:
        f1 = f1_score(labels, predictions)
    except ValueError:
        f1 = 0.0

    try:
        acc = accuracy_score(labels, predictions)
    except ValueError:
        acc = 0.0

    return {
        'auroc': auroc,
        'aupr': aupr,
        'f1_score': f1,
        'accuracy': acc,
        'threshold': threshold,
        'percentile': percentile,
        'method': method,
        'total_samples': len(scores),
        'normal_samples': np.sum(labels == 0),
        'anomaly_samples': np.sum(labels == 1),
        'defect_types': list(set(defect_types))
    }


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    print("Example usage:")
    print("from train import train_model, set_seed")
    print("set_seed(42)")
    print("train_model(model, train_loader, config, valid_loader)")