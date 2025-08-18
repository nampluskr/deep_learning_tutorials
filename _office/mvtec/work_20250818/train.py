import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import (roc_auc_score, 
    average_precision_score, f1_score, accuracy_score)

from dataclasses import fields, asdict
import json
import os
import sys
from tqdm import tqdm
from time import time

# from metrics import get_criterion, get_metrics, compute_reconstruction_error
from metrics import get_loss_fn, get_metrics, compute_reconstruction_error
from config import Config, get_config_prefix


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


# =============================================================================
# Train Model for Data Loader
# =============================================================================

def train_model(model, train_loader, config, valid_loader=None):
    """Main training loop for autoencoder model"""
    criterion = get_loss_fn(loss_type=config.loss_type)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    metrics = get_metrics(metric_names=['psnr', 'ssim'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # === Early Stopping variables ===
    best_loss = float('inf')
    patience = config.early_stopping_patience
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, config.num_epochs + 1):
        start_time = time()
        # Training phase
        train_results = train_epoch(model, train_loader, criterion, optimizer, metrics=metrics)
        train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

        if valid_loader is not None:
            # Validation phase
            valid_results = validate_epoch(model, valid_loader, criterion, metrics=metrics)
            valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

            elapsed_time = time() - start_time
            print(f" [{epoch:2d}/{config.num_epochs}] "
                  f"{train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")

            # Learning rate scheduling based on validation loss
            # old_lr = scheduler.get_last_lr()[0]
            scheduler.step(valid_results["loss"])
            # new_lr = scheduler.get_last_lr()[0]
            # if new_lr != old_lr:
            #     print(f" > Learning rate: {old_lr:.4e} -> {new_lr:.4e}")

            # === Early Stopping check (only if enabled) ===
            if config.early_stopping:
                if valid_results["loss"] < best_loss:
                    # Validation improved, save best model
                    best_loss = valid_results["loss"]
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    # No improvement
                    patience_counter += 1
                    print(f" > EarlyStopping counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(" > Early stopping triggered!")
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)  # restore best model
                        break
        else:
            # No validation loader (train-only mode)
            elapsed_time = time() - start_time
            print(f" [{epoch:2d}/{config.num_epochs}] {train_info} "
                  f"({elapsed_time:.1f}s)")


# =============================================================================
# Train and Evaluate the model for One Epoch
# =============================================================================

def get_tqdm_stream():
    """Get the appropriate stream for tqdm output"""
    return getattr(sys, "__stdout__", None) or getattr(sys.stdout, "terminal", None) or sys.stdout


def train_epoch(model, data_loader, criterion, optimizer, metrics={}):
    """Train model for one epoch"""
    device = next(model.parameters()).device
    model.train()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    with tqdm(data_loader, desc="Train", file=get_tqdm_stream(), 
              ascii=True, dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Use only normal data (labels == 0) for training
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]

            optimizer.zero_grad()
            reconstructed, latent, features = model(normal_images)
            loss = criterion(reconstructed, normal_images)
            loss.backward()
            optimizer.step()

            # Update metrics
            results["loss"] += loss.item()
            with torch.no_grad():
                for metric_name, metric_fn in metrics.items():
                    try:
                        metric_value = metric_fn(reconstructed, normal_images)
                        results[metric_name] += metric_value.item()
                    except Exception as e:
                        print(f"Warning: Error computing {metric_name}: {e}")
                        results[metric_name] += 0.0

            num_batches += 1
            progress_info = {f'{key}': f'{value/num_batches:.3f}'
                             for key, value in results.items()}
            progress_bar.set_postfix(progress_info)

    return {key: value / len(data_loader) for key, value in results.items()}


@torch.no_grad()
def validate_epoch(model, data_loader, criterion, metrics={}):
    """Evaluate model for one epoch"""
    device = next(model.parameters()).device
    model.eval()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    with tqdm(data_loader, desc="Evaluation", file=get_tqdm_stream(), 
              ascii=True, dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Use only normal data (labels == 0) for evaluation
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]
            reconstructed, latent, features = model(normal_images)
            loss = criterion(reconstructed, normal_images)

            results["loss"] += loss.item()
            for metric_name, metric_fn in metrics.items():
                try:
                    metric_value = metric_fn(reconstructed, normal_images)
                    results[metric_name] += metric_value.item()
                except Exception as e:
                    print(f"Warning: Error computing {metric_name} in evaluation: {e}")
                    results[metric_name] += 0.0

            num_batches += 1
            progress_info = {f'{key}': f'{value/num_batches:.3f}'
                             for key, value in results.items()}
            progress_bar.set_postfix(progress_info)

    return {key: value / len(data_loader) for key, value in results.items()}


# =============================================================================
# Save log, model and configuration
# =============================================================================

class Logger:
    """Redirect stdout to both console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_log(config):
    """Save configuration settings to a log file"""
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    prefix = get_config_prefix(config)
    save_dir = os.path.join(results_dir, prefix)
    os.makedirs(save_dir, exist_ok=True)

    log_filename = prefix + "_log.txt"
    log_path = os.path.join(save_dir, log_filename)

    sys.stdout = Logger(log_path)
    print(f" > Log saved to ./results/.../{log_filename}")


def save_model(model, config):
    """Save model and configuration to disk"""
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    prefix = get_config_prefix(config)
    save_dir = os.path.join(results_dir, prefix)
    os.makedirs(save_dir, exist_ok=True)

    # Save model state dictionary
    model_filename = prefix + "_model.pth"
    model_path = os.path.join(save_dir, model_filename)
    config.model_path = model_path
    print(f" > Model saved to ./results/.../{model_filename}")
    torch.save(model.state_dict(), model_path)

    # Save configuration
    config_filename = prefix + "_config.json"
    config_path = os.path.join(save_dir, config_filename)
    config.config_path = config_path
    print(f" > Config saved to ./results/.../{config_filename}")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)


def load_weights(model, model_path):
    """Load the model state"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path))
    return model


# =============================================================================
# Anomaly Detection Evaluation
# =============================================================================

@torch.no_grad()
def compute_anomaly_scores(model, data_loader, method='mse'):
    """Compute anomaly scores for all samples in the data loader"""
    device = next(model.parameters()).device
    model.eval()

    anomaly_scores = []
    labels = []
    defect_types = []

    with tqdm(data_loader, desc="Computing anomaly scores", file=get_tqdm_stream(), 
              ascii=True, dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            batch_images = batch['image'].to(device)
            batch_labels = batch['label'].cpu().numpy()
            batch_defect_types = batch['defect_type']

            reconstructed, _, _ = model(batch_images)
            batch_scores = compute_reconstruction_error(reconstructed, batch_images, method=method)

            anomaly_scores.extend(batch_scores.cpu().numpy())
            labels.extend(batch_labels)
            defect_types.extend(batch_defect_types)

    return np.array(anomaly_scores), np.array(labels), defect_types


def evaluate_anomaly_detection(model, test_loader, method='mse', percentile=95):
    """Evaluate anomaly detection performance"""
    # Compute anomaly scores and threshold
    scores, labels, defect_types = compute_anomaly_scores(model, test_loader, method)
    threshold = np.percentile(scores, percentile)
    predictions = (scores > threshold).astype(int)

    # Compute metrics
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
