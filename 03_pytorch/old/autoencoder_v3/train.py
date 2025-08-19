import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import sys
from tqdm import tqdm
import time
from copy import deepcopy


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


def denormalize_images(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize images from ImageNet normalization back to [0, 1] range"""
    device = images.device
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)
    
    # Denormalize: x = x_norm * std + mean
    denorm_images = images * std + mean
    
    # Clamp to [0, 1] range to ensure valid pixel values
    return torch.clamp(denorm_images, 0.0, 1.0)


def train_model(model, train_loader, config, valid_loader=None):
    """Main training loop for autoencoder model"""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Additional metrics for monitoring
    metrics = {
        "l1": nn.L1Loss(),
        # Removed BCELoss due to normalization compatibility issues
    }
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        # Training phase
        train_results = train_epoch(model, train_loader, criterion, optimizer, metrics=metrics)
        train_info = ", ".join([f'{key}={value:.4f}' for key, value in train_results.items()])

        if valid_loader is not None:
            # Validation phase
            valid_results = evaluate_epoch(model, valid_loader, criterion, metrics=metrics)
            valid_info = ", ".join([f'{key}={value:.4f}' for key, value in valid_results.items()])

            print(f" [{epoch:2d}/{config.num_epochs}] "
                  f"{train_info} | (val) {valid_info}")

            # Update learning rate based on validation loss
            scheduler.step(valid_results["loss"])
            
            # Track best model
            if valid_results["loss"] < best_loss:
                best_loss = valid_results["loss"]
        else:
            print(f" [{epoch:2d}/{config.num_epochs}] {train_info}")


def train_epoch(model, data_loader, criterion, optimizer, metrics={}):
    """Train model for one epoch"""
    device = next(model.parameters()).device
    model.train()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    with tqdm(data_loader, desc="Train", file=sys.stdout, ascii=True,
              dynamic_ncols=True, leave=False) as progress_bar:
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

            # Handle normalization: if images are normalized (contain negative values), denormalize for loss calculation
            if normal_images.min() < 0:
                # Images are normalized, denormalize for consistent comparison with sigmoid output
                target_images = denormalize_images(normal_images)
            else:
                # Images are already in [0, 1] range
                target_images = normal_images

            loss = criterion(reconstructed, target_images)
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
            progress_info = {f'{key}': f'{value/num_batches:.4f}'
                             for key, value in results.items()}
            progress_bar.set_postfix(progress_info)

    return {key: value / len(data_loader) for key, value in results.items()}


@torch.no_grad()
def evaluate_epoch(model, data_loader, criterion, metrics={}):
    """Evaluate model for one epoch"""
    device = next(model.parameters()).device
    model.eval()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    with tqdm(data_loader, desc="Evaluation", file=sys.stdout, ascii=True,
              dynamic_ncols=True, leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Use only normal data (labels == 0) for evaluation
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]
            reconstructed, latent, features = model(normal_images)

            # Handle normalization: if images are normalized (contain negative values), denormalize for loss calculation
            if normal_images.min() < 0:
                # Images are normalized, denormalize for consistent comparison with sigmoid output
                target_images = denormalize_images(normal_images)
            else:
                # Images are already in [0, 1] range
                target_images = normal_images

            loss = criterion(reconstructed, target_images)

            results["loss"] += loss.item()
            for metric_name, metric_fn in metrics.items():
                try:
                    metric_value = metric_fn(reconstructed, target_images)
                    results[metric_name] += metric_value.item()
                except Exception as e:
                    print(f"Warning: Error computing {metric_name} in evaluation: {e}")
                    results[metric_name] += 0.0

            num_batches += 1
            progress_info = {f'{key}': f'{value/num_batches:.4f}'
                             for key, value in results.items()}
            progress_bar.set_postfix(progress_info)

    return {key: value / len(data_loader) for key, value in results.items()}


@torch.no_grad()
def compute_anomaly_scores(model, data_loader, criterion):
    """Compute anomaly scores for all samples in the data loader"""
    device = next(model.parameters()).device
    model.eval()
    
    anomaly_scores = []
    labels = []
    defect_types = []
    
    with tqdm(data_loader, desc="Computing anomaly scores", ascii=True) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            batch_labels = batch['label'].cpu().numpy()
            batch_defect_types = batch['defect_type']
            
            reconstructed, _, _ = model(images)
            
            # Handle normalization for consistent comparison
            if images.min() < 0:
                # Images are normalized, denormalize for consistent comparison
                target_images = denormalize_images(images)
            else:
                # Images are already in [0, 1] range
                target_images = images
            
            # Compute per-sample reconstruction error
            batch_scores = torch.mean((target_images - reconstructed) ** 2, dim=[1, 2, 3])
            
            anomaly_scores.extend(batch_scores.cpu().numpy())
            labels.extend(batch_labels)
            defect_types.extend(batch_defect_types)
    
    return np.array(anomaly_scores), np.array(labels), defect_types


class TrainingMetrics:
    """Helper class to track and compute training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.l1_losses = []
    
    def update(self, loss, l1_loss=None):
        """Update metrics with new values"""
        self.losses.append(loss)
        if l1_loss is not None:
            self.l1_losses.append(l1_loss)
    
    def get_averages(self):
        """Get average values of all metrics"""
        return {
            'loss': np.mean(self.losses) if self.losses else 0.0,
            'l1': np.mean(self.l1_losses) if self.l1_losses else 0.0
        }


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    print("Example usage:")
    print("from train import train_model, set_seed")
    print("set_seed(42)")
    print("train_model(model, train_loader, config, valid_loader)")