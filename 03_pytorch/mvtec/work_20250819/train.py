"""
Training utilities and functions for model training
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as T
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, random_split

import numpy as np
import random
import sys
from tqdm import tqdm
from time import time


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
# Fractor Functions
# =============================================================================

def split_dataset(train_dataset, valid_dataset, valid_ratio=0.2, seed=42):
    valid_size = int(valid_ratio * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed))
    train_dataset = Subset(train_dataset, train_subset.indices)
    valid_dataset = Subset(valid_dataset, valid_subset.indices)
    return train_dataset, valid_dataset


def get_dataloader(loader_type='train', **loader_params):
    """Get dataloader with split-specific default settings"""
    available_types = ['train', 'valid', 'test']

    if loader_type not in available_types:
        raise ValueError(f"Unknown loader type: {loader_type}. Available: {available_types}")

    # Extract required parameters
    dataset = loader_params.pop('dataset')
    batch_size = loader_params.pop('batch_size')
    shuffle = loader_params.pop('shuffle', None)

    # Split-specific default settings
    type_defaults = {
        'train': {'shuffle': True, 'drop_last': True},
        'valid': {'shuffle': False, 'drop_last': False},
        'test': {'shuffle': False, 'drop_last': False}
    }

    defaults = type_defaults[loader_type].copy()

    if shuffle is not None:
        defaults['shuffle'] = shuffle

    final_params = {**defaults, **loader_params}

    if 'num_workers' not in final_params:
        import os
        final_params['num_workers'] = min(4, os.cpu_count() or 1)

    dataloader = DataLoader(dataset, batch_size=batch_size, **final_params)
    print()
    print(f" > Creating dataloader: type={loader_type}, batch_size={batch_size}")
    print(f" > Settings: {final_params}")
    print(f" > Dataset size: {len(dataset)}, Batches: {len(dataloader)}")

    return dataloader


def get_optimizer(optimizer_type='adamw', **optimizer_params):
    """Get optimizer with configurable parameters"""
    available_types = ['adamw', 'adam', 'sgd', 'rmsprop']

    if optimizer_type not in available_types:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {available_types}")

    # Extract required parameters
    model = optimizer_params.pop('model')
    lr = optimizer_params.pop('lr', 1e-3)

    # Optimizer-specific default parameters
    type_defaults = {
        'adamw': {'weight_decay': 1e-5, 'betas': (0.9, 0.999)},
        'adam': {'weight_decay': 1e-5, 'betas': (0.9, 0.999)},
        'sgd': {'momentum': 0.9, 'weight_decay': 1e-4},
        'rmsprop': {'momentum': 0.9, 'weight_decay': 1e-5}
    }

    final_params = {**type_defaults[optimizer_type], **optimizer_params}
    final_params['lr'] = lr

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **final_params)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **final_params)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **final_params)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), **final_params)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f" > Creating optimizer: {optimizer_type}")
    print(f" > Learning rate: {lr}, Trainable parameters: {total_params:,}")
    print(f" > Optimizer settings: {final_params}")

    return optimizer


def get_scheduler(scheduler_type='plateau', **scheduler_params):
    """Get learning rate scheduler with configurable parameters"""
    available_types = ['plateau', 'cosine', 'step', 'exponential', 'none']

    if scheduler_type not in available_types:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {available_types}")

    if scheduler_type == 'none':
        print("No learning rate scheduler will be used")
        return None

    # Extract required parameters
    optimizer = scheduler_params.pop('optimizer')

    # Scheduler-specific default parameters
    type_defaults = {
        'plateau': {'mode': 'min', 'factor': 0.5, 'patience': 5},
        'cosine': {'T_max': 50, 'eta_min': 1e-6},
        'step': {'step_size': 10, 'gamma': 0.1},
        'exponential': {'gamma': 0.95}
    }

    final_params = {**type_defaults[scheduler_type], **scheduler_params}

    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **final_params)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **final_params)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **final_params)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **final_params)

    current_lr = optimizer.param_groups[0]['lr']
    print()
    print(f" > Creating scheduler: {scheduler_type}")
    print(f" > Current learning rate: {current_lr}")
    print(f" > Scheduler settings: {final_params}")

    return scheduler


# =============================================================================
# Train Model for Data Loader
# =============================================================================

class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = next(model.parameters()).device

    def fit(self, train_loader, num_epochs, valid_loader=None, scheduler=None,
            early_stop=False):

        # === Early Stopping variables ===
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            # Training phase
            train_results = train_epoch(self.model, train_loader,
                self.loss_fn, self.optimizer, metrics=self.metrics)
            train_info = ", ".join([f'{key}={value:.3f}'
                                    for key, value in train_results.items()])

            if valid_loader is not None:
                # Validation phase
                valid_results = validate_epoch(self.model, valid_loader,
                    self.loss_fn, metrics=self.metrics)
                valid_info = ", ".join([f'{key}={value:.3f}'
                                        for key, value in valid_results.items()])

                elapsed_time = time() - start_time
                print(f" [{epoch:2d}/{num_epochs}] "
                    f"{train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")

                # Learning rate scheduling based on validation loss
                # old_lr = scheduler.get_last_lr()[0]
                scheduler.step(valid_results["loss"])
                # new_lr = scheduler.get_last_lr()[0]
                # if new_lr != old_lr:
                #     print(f" > Learning rate: {old_lr:.4e} -> {new_lr:.4e}")

                # === Early Stopping check (only if enabled) ===
                if early_stop:
                    if valid_results["loss"] < best_loss:
                        # Validation improved, save best model
                        best_loss = valid_results["loss"]
                        best_model_state = self.model.state_dict()
                        patience_counter = 0
                    else:
                        # No improvement
                        patience_counter += 1
                        print(f" > EarlyStopping counter: {patience_counter}/{patience}")
                        if patience_counter >= patience:
                            print(" > Early stopping triggered!")
                            if best_model_state is not None:
                                self.model.load_state_dict(best_model_state)  # restore best model
                            break
            else:
                # No validation loader (train-only mode)
                elapsed_time = time() - start_time
                print(f" [{epoch:2d}/{num_epochs}] {train_info} "
                    f"({elapsed_time:.1f}s)")
        history = {"loss": []}
        return history


# =============================================================================
# Train and Evaluate the model for One Epoch
# =============================================================================

def get_tqdm_stream():
    """Get the appropriate stream for tqdm output"""
    return getattr(sys, "__stdout__", None) or getattr(sys.stdout, "terminal", None) or sys.stdout


def train_epoch(model, data_loader, loss_fn, optimizer, metrics={}):
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
            loss = loss_fn(reconstructed, normal_images)
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
def validate_epoch(model, data_loader, loss_fn, metrics={}):
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
            loss = loss_fn(reconstructed, normal_images)

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


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    print("Example usage:")
    print("from train import train_model, set_seed")
    print("set_seed(42)")
    print("train_model(model, train_loader, config, valid_loader)")
