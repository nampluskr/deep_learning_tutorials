"""
Training utilities and functions for model training
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
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

def get_transforms(img_size=256):
    """Get training and testing transforms for data augmentation"""

    train_transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
        T.ToTensor(),
    ]
    test_transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ]
    return T.Compose(train_transforms), T.Compose(test_transforms)


def split_dataset(train_dataset, valid_dataset, valid_ratio=0.2, seed=42):
    valid_size = int(valid_ratio * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed))
    train_dataset = Subset(train_dataset, train_subset.indices)
    valid_dataset = Subset(valid_dataset, valid_subset.indices)
    return train_dataset, valid_dataset


def get_dataloader(dataset, batch_size, split, **loader_params):
    if split == "train":
        dataloader = DataLoader(dataset, batch_size, 
            shuffle=True, drop_last=True, **loader_params)
    else:
        dataloader = DataLoader(dataset, batch_size, 
            shuffle=False, drop_last=False, **loader_params)
    return dataloader


def get_optimizer(model, optimizer_type, **optim_params):
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
        )
    return optimizer

def get_scheduler(optimizer, scheduler_type, **scheduler_params):
    if scheduler_type == "default":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
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
