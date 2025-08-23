import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import logging
import os
from time import time
from copy import deepcopy


def get_logger(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'experiment.log')

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create file handler with timestamp and level
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate console output
    logger.propagate = False
    return logger


def get_optimizer(model, optimizer_type='adam', **optimizer_params):
    """Factory function to create an optimizer"""
    available_optimizers = ['adam', 'sgd', 'adamw']
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adam':
        params = {'lr': 0.001, 'weight_decay': 1e-5}
        params.update(optimizer_params)
        return optim.Adam(model.parameters(), **params)

    elif optimizer_type == 'sgd':
        params = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-5}
        params.update(optimizer_params)
        return optim.SGD(model.parameters(), **params)

    elif optimizer_type == 'adamw':
        params = {'lr': 0.001, 'weight_decay': 1e-5}
        params.update(optimizer_params)
        return optim.AdamW(model.parameters(), **params)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available optimizers: {available_optimizers}")


def get_scheduler(optimizer, scheduler_type='step', **scheduler_params):
    """Factory function to create a learning rate scheduler"""
    available_schedulers = ['step', 'multi_step', 'exponential', 'cosine', 'reduce_plateau', 'none']
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step':
        params = {'step_size': 10, 'gamma': 0.1}
        params.update(scheduler_params)
        return optim.lr_scheduler.StepLR(optimizer, **params)

    elif scheduler_type == 'multi_step':
        params = {'gamma': 0.1, 'milestones': [30, 80]}
        params.update(scheduler_params)
        return optim.lr_scheduler.MultiStepLR(optimizer, **params)

    elif scheduler_type == 'exponential':
        params = {'gamma': 0.9}
        params.update(scheduler_params)
        return optim.lr_scheduler.ExponentialLR(optimizer, **params)

    elif scheduler_type == 'cosine':
        params = {'T_max': 100, 'eta_min': 0.0}
        params.update(scheduler_params)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)

    elif scheduler_type == 'reduce_plateau':
        params = {'mode': 'min', 'factor': 0.5, 'patience': 5, 'verbose': True}
        params.update(scheduler_params)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available schedulers: {available_schedulers}")


def train_batch(model, batch_data, optimizer, loss_fn, metrics={}):
    """General training function for a single batch"""
    device = next(model.parameters()).device
    model.train()

    # Move data to device
    for key, value in batch_data.items():
        if torch.is_tensor(value):
            batch_data[key] = value.to(device)

    optimizer.zero_grad()

    # Forward pass
    model_outputs = model(batch_data)

    # Compute loss
    loss = loss_fn({'preds': model_outputs, 'targets': batch_data})

    # Backward pass
    loss.backward()
    optimizer.step()

    # Initialize results dict
    results = {'loss': loss.item()}

    # Compute metrics
    with torch.no_grad():
        for metric_name, metric_fn in metrics.items():
            metric_value = metric_fn({'preds': model_outputs, 'targets': batch_data})
            results[metric_name] = metric_value.item()

    return results


@torch.no_grad()
def validate_batch(model, batch_data, loss_fn, metrics={}):
    """General validation function for a single batch"""
    device = next(model.parameters()).device
    model.eval()

    # Move data to device
    for key, value in batch_data.items():
        if torch.is_tensor(value):
            batch_data[key] = value.to(device)

    # Forward pass
    model_outputs = model(batch_data)

    # Compute loss
    loss = loss_fn({'preds': model_outputs, 'targets': batch_data})

    # Initialize results dict
    results = {'loss': loss.item()}

    # Compute metrics
    for metric_name, metric_fn in metrics.items():
        metric_value = metric_fn({'preds': model_outputs, 'targets': batch_data})
        results[metric_name] = metric_value.item()

    return results


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False



from time import time
from tqdm import tqdm
import torch.optim as optim

class Trainer:
    """Wrapper trainer class for anomaly detection models"""

    def __init__(self, model, optimizer, loss_fn, metrics={}, scheduler=None, 
                 logger=None, early_stopping=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.scheduler = scheduler
        self.logger = logger
        self.early_stopping = early_stopping
        self.device = next(model.parameters()).device

        # Initialize history tracking
        self.history = {'loss': []}
        self.history.update({name: [] for name in metrics.keys()})
        self.history.update({f"val_{name}": [] for name in ['loss'] + list(metrics.keys())})

    def log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        print(message)

    def _format_results(self, results, prefix=""):
        return ', '.join([f"{prefix}{k}={v:.3f}" for k,v in results.items()])

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Main training loop using train_batch/validate_batch"""
        self.log(f"Starting training for {num_epochs} epochs...")
        if hasattr(self.model, 'model_type'):
            self.log(f"Model type: {self.model.model_type}")
        self.log(f"Device: {self.device}")
        self.log(f"Training samples: {len(train_loader.dataset)}")
        if valid_loader:
            self.log(f"Validation samples: {len(valid_loader.dataset)}")

        for epoch in range(1, num_epochs + 1):
            start_time = time()

            # --- Training phase ---
            self.model.train()
            total_loss, total_metrics, num_batches = 0.0, {m:0.0 for m in self.metrics}, 0

            with tqdm(train_loader, desc=f"Train {epoch}/{num_epochs}", leave=False) as pbar:
                for batch_data in pbar:
                    batch_results = train_batch(
                        self.model, batch_data, self.optimizer,
                        self.loss_fn, self.metrics
                    )
                    total_loss += batch_results['loss']
                    for m in self.metrics:
                        total_metrics[m] += batch_results[m]
                    num_batches += 1
                    avg_loss = total_loss / num_batches
                    avg_metrics = {m: total_metrics[m]/num_batches for m in self.metrics}
                    pbar.set_postfix({'loss': f"{avg_loss:.3f}", **{m:f"{v:.3f}" for m,v in avg_metrics.items()}})

            train_results = {'loss': total_loss/num_batches}
            train_results.update({m: total_metrics[m]/num_batches for m in self.metrics})
            for k,v in train_results.items():
                self.history[k].append(v)

            # --- Validation phase ---
            if valid_loader is not None:
                self.model.eval()
                total_loss, total_metrics, num_batches = 0.0, {m:0.0 for m in self.metrics}, 0
                with tqdm(valid_loader, desc=f"Valid {epoch}/{num_epochs}", leave=False) as pbar:
                    for batch_data in pbar:
                        batch_results = validate_batch(
                            self.model, batch_data,
                            self.loss_fn, self.metrics
                        )
                        total_loss += batch_results['loss']
                        for m in self.metrics:
                            total_metrics[m] += batch_results[m]
                        num_batches += 1
                        avg_loss = total_loss / num_batches
                        avg_metrics = {m: total_metrics[m]/num_batches for m in self.metrics}
                        pbar.set_postfix({'loss': f"{avg_loss:.3f}", **{m:f"{v:.3f}" for m,v in avg_metrics.items()}})

                valid_results = {'loss': total_loss/num_batches}
                valid_results.update({m: total_metrics[m]/num_batches for m in self.metrics})
                for k,v in valid_results.items():
                    self.history[f"val_{k}"].append(v)

                # 로그 출력
                epoch_time = time() - start_time
                self.log(f"[Epoch {epoch:2d}/{num_epochs}] "
                         f"{self._format_results(train_results)} | "
                         f"{self._format_results(valid_results, prefix='val_')} "
                         f"({epoch_time:.0f}s)")

                # Early stopping 체크
                if self.early_stopping is not None:
                    if self.early_stopping(valid_results['loss'], self.model):
                        self.log(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                # Validation 없음
                epoch_time = time() - start_time
                self.log(f"[Epoch {epoch:2d}/{num_epochs}] "
                         f"{self._format_results(train_results)} "
                         f"({epoch_time:.0f}s)")

            # 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if valid_loader is not None:
                        self.scheduler.step(valid_results['loss'])
                else:
                    self.scheduler.step()

        self.log("Training completed!")
        return self.history


if __name__ == "__main__":
    pass