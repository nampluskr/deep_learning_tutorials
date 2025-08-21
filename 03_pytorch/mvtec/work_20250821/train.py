import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import logging
import os
from time import time


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


def train_epoch(model, train_loader, optimizer, loss_fn, metrics={}):
    """General training epoch function for supervised learning"""
    device = next(model.parameters()).device
    model.train()

    # Initialize result accumulation
    total_loss = 0.0
    total_metrics = {name: 0.0 for name in metrics.keys()}
    num_batches = 0

    with tqdm(train_loader, desc="Training", leave=False, file=sys.stdout,
              dynamic_ncols=True, ncols=100, ascii=True) as pbar:
        
        for batch_idx, batch_data in enumerate(pbar):
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
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            with torch.no_grad():
                for metric_name, metric_fn in metrics.items():
                    metric_value = metric_fn({'preds': model_outputs, 'targets': batch_data})
                    total_metrics[metric_name] += metric_value.item()
            
            # Update progress bar
            current_loss = total_loss / num_batches
            current_metrics = {name: value / num_batches for name, value in total_metrics.items()}
            pbar_dict = {'loss': f"{current_loss:.3f}"}
            pbar_dict.update({name: f"{value:.3f}" for name, value in current_metrics.items()})
            pbar.set_postfix(pbar_dict)

    # Return average results
    results = {'loss': total_loss / max(num_batches, 1)}
    results.update({name: value / max(num_batches, 1) for name, value in total_metrics.items()})
    
    return results


@torch.no_grad()
def validate_epoch(model, valid_loader, loss_fn, metrics={}):
    """General validation epoch function for supervised learning"""
    device = next(model.parameters()).device
    model.eval()

    # Initialize result accumulation
    total_loss = 0.0
    total_metrics = {name: 0.0 for name in metrics.keys()}
    num_batches = 0

    with tqdm(valid_loader, desc="Validation", leave=False, file=sys.stdout,
              dynamic_ncols=True, ncols=100, ascii=True) as pbar:
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            for key, value in batch_data.items():
                if torch.is_tensor(value):
                    batch_data[key] = value.to(device)
            
            # Forward pass
            model_outputs = model(batch_data)
            
            # Compute loss
            loss = loss_fn({'preds': model_outputs, 'targets': batch_data})
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            for metric_name, metric_fn in metrics.items():
                metric_value = metric_fn({'preds': model_outputs, 'targets': batch_data})
                total_metrics[metric_name] += metric_value.item()
            
            # Update progress bar
            current_loss = total_loss / num_batches
            current_metrics = {name: value / num_batches for name, value in total_metrics.items()}
            pbar_dict = {'loss': f"{current_loss:.3f}"}
            pbar_dict.update({name: f"{value:.3f}" for name, value in current_metrics.items()})
            pbar.set_postfix(pbar_dict)

    # Return average results
    results = {'loss': total_loss / max(num_batches, 1)}
    results.update({name: value / max(num_batches, 1) for name, value in total_metrics.items()})
    
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
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


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
        """Log message to file and print to console"""
        if self.logger:
            # Log to file with level and timestamp
            if level == 'info':
                self.logger.info(message)
            elif level == 'debug':
                self.logger.debug(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            else:
                self.logger.info(message)

        # Print clean message to terminal
        print(message)

    def _format_results(self, results, prefix=""):
        """Format results dictionary for logging"""
        formatted = []
        for key, value in results.items():
            if isinstance(value, float):
                formatted.append(f"{prefix}{key}={value:.3f}")
            else:
                formatted.append(f"{prefix}{key}={value}")
        return ', '.join(formatted)

    def compute_loss(self, data_dict):
        """Compute loss for anomaly detection models"""
        preds = data_dict['preds']
        targets = data_dict['targets']
        
        # Extract target tensor (if needed)
        if isinstance(targets, dict):
            target_tensor = targets['target']
        else:
            target_tensor = targets
        
        # Handle different model types
        if hasattr(self.model, 'model_type') and self.model.model_type == "fastflow":
            # FastFlow model - unsupervised, only needs pred
            return self.loss_fn(preds)
            
        elif hasattr(self.model, 'model_type') and self.model.model_type == "vae":
            # VAE model - need pred, target, mu, logvar
            if isinstance(preds, dict) and 'mu' in preds and 'logvar' in preds:
                pred_tensor = preds['reconstructed']
                mu = preds['mu']
                logvar = preds['logvar']
                return self.loss_fn(pred_tensor, target_tensor, mu, logvar)
            else:
                raise ValueError("VAE model should return dict with 'reconstructed', 'mu', and 'logvar'")
        else:
            # Standard autoencoder - need pred, target
            if isinstance(preds, dict) and 'reconstructed' in preds:
                pred_tensor = preds['reconstructed']
            else:
                pred_tensor = preds
            
            return self.loss_fn(pred_tensor, target_tensor)

    def compute_metric(self, metric_fn, data_dict):
        """Compute a single metric for anomaly detection models"""
        preds = data_dict['preds']
        targets = data_dict['targets']
        
        # Extract target tensor (if needed)
        if isinstance(targets, dict):
            target_tensor = targets['target']
        else:
            target_tensor = targets
        
        # Handle different model types for the metric
        if hasattr(self.model, 'model_type') and self.model.model_type == "fastflow":
            # FastFlow model - unsupervised, metric only needs pred
            try:
                # Try FastFlow-specific metric call (pred only)
                metric_value = metric_fn(preds)
            except TypeError:
                # Fall back to pred, target call (though target won't be used)
                metric_value = metric_fn(preds, None)
            
        elif hasattr(self.model, 'model_type') and self.model.model_type == "vae":
            # VAE model - check if this is a VAE-specific metric
            if isinstance(preds, dict) and 'mu' in preds and 'logvar' in preds:
                try:
                    # Try VAE-specific metric call (pred, target, mu, logvar)
                    pred_tensor = preds['reconstructed']
                    mu = preds['mu']
                    logvar = preds['logvar']
                    metric_value = metric_fn(pred_tensor, target_tensor, mu, logvar)
                except TypeError:
                    # Fall back to standard metric call (pred, target)
                    pred_tensor = preds['reconstructed'] if isinstance(preds, dict) else preds
                    metric_value = metric_fn(pred_tensor, target_tensor)
            else:
                # Standard reconstruction metric
                pred_tensor = preds['reconstructed'] if isinstance(preds, dict) else preds
                metric_value = metric_fn(pred_tensor, target_tensor)
        else:
            # Standard autoencoder metrics
            pred_tensor = preds['reconstructed'] if isinstance(preds, dict) else preds
            metric_value = metric_fn(pred_tensor, target_tensor)
        
        return metric_value

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Main training loop using general train/validate functions"""
        self.log(f"Starting training for {num_epochs} epochs...")
        if hasattr(self.model, 'model_type'):
            self.log(f"Model type: {self.model.model_type}")
        self.log(f"Device: {self.device}")
        self.log(f"Training samples: {len(train_loader.dataset)}")
        if valid_loader:
            self.log(f"Validation samples: {len(valid_loader.dataset)}")

        # Create wrapped loss and metric functions that use compute_loss/compute_metric
        def wrapped_loss_fn(data_dict):
            return self.compute_loss(data_dict)
        
        wrapped_metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            def create_metric_wrapper(metric_function):
                def metric_wrapper(data_dict):
                    return self.compute_metric(metric_function, data_dict)
                return metric_wrapper
            wrapped_metrics[metric_name] = create_metric_wrapper(metric_fn)

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            
            # Training phase using general function
            train_results = train_epoch(self.model, train_loader, self.optimizer,
                                      wrapped_loss_fn, wrapped_metrics)

            # Store training results
            for name, value in train_results.items():
                if name in self.history:
                    self.history[name].append(value)

            # Validation phase using general function
            if valid_loader is not None:
                valid_results = validate_epoch(self.model, valid_loader,
                                             wrapped_loss_fn, wrapped_metrics)

                # Store validation results
                for name, value in valid_results.items():
                    val_name = f"val_{name}"
                    if val_name in self.history:
                        self.history[val_name].append(value)

                # Log epoch summary
                train_summary = self._format_results(train_results)
                valid_summary = self._format_results(valid_results, prefix="val_")
                epoch_time = time() - start_time
                self.log(f"[Epoch {epoch:2d}/{num_epochs}] {train_summary} | {valid_summary} ({epoch_time:.0f}s)")
                
                # Early stopping check
                if self.early_stopping is not None:
                    if self.early_stopping(valid_results['loss'], self.model):
                        self.log(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                # Log epoch summary (training only)
                train_summary = self._format_results(train_results)
                epoch_time = time() - start_time
                self.log(f"[Epoch {epoch:2d}/{num_epochs}] {train_summary} ({epoch_time:.0f}s)")
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if valid_loader is not None:
                        self.scheduler.step(valid_results['loss'])
                else:
                    self.scheduler.step()

        self.log("Training completed!")
        return self.history

    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, filepath)
        self.log(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.log(f"Checkpoint loaded from {filepath}")


if __name__ == "__main__":
    pass