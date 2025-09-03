import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
import os
from time import time
from copy import deepcopy


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(output_dir):
    """Create logger for experiment tracking."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'experiment.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class EarlyStopper:
    """Early stopping utility."""
    def __init__(self, patience=5, min_delta=1e-4, restore_best_weights=True):
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


class EpochStopper:
    """Simple epoch-based stopping."""
    def __init__(self, max_epoch=10):
        self.max_epoch = max_epoch
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        self.current_epoch += 1
        return self.current_epoch >= self.max_epoch


class BaseTrainer(ABC):
    """Base trainer class with unified interface for all anomaly detection models."""
    
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        """Initialize trainer with modeler and training components."""
        self.modeler = modeler
        self.model = modeler.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.logger = logger
        self.metric_names = self.modeler.get_metric_names()

    def log(self, message, level='info'):
        """Unified logging interface."""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        print(message)

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Main training loop using training_step + validation_step."""
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history['val_loss'] = []
            history.update({f"val_{name}": [] for name in self.metric_names})

        self.log("\n > Training started...")

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            
            # Training epoch
            train_results = self.run_epoch(train_loader, epoch, num_epochs, mode='train')
            
            # Format training info
            train_info = self._format_train_info(train_results)

            # Update training history
            for key, value in train_results.items():
                if key in history:
                    history[key].append(value)

            # Validation epoch
            valid_results = {}
            if valid_loader is not None:
                valid_results = self.run_epoch(valid_loader, epoch, num_epochs, mode='valid')

                # Format validation info
                valid_info = self._format_valid_info(valid_results)

                # Update validation history
                for key, value in valid_results.items():
                    if key in history:
                        history[key].append(value)

                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

            # Learning rate scheduling
            self._update_learning_rate(epoch, train_results, valid_results)

            # Check stopping condition
            if self._check_stopping_condition(epoch, train_results, valid_results):
                break

        self.log(" > Training completed!")
        return history

    def test(self, test_loader):
        """Comprehensive evaluation using compute_anomaly_maps/scores."""
        self.log(" > Testing started...")
        self.model.eval()
        
        all_anomaly_maps = []
        all_anomaly_scores = []
        all_labels = []
        total_samples = 0
        
        desc = "Testing"
        with tqdm(test_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                with torch.no_grad():
                    # Compute anomaly maps and scores
                    anomaly_maps = self.modeler.compute_anomaly_maps(inputs)
                    anomaly_scores = self.modeler.compute_anomaly_scores(inputs)
                    labels = inputs["label"]

                    all_anomaly_maps.append(anomaly_maps.cpu())
                    all_anomaly_scores.append(anomaly_scores.cpu())
                    all_labels.append(labels.cpu())
                    total_samples += labels.shape[0]
                
                pbar.set_postfix({'samples': total_samples})

        # Concatenate all results
        anomaly_maps_tensor = torch.cat(all_anomaly_maps, dim=0)
        anomaly_scores_tensor = torch.cat(all_anomaly_scores, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        
        # Compute comprehensive metrics
        test_metrics = self._compute_test_metrics(anomaly_scores_tensor, labels_tensor)
        
        self.log(f" > Testing completed: {total_samples} samples processed")
        
        return {
            'anomaly_maps': anomaly_maps_tensor,
            'anomaly_scores': anomaly_scores_tensor,
            'labels': labels_tensor,
            'metrics': test_metrics
        }

    def predict(self, test_loader):
        """Simple prediction returning scores and labels for evaluation."""
        self.model.eval()
        all_scores, all_labels = [], []

        desc = "Predict"
        with tqdm(test_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                with torch.no_grad():
                    scores = self.modeler.compute_anomaly_scores(inputs)
                    labels = inputs["label"]

                    all_scores.append(scores.cpu())
                    all_labels.append(labels.cpu())

        scores_tensor = torch.cat(all_scores, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        return scores_tensor, labels_tensor

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Internal method for running single epoch."""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        
        num_batches = 0

        desc = f"{mode.capitalize()} [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                if mode == 'train':
                    batch_results = self.modeler.training_step(inputs, self.optimizer)
                else:  # mode == 'valid'
                    # validation_step: same computation as training_step but no backprop
                    batch_results = self.modeler.validation_step(inputs)

                # Extract loss (with or without val_ prefix)
                loss = batch_results.get('loss', batch_results.get('val_loss', 0.0))
                total_loss += loss
                
                # Accumulate metrics (handle both prefixed and non-prefixed keys)
                for metric_name in self.metric_names:
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]
                    elif f'val_{metric_name}' in batch_results:
                        total_metrics[metric_name] += batch_results[f'val_{metric_name}']
                
                num_batches += 1

                # Update progress bar
                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in self.metric_names}
                
                progress_info = {'loss': f"{avg_loss:.3f}"}
                progress_info.update({name: f"{value:.3f}" for name, value in avg_metrics.items() if value != 0.0})
                
                pbar.set_postfix(progress_info)

        # Prepare results with consistent naming
        if mode == 'train':
            results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
            results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        else:  # mode == 'valid'
            results = {'val_loss': total_loss / num_batches if num_batches > 0 else 0.0}
            results.update({f"val_{name}": total_metrics[name] / num_batches for name in self.metric_names})
        
        return results

    def _format_train_info(self, train_results):
        """Format training results for logging."""
        if 'total_samples' in train_results:
            return f"collected_samples={train_results['total_samples']}"
        else:
            # Format all non-zero metrics
            info_parts = []
            for key, value in train_results.items():
                if isinstance(value, (int, float)) and value != 0.0:
                    info_parts.append(f"{key}={value:.3f}")
            return ", ".join(info_parts) if info_parts else "no_metrics"

    def _format_valid_info(self, valid_results):
        """Format validation results for logging."""
        info_parts = []
        
        # Add validation loss (remove val_ prefix for display)
        if 'val_loss' in valid_results:
            info_parts.append(f"loss={valid_results['val_loss']:.3f}")
        
        # Add validation metrics (remove val_ prefix for display)
        for key, value in valid_results.items():
            if key.startswith('val_') and key != 'val_loss' and isinstance(value, (int, float)) and value != 0.0:
                display_name = key.replace('val_', '')
                info_parts.append(f"{display_name}={value:.3f}")
        
        return ", ".join(info_parts) if info_parts else "no_metrics"

    def _update_learning_rate(self, epoch, train_results, valid_results):
        """Update learning rate using scheduler."""
        if self.scheduler is not None:
            last_lr = self.optimizer.param_groups[0]['lr']
            
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Use validation loss for plateau scheduler if available
                metric = valid_results.get('val_loss', train_results.get('loss', 0.0))
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            if abs(current_lr - last_lr) > 1e-12:
                self.log(f" > learning rate changed: {last_lr:.3e} => {current_lr:.3e}")

    def _check_stopping_condition(self, epoch, train_results, valid_results):
        """Check if training should stop."""
        if self.stopper is not None:
            # Use validation loss for early stopping if available
            current_loss = valid_results.get('val_loss', train_results.get('loss', 0.0))
            should_stop = self.stopper(current_loss, self.modeler.model)
            if should_stop:
                self.log(f" > Training stopped by stopper at epoch {epoch}")
                return True
        return False

    def _compute_test_metrics(self, scores, labels):
        """Compute comprehensive test metrics."""
        test_metrics = {}
        
        try:
            # Import metrics
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            labels_np = labels.cpu().numpy()
            scores_np = scores.cpu().numpy()
            
            # Basic metrics
            test_metrics['auroc'] = roc_auc_score(labels_np, scores_np)
            test_metrics['aupr'] = average_precision_score(labels_np, scores_np)
            
            # Score distribution analysis
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            
            if normal_mask.any() and anomaly_mask.any():
                normal_scores = scores[normal_mask]
                anomaly_scores = scores[anomaly_mask]
                
                test_metrics.update({
                    'score_mean': scores.mean().item(),
                    'score_std': scores.std().item(),
                    'normal_mean': normal_scores.mean().item(),
                    'anomaly_mean': anomaly_scores.mean().item(),
                    'separation': (anomaly_scores.mean() - normal_scores.mean()).item(),
                    'score_range': (scores.max() - scores.min()).item(),
                })
            
        except ImportError:
            test_metrics = {'auroc': 0.0, 'aupr': 0.0}
        
        return test_metrics

    def save_model(self, path):
        """Save model state."""
        self.modeler.save_model(path)

    def load_model(self, path):
        """Load model state."""
        self.modeler.load_model(path)

    @property
    @abstractmethod
    def trainer_type(self):
        """Return trainer type identifier."""
        pass


if __name__ == "__main__":
    pass