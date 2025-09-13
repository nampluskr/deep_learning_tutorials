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
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Get available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(output_dir):
    """Create logger for experiment tracking"""
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


def get_optimizer(model, name='adamw', **params):
    """Factory function for optimizers"""
    available_list = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'adamw': optim.AdamW,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {'lr': 0.001}
    default_params.update(params)
    return selected(model.parameters(), **default_params)


def get_scheduler(optimizer, name='plateau', **params):
    """Factory function for learning rate schedulers"""
    available_list = {
        'step': optim.lr_scheduler.StepLR,
        'multi_step': optim.lr_scheduler.MultiStepLR,
        'exponential': optim.lr_scheduler.ExponentialLR,
        'cosine': optim.lr_scheduler.CosineAnnealingLR,
        'plateau': optim.lr_scheduler.ReduceLROnPlateau,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(optimizer, **default_params)


class EarlyStopper:
    """Early stopping utility"""
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
    """Simple epoch-based stopping"""
    def __init__(self, max_epoch=10):
        self.max_epoch = max_epoch
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        self.current_epoch += 1
        return self.current_epoch >= self.max_epoch


def get_stopper(name='early_stop', **params):
    """Factory function for stopping criteria"""
    available_list = {
        'early_stop': EarlyStopper,
        'epoch_stop': EpochStopper,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(**default_params)


class BaseTrainer(ABC):
    """Base trainer class with common functionality"""
    
    def __init__(self, modeler, scheduler=None, stopper=None, logger=None):
        self.modeler = modeler
        self.model = modeler.model
        self.metrics = modeler.metrics
        self.optimizer = modeler.configure_optimizers()
        self.scheduler = scheduler
        self.stopper = stopper
        self.logger = logger
        self.metric_names = self.modeler.get_metric_names()

    def log(self, message, level='info'):
        """Unified logging interface"""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        print(message)

    def update_learning_rate(self, epoch, train_results, valid_results):
        """Update learning rate using scheduler"""
        if self.scheduler is not None:
            last_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                metric = valid_results.get('loss', train_results['loss'])
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            if abs(current_lr - last_lr) > 1e-12:
                self.log(f" > learning rate changed: {last_lr:.3e} => {current_lr:.3e}")

    def check_stopping_condition(self, epoch, train_results, valid_results):
        """Check if training should stop"""
        if self.stopper is not None:
            current_loss = valid_results.get('loss', train_results['loss'])

            if hasattr(self.stopper, 'update_metrics'):
                current_metrics = {**train_results}
                if valid_results:
                    current_metrics.update(valid_results)
                self.stopper.update_metrics(current_metrics)

            should_stop = self.stopper(current_loss, self.modeler.model)
            if should_stop:
                self.log(f"Training stopped by stopper at epoch {epoch}")
                return True
        return False

    @torch.no_grad()
    def predict(self, test_loader):
        """Common predict functionality for all trainers"""
        self.model.eval()
        all_scores, all_labels = [], []

        desc = "Predict"
        with tqdm(test_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                scores = self.modeler.predict_step(inputs)
                labels = inputs["label"]

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        scores_tensor = torch.cat(all_scores, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        return scores_tensor, labels_tensor

    def save_model(self, path):
        """Save model state"""
        if hasattr(self.modeler, 'save_model'):
            self.modeler.save_model(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state"""
        if hasattr(self.modeler, 'load_model'):
            self.modeler.load_model(path)
        else:
            state_dict = torch.load(path, map_location=self.modeler.device)
            self.model.load_state_dict(state_dict)

    @abstractmethod
    def fit(self, train_loader, num_epochs=None, valid_loader=None):
        """Abstract method for training implementation"""
        pass

    @abstractmethod
    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Abstract method for running a single epoch"""
        pass

    @property
    @abstractmethod
    def trainer_type(self):
        """Return trainer type identifier"""
        pass
    

class GradientTrainer(BaseTrainer):
    """Trainer for gradient-based anomaly detection models (AE, STFPM, DRAEM, DFM)"""
    
    def __init__(self, modeler, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, scheduler, stopper, logger)

    @property
    def trainer_type(self):
        return "gradient"

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Standard gradient-based training with multiple epochs (matching existing trainer.py)"""
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metric_names)})

        self.log("\n > Training started...")

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            train_results = self.run_epoch(train_loader, epoch, num_epochs, mode='train')
            
            # Format training info (matching existing trainer.py)
            if 'total_samples' in train_results:
                train_info = f"collected_samples={train_results['total_samples']}"
            else:
                train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

            # Update training history
            for key, value in train_results.items():
                if key in history:
                    history[key].append(value)

            # Validation phase
            valid_results = {}
            if valid_loader is not None:
                # Memory-based models need fitting before validation (keeping existing logic)
                if hasattr(self.modeler, 'fit') and hasattr(self.modeler, '_fitted') and not self.modeler._fitted:
                    self.log(" > Fitting model parameters...")
                    self.modeler.fit()

                valid_results = self.run_epoch(valid_loader, epoch, num_epochs, mode='valid')

                # Format validation info (matching existing trainer.py)
                if 'separation' in valid_results:
                    valid_info = f"score_sep={valid_results['separation']:.3f}"
                else:
                    valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

                # Update validation history
                for key, value in valid_results.items():
                    val_key = f"val_{key}"
                    if val_key in history:
                        history[val_key].append(value)

                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

            # Learning rate scheduling
            self.update_learning_rate(epoch, train_results, valid_results)

            # Check stopping condition
            if self.check_stopping_condition(epoch, train_results, valid_results):
                break

        # Final fitting for memory-based models (keeping existing logic for compatibility)
        if hasattr(self.modeler, 'fit') and hasattr(self.modeler, '_fitted') and not self.modeler._fitted:
            self.log("\n > Fitting model parameters...")
            self.modeler.fit()

        self.log(" > Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Run a single epoch for gradient-based training (matching existing trainer.py)"""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        
        # Memory-specific accumulators (for compatibility with STFPM)
        total_samples = 0
        separations = []
        embedding_means = []
        num_batches = 0

        desc = f"{mode.capitalize()} [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                if mode == 'train':
                    batch_results = self.modeler.train_step(inputs, self.optimizer)
                else:
                    batch_results = self.modeler.validate_step(inputs)

                total_loss += batch_results['loss']
                
                # Collect memory-specific metrics (for STFPM compatibility)
                if 'total_samples' in batch_results:
                    total_samples = batch_results['total_samples']
                if 'separation' in batch_results:
                    separations.append(batch_results['separation'])
                if 'avg_embedding_mean' in batch_results:
                    embedding_means.append(batch_results['avg_embedding_mean'])
                
                # Standard metrics
                for metric_name in self.modeler.metrics.keys():
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]
                
                num_batches += 1

                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in self.metric_names}

                # Progress bar update (matching existing trainer.py logic)
                if 'memory_batches' in batch_results:
                    # For STFPM and similar models that monitor memory
                    pbar.set_postfix({
                        'batches': batch_results['memory_batches'],
                        'samples': batch_results['total_samples'],
                        'emb_mean': f"{batch_results['embedding_mean']:.3f}",
                        'emb_std': f"{batch_results['embedding_std']:.3f}",
                    })
                elif 'memory_samples' in batch_results:
                    # For STFPM specifically
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        'mem_samples': batch_results['memory_samples'],
                        'emb_mean': f"{batch_results.get('embedding_mean', 0.0):.3f}",
                    })
                else:
                    # Standard gradient-based progress
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        **{name: f"{value:.3f}" for name, value in avg_metrics.items()}
                    })

        # Prepare results (matching existing trainer.py)
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        
        # Add memory-specific results (for STFPM compatibility)
        if total_samples > 0:
            results['total_samples'] = total_samples
        if separations:
            results['separation'] = sum(separations) / len(separations)
        if embedding_means:
            results['avg_embedding_mean'] = sum(embedding_means) / len(embedding_means)
        
        return results

    def fit_with_early_stopping(self, train_loader, max_epochs, valid_loader, patience=10, min_delta=1e-4):
        """Convenience method for training with early stopping"""
        
        original_stopper = self.stopper
        self.stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best_weights=True)
        
        try:
            history = self.fit(train_loader, max_epochs, valid_loader)
        finally:
            self.stopper = original_stopper
            
        return history

    def fit_with_lr_scheduling(self, train_loader, num_epochs, valid_loader=None, 
                              scheduler_type='plateau', **scheduler_params):
        """Convenience method for training with learning rate scheduling"""
        
        original_scheduler = self.scheduler
        self.scheduler = get_scheduler(self.optimizer, scheduler_type, **scheduler_params)
        
        try:
            history = self.fit(train_loader, num_epochs, valid_loader)
        finally:
            self.scheduler = original_scheduler
            
        return history

    def get_memory_stats(self):
        """Get memory bank statistics if available (for STFPM compatibility)"""
        if hasattr(self.modeler, 'get_memory_stats'):
            return self.modeler.get_memory_stats()
        elif hasattr(self.model, 'memory_bank'):
            if len(self.model.memory_bank) == 0:
                return {'total_samples': 0}
            
            total_samples = sum(emb.shape[0] for emb in self.model.memory_bank)
            return {'total_samples': total_samples}
        else:
            return {'total_samples': 0}