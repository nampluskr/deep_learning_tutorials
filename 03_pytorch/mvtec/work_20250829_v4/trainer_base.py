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