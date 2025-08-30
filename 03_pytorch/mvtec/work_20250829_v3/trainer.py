import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import sys
import logging
import os
from time import time
from copy import deepcopy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(output_dir):
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


def get_stopper(name='early_stop', **params):
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


class Trainer:
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
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        print(message)

    def update_learning_rate(self, epoch, train_results, valid_results):
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

    def fit(self, train_loader, num_epochs, valid_loader=None):
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metric_names)})

        self.log("\n > Training started...")

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            train_results = self.run_epoch(train_loader, epoch, num_epochs, mode='train')
            train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

            for key, value in train_results.items():
                history[key].append(value)

            valid_results = {}
            if valid_loader is not None:
                valid_results = self.run_epoch(valid_loader, epoch, num_epochs, mode='valid')
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

                for key, value in valid_results.items():
                    history[f"val_{key}"].append(value)
                    
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] " f"{train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] " f"{train_info} ({elapsed_time:.1f}s)")

            self.update_learning_rate(epoch, train_results, valid_results)

            if self.check_stopping_condition(epoch, train_results, valid_results):
                break

        self.log(" > Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        desc = f"{mode.capitalize()} [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                if mode == 'train':
                    batch_results = self.modeler.train_step(inputs, self.optimizer)
                else:
                    batch_results = self.modeler.validate_step(inputs)

                total_loss += batch_results['loss']
                for metric_name in self.modeler.metrics.keys():
                    total_metrics[metric_name] += batch_results[metric_name]
                num_batches += 1

                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in self.metric_names}
                pbar.set_postfix({
                    'loss': f"{avg_loss:.3f}",
                    **{name: f"{value:.3f}" for name, value in avg_metrics.items()}
                })

        results = {'loss': total_loss / num_batches}
        results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        return results


class EarlyStopper:
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
    def __init__(self, max_epoch=10):
        self.max_epoch = max_epoch
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        self.current_epoch += 1
        return self.current_epoch >= self.max_epoch


if __name__ == "__main__":
    pass