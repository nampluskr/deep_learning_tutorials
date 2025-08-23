import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import logging
import os
from time import time
from copy import deepcopy


class Trainer:
    def __init__(self, modeler, optimizer, scheduler=None, logger=None, stopper=None):
        self.model = modeler.model
        self.loss_fn = modeler.loss_fn
        self.metrics = modeler.metrics
        self.device = modeler.device

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.stopper = stopper

    def fit(self, train_loader, num_epochs, valid_loader=None):
        history = {'loss': []}
        history.update({name: [] for name in self.metrics.keys()})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metrics.keys())})

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            self.model.train()
            train_results = self.run_epoch(train_loader, epoch, num_epochs, 'train')

            for key, value in train_results.items():
                history[key].append(value)

            valid_results = {}
            if valid_loader is not None:
                self.model.eval()
                valid_results = self.run_epoch(valid_loader, epoch, num_epochs, 'valid')

                for key, value in valid_results.items():
                    history[f"val_{key}"].append(value)

            epoch_time = time() - start_time
            # self._log_epoch_summary(epoch, num_epochs, train_results, valid_results, epoch_time)
            # self._update_scheduler(valid_results.get('loss', train_results['loss']))

            if self.stopper is not None:
                current_loss = valid_results.get('loss', train_results['loss'])

                if hasattr(self.stopper, 'update_metrics'):
                    current_metrics = {**train_results}
                    if valid_results:
                        current_metrics.update(valid_results)
                    self.stopper.update_metrics(current_metrics)

                should_stop = self.stopper(current_loss, self.model)
                if should_stop:
                    self.log(f"Training stopped by stopper at epoch {epoch}")
                    break

        self.log("Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metrics.keys()}
        num_batches = 0

        desc=f"{mode.capitalize()} [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=True) as pbar:
            for data in pbar:
                if mode == 'train':
                    batch_results = self.model.train_step(data,
                        self.optimizer, self.loss_fn, self.metrics, self.device)
                else:
                    batch_results = self.model.validate_step(data,
                        self.loss_fn, self.metrics, self.device)

                total_loss += batch_results['loss']
                for metric_name in self.metrics.keys():
                    total_metrics[metric_name] += batch_results[metric_name]
                num_batches += 1

                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in self.metrics.keys()}
                pbar.set_postfix({'loss': f"{avg_loss:.3f}", **{name: f"{value:.3f}" for name, value in avg_metrics.items()}})

        results = {'loss': total_loss / num_batches}
        results.update({name: total_metrics[name] / num_batches for name in self.metrics.keys()})
        return results

    def log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        print(message)


def get_logger(output_dir):
    """Setup logging configuration"""
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


def get_optimizer(model, optimizer_type='adamw', **optimizer_params):
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


def get_scheduler(optimizer, scheduler_type='plateau', **scheduler_params):
    """Factory function to create a learning rate scheduler"""
    available_schedulers = ['step', 'multi_step', 'exponential', 'cosine', 'plateau', 'none']
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

    elif scheduler_type == 'plateau':
        params = {'mode': 'min', 'factor': 0.5, 'patience': 5}
        params.update(scheduler_params)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available schedulers: {available_schedulers}")



if __name__ == "__main__":
    pass