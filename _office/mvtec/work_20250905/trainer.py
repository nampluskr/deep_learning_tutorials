import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
import os
from time import time
from copy import deepcopy

from metrics import AUROCMetric, AUPRMetric, AccuracyMetric, PrecisionMetric
from metrics import RecallMetric, F1Metric, OptimalThresholdMetric



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


# ===================================================================
# Base Trainer
# ===================================================================

class BaseTrainer(ABC):
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        self.modeler = modeler
        self.model = modeler.model
        self.metrics = modeler.metrics
        self.optimizer = optimizer
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

    @torch.no_grad()
    def predict(self, test_loader):
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

    @torch.no_grad()
    def evaluate(self, test_loader, threshold_method="f1"):
        scores, labels = self.predict(test_loader)

        results = {}
        results["auroc"] = AUROCMetric()(labels, scores)
        results["aupr"] = AUPRMetric()(labels, scores)

        threshold = OptimalThresholdMetric(method=threshold_method)(labels, scores)
        results["threshold"] = threshold

        predictions = (scores >= threshold).float()
        results["accuracy"] = AccuracyMetric()(labels, predictions)
        results["precision"] = PrecisionMetric()(labels, predictions)
        results["recall"] = RecallMetric()(labels, predictions)
        results["f1"] = F1Metric()(labels, predictions)
        return results

    def save_model(self, path):
        if hasattr(self.modeler, 'save_model'):
            self.modeler.save_model(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        if hasattr(self.modeler, 'load_model'):
            self.modeler.load_model(path)
        else:
            state_dict = torch.load(path, map_location=self.modeler.device)
            self.model.load_state_dict(state_dict)

    @abstractmethod
    def fit(self, train_loader, num_epochs=None, valid_loader=None):
        pass

    @abstractmethod
    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        pass


# ===================================================================
# Gradient Trainer: AEModeler / STFPMModeler
# ===================================================================

class GradientTrainer(BaseTrainer):
    """Trainer for gradient-based anomaly detection models (AE, STFPM, DRAEM, DFM)"""
    
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, optimizer, scheduler, stopper, logger)

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
                if key in history:
                    history[key].append(value)

            valid_results = {}
            if valid_loader is not None:
                valid_results = self.run_epoch(valid_loader, epoch, num_epochs, mode='valid')
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

                for key, value in valid_results.items():
                    val_key = f"val_{key}"
                    if val_key in history:
                        history[val_key].append(value)

                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

            self.update_learning_rate(epoch, train_results, valid_results)
            if self.check_stopping_condition(epoch, train_results, valid_results):
                break

        self.log(" > Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        num_batches = 0
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}

        desc = f"{mode.capitalize()} [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                if mode == 'train':
                    batch_results = self.modeler.train_step(inputs, self.optimizer)
                else:
                    batch_results = self.modeler.validate_step(inputs)

                total_loss += batch_results['loss']
                for metric_name in self.modeler.metrics.keys():
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]
                
                num_batches += 1
                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in self.metric_names}

                pbar.set_postfix({'loss': f"{avg_loss:.3f}",
                    **{name: f"{value:.3f}" for name, value in avg_metrics.items()}})

        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        return results
