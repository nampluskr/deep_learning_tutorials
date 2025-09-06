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


class EarlyStopper:
    """Early stopping handler with best weights restoration."""

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


class BaseTrainer(ABC):
    """Base trainer with separated epoch methods for different model paradigms."""

    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        """Initialize trainer with modeler and training components."""
        self.modeler = modeler
        self.model = modeler.model
        self.metrics = modeler.metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.logger = logger
        self.metric_names = self.modeler.get_metric_names()
        
    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Complete training loop with validation."""
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metric_names)})

        self.log("\n > Training started...")
        for epoch in range(1, num_epochs + 1):
            start_time = time()

            # === Training Phase ===
            self.model.train()
            train_results = self.train_epoch(train_loader, epoch, num_epochs)

            # Store training results
            for key, value in train_results.items():
                if key in history:
                    history[key].append(value)

            # === Validation Phase ===
            valid_results = {}
            if valid_loader is not None:
                self.model.train()  # Keep training mode for validation monitoring
                valid_results = self.validation_epoch(valid_loader)

                # Store validation results
                for key, value in valid_results.items():
                    val_key = f"val_{key}"
                    if val_key in history:
                        history[val_key].append(value)

            # Logging
            train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])
            if valid_results:
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

            # Learning rate scheduling and early stopping
            self.check_learning_rate(epoch, train_results, valid_results)
            if self.check_early_stop(epoch, train_results, valid_results):
                break

        self.log(" > Training completed!")
        return history

    def predict(self, test_loader):
        """Deployment-optimized inference."""
        self.model.eval()
        return self.predict_epoch(test_loader)

    def test(self, test_loader, threshold_method="f1"):
        """Academic evaluation with comprehensive metrics."""
        self.model.eval()
        predictions = self.test_epoch(test_loader)

        scores = predictions['pred_scores']
        labels = predictions['gt_labels']

        # Calculate evaluation metrics
        results = {}
        results["auroc"] = AUROCMetric()(labels, scores)
        results["aupr"] = AUPRMetric()(labels, scores)

        threshold = OptimalThresholdMetric(method=threshold_method)(labels, scores)
        results["threshold"] = threshold

        binary_predictions = (scores >= threshold).float()
        results["accuracy"] = AccuracyMetric()(labels, binary_predictions)
        results["precision"] = PrecisionMetric()(labels, binary_predictions)
        results["recall"] = RecallMetric()(labels, binary_predictions)
        results["f1"] = F1Metric()(labels, binary_predictions)

        return results

    # ===================================================================
    # Separated Epoch Methods
    # ===================================================================

    def train_epoch(self, train_loader, epoch, num_epochs):
        """Training epoch for training monitoring (with Backpropagation)."""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        desc = f"Train [{epoch}/{num_epochs}]"
        with tqdm(train_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                batch_results = self.modeler.train_step(inputs, self.optimizer)

                total_loss += batch_results['loss']
                for metric_name in self.metric_names:
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]

                num_batches += 1
                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches
                             for name in self.metric_names}

                pbar.set_postfix({'loss': f"{avg_loss:.3f}",
                    **{name: f"{value:.3f}" for name, value in avg_metrics.items()}})

        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches
                       for name in self.metric_names})
        return results

    def validation_epoch(self, data_loader):
        """Validation epoch for training monitoring.(without Backpropagation)."""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        with torch.no_grad():
            desc = "Validation"
            with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
                for inputs in pbar:
                    # Use validation_step for training mode outputs + loss calculation
                    batch_results = self.modeler.validation_step(inputs)

                    total_loss += batch_results['loss']
                    for metric_name in self.metric_names:
                        if metric_name in batch_results:
                            total_metrics[metric_name] += batch_results[metric_name]

                    num_batches += 1
                    avg_loss = total_loss / num_batches
                    avg_metrics = {name: total_metrics[name] / num_batches
                                 for name in self.metric_names}

                    pbar.set_postfix({'loss': f"{avg_loss:.3f}",
                        **{name: f"{value:.3f}" for name, value in avg_metrics.items()}})

        # Return validation results for monitoring
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches
                       for name in self.metric_names})
        return results

    def predict_epoch(self, data_loader):
        """Prediction epoch for deployment scenarios."""
        all_pred_scores = []
        all_anomaly_maps = []

        with torch.no_grad():
            desc = "Predict"
            with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
                for inputs in pbar:
                    # Use predict_step for eval mode outputs
                    batch_results = self.modeler.predict_step(inputs)

                    all_pred_scores.append(batch_results['pred_scores'].cpu())
                    if 'anomaly_maps' in batch_results:
                        all_anomaly_maps.append(batch_results['anomaly_maps'].cpu())

                    pbar.set_postfix({'samples': len(all_pred_scores)})

        # Return prediction results for deployment
        results = {
            'pred_scores': torch.cat(all_pred_scores, dim=0)
        }

        if all_anomaly_maps:
            results['anomaly_maps'] = torch.cat(all_anomaly_maps, dim=0)

        return results

    def test_epoch(self, data_loader):
        """Test epoch for academic evaluation."""
        all_pred_scores = []
        all_anomaly_maps = []
        all_gt_labels = []

        with torch.no_grad():
            desc = "Test"
            with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
                for inputs in pbar:
                    # Use test_step for eval mode outputs
                    batch_results = self.modeler.test_step(inputs)

                    all_pred_scores.append(batch_results['pred_scores'].cpu())
                    if 'anomaly_maps' in batch_results:
                        all_anomaly_maps.append(batch_results['anomaly_maps'].cpu())

                    # Collect ground truth labels for evaluation
                    if 'label' in inputs:
                        all_gt_labels.append(inputs['label'].cpu())

                    pbar.set_postfix({'samples': len(all_pred_scores)})

        # Return test results for evaluation
        results = {
            'pred_scores': torch.cat(all_pred_scores, dim=0),
            'gt_labels': torch.cat(all_gt_labels, dim=0) if all_gt_labels else None
        }

        if all_anomaly_maps:
            results['anomaly_maps'] = torch.cat(all_anomaly_maps, dim=0)

        return results

    # ===================================================================
    # Helper Methods
    # ===================================================================

    def check_learning_rate(self, epoch, train_results, valid_results):
        """Check and update learning rate based on scheduler."""
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

    def check_early_stop(self, epoch, train_results, valid_results):
        """Check early stopping condition based on validation performance."""
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

    def save_model(self, path):
        """Save model state to disk."""
        if hasattr(self.modeler, 'save_model'):
            self.modeler.save_model(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state from disk."""
        if hasattr(self.modeler, 'load_model'):
            self.modeler.load_model(path)
        else:
            state_dict = torch.load(path, map_location=self.modeler.device)
            self.model.load_state_dict(state_dict)

    def log(self, message, level='info'):
        """Log training progress and information."""
        print(message)
        if self.logger:
            lines = message.split('\n')

            for i, line in enumerate(lines):
                if i == 0 and message.startswith('\n'):
                    self.logger.info(" ")
                if line.strip():
                    getattr(self.logger, level, self.logger.info)(line)
                elif i > 0:
                    self.logger.info(" ")

# ===================================================================
# Paradigm-Specific Trainers
# ===================================================================

class ReconstructionTrainer(BaseTrainer):
    """Trainer for reconstruction-based anomaly detection models."""

    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, optimizer, scheduler, stopper, logger)


class DistillationTrainer(BaseTrainer):
    """Trainer for distillation-based anomaly detection models."""

    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, optimizer, scheduler, stopper, logger)


class FlowTrainer(BaseTrainer):
    """Trainer for normalizing flow-based anomaly detection models."""

    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, optimizer, scheduler, stopper, logger)


class MemoryTrainer(BaseTrainer):
    """Trainer for memory-based anomaly detection models."""

    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, optimizer, scheduler, stopper, logger)
