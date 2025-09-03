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
    """Base trainer class with Template Method pattern."""
    
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None, **kwargs):
        self.modeler = modeler
        self.model = modeler.model
        self.metrics = modeler.metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.logger = logger
        self.metric_names = self.modeler.get_metric_names()

    # ========================================================================
    # Template Methods - Provide common workflow for training
    # ========================================================================

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Template method for training workflow."""
        history = self._initialize_history()
        
        self.log(f"\n > {self.trainer_type.capitalize()} training started...")

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            
            # Training epoch
            train_results = self.train_epoch(train_loader, epoch, num_epochs)
            self._update_history(history, train_results, 'train')

            # Validation epoch (for early stopping)
            valid_results = {}
            if valid_loader is not None:
                valid_results = self.validate_epoch(valid_loader, epoch, num_epochs)
                self._update_history(history, valid_results, 'valid')

            # Post-epoch processing
            elapsed_time = time() - start_time
            self._log_epoch_results(epoch, num_epochs, train_results, valid_results, elapsed_time)
            self._update_scheduler(train_results, valid_results)

            # Check stopping condition
            if self._check_stopping_condition(epoch, train_results, valid_results):
                break

        # Post-training hook
        self._post_training_hook(train_loader)
        
        self.log(" > Training completed!")
        return history

    def evaluate(self, test_loader):
        """Template method for evaluation workflow."""
        self.log(" > Evaluation started...")
        start_time = time()
        
        results = self.evaluate_epoch(test_loader, desc="Evaluate")
        
        elapsed_time = time() - start_time
        self._log_evaluation_results(results, elapsed_time)
        
        return results

    @torch.no_grad()
    def predict(self, test_loader):
        """Template method for prediction workflow."""
        self.log(" > Prediction started...")
        start_time = time()
        
        scores, labels = self.predict_epoch(test_loader, desc="Predict")
        
        elapsed_time = time() - start_time
        self.log(f" > Prediction completed: {len(scores)} samples ({elapsed_time:.1f}s)")
        
        return scores, labels

    # ========================================================================
    # Epoch-level Template Methods
    # ========================================================================

    def train_epoch(self, data_loader, epoch, num_epochs):
        """Template method for training epoch."""
        self.modeler.model.train()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        desc = self._get_train_epoch_desc(epoch, num_epochs)
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Training step with backpropagation
                batch_results = self.modeler.train_step(inputs, self.optimizer)

                # Accumulate results
                total_loss += batch_results['loss']
                self._accumulate_metrics(total_metrics, batch_results)
                num_batches += 1

                # Update progress bar
                avg_results = self._compute_average_results(total_loss, total_metrics, num_batches)
                self._update_train_progress_bar(pbar, avg_results)

        return self._prepare_epoch_results(total_loss, total_metrics, num_batches)

    def validate_epoch(self, data_loader, epoch, num_epochs):
        """Template method for validation epoch."""
        self.modeler.model.eval()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        desc = self._get_valid_epoch_desc(epoch, num_epochs)
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Validation step without backpropagation
                batch_results = self.modeler.validate_step(inputs)

                # Accumulate results
                total_loss += batch_results['loss']
                self._accumulate_metrics(total_metrics, batch_results)
                num_batches += 1

                # Update progress bar
                avg_results = self._compute_average_results(total_loss, total_metrics, num_batches)
                self._update_valid_progress_bar(pbar, avg_results)

        return self._prepare_epoch_results(total_loss, total_metrics, num_batches)

    def evaluate_epoch(self, data_loader, desc="Evaluate"):
        """Template method for evaluation epoch."""
        self.modeler.model.eval()
        
        all_scores = []
        all_labels = []
        score_stats = []
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Evaluation step - generates anomaly maps and scores
                batch_results = self.modeler.evaluate_step(inputs)
                
                # Collect results
                self._collect_evaluation_batch_results(batch_results, inputs, all_scores, all_labels, score_stats)
                num_batches += 1

                # Update progress bar
                self._update_eval_progress_bar(pbar, score_stats)

        return self._prepare_evaluation_results(all_scores, all_labels, score_stats)

    def predict_epoch(self, data_loader, desc="Predict"):
        """Template method for prediction epoch."""
        self.modeler.model.eval()
        
        all_scores = []
        all_labels = []

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Prediction step - returns only scores
                scores = self.modeler.predict_step(inputs)
                labels = inputs["label"]

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        return torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0)

    # ========================================================================
    # Hook Methods - For trainer-specific customization
    # ========================================================================

    def _initialize_history(self):
        """Hook: Initialize training history (default implementation)."""
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        return history

    def _get_train_epoch_desc(self, epoch, num_epochs):
        """Hook: Get training epoch description."""
        return f"Train [{epoch}/{num_epochs}]"

    def _get_valid_epoch_desc(self, epoch, num_epochs):
        """Hook: Get validation epoch description."""
        return f"Valid [{epoch}/{num_epochs}]"

    def _update_train_progress_bar(self, pbar, avg_results):
        """Hook: Update training progress bar."""
        progress_info = {'loss': f"{avg_results['loss']:.3f}"}
        for name, value in avg_results.items():
            if name != 'loss' and value != 0.0:
                progress_info[name] = f"{value:.3f}"
        pbar.set_postfix(progress_info)

    def _update_valid_progress_bar(self, pbar, avg_results):
        """Hook: Update validation progress bar."""
        self._update_train_progress_bar(pbar, avg_results)  # Default: same as training

    def _update_eval_progress_bar(self, pbar, score_stats):
        """Hook: Update evaluation progress bar."""
        if score_stats:
            avg_separation = sum(s.get('separation', 0.0) for s in score_stats) / len(score_stats)
            pbar.set_postfix({'separation': f"{avg_separation:.3f}"})

    def _log_epoch_results(self, epoch, num_epochs, train_results, valid_results, elapsed_time):
        """Hook: Log epoch results."""
        train_info = self._format_results_for_logging(train_results)

        if valid_results:
            valid_info = self._format_results_for_logging(valid_results, prefix="val")
            self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | {valid_info} ({elapsed_time:.1f}s)")
        else:
            self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

    def _log_evaluation_results(self, results, elapsed_time):
        """Hook: Log evaluation results."""
        if 'separation' in results:
            self.log(f" > Evaluation completed: separation={results['separation']:.3f} ({elapsed_time:.1f}s)")
        else:
            self.log(f" > Evaluation completed ({elapsed_time:.1f}s)")

    def _format_results_for_logging(self, results, prefix=""):
        """Hook: Format results for logging."""
        prefix_str = f"({prefix}) " if prefix else ""
        
        if 'separation' in results:
            return f"{prefix_str}separation={results['separation']:.3f}"
        else:
            formatted_items = []
            for key, value in results.items():
                if key not in ['scores', 'labels', 'pred_scores', 'anomaly_maps']:
                    formatted_items.append(f'{key}={value:.3f}')
            return f"{prefix_str}" + ", ".join(formatted_items)

    def _post_training_hook(self, train_loader):
        """Hook: Post-training processing (e.g., fitting for memory models)."""
        # Default implementation for memory-based models compatibility
        if hasattr(self.modeler, 'fit') and hasattr(self.modeler, '_fitted') and not self.modeler._fitted:
            self.log("\n > Fitting model parameters...")
            self.modeler.fit()

    # ========================================================================
    # Helper Methods - Common utilities
    # ========================================================================

    def _accumulate_metrics(self, total_metrics, batch_results):
        """Accumulate metrics from batch results."""
        for metric_name in total_metrics.keys():
            if metric_name in batch_results:
                total_metrics[metric_name] += batch_results[metric_name]

    def _compute_average_results(self, total_loss, total_metrics, num_batches):
        """Compute average results from accumulated values."""
        if num_batches == 0:
            return {'loss': 0.0}
        
        avg_results = {'loss': total_loss / num_batches}
        avg_results.update({name: total_metrics[name] / num_batches for name in total_metrics.keys()})
        return avg_results

    def _prepare_epoch_results(self, total_loss, total_metrics, num_batches):
        """Prepare final epoch results."""
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in total_metrics.keys()})
        return results

    def _collect_evaluation_batch_results(self, batch_results, inputs, all_scores, all_labels, score_stats):
        """Collect batch results during evaluation."""
        if 'pred_scores' in batch_results:
            all_scores.append(batch_results['pred_scores'].cpu())
            all_labels.append(inputs['label'].cpu())
            
            # Collect statistics
            stats = {}
            for key in ['score_mean', 'score_std', 'separation']:
                if key in batch_results:
                    stats[key] = batch_results[key]
            if stats:
                score_stats.append(stats)

    def _prepare_evaluation_results(self, all_scores, all_labels, score_stats):
        """Prepare final evaluation results."""
        results = {
            'scores': torch.cat(all_scores, dim=0) if all_scores else torch.tensor([]),
            'labels': torch.cat(all_labels, dim=0) if all_labels else torch.tensor([]),
        }
        
        # Add aggregated statistics
        if score_stats:
            for key in ['score_mean', 'score_std', 'separation']:
                values = [s[key] for s in score_stats if key in s]
                if values:
                    results[key] = sum(values) / len(values)
        
        return results

    def _update_history(self, history, results, prefix=''):
        """Update training history with epoch results."""
        key_prefix = f"{prefix}_" if prefix else ""
        
        for key, value in results.items():
            if key in ['scores', 'labels', 'pred_scores', 'anomaly_maps']:
                continue
                
            history_key = f"{key_prefix}{key}"
            if history_key not in history:
                history[history_key] = []
            history[history_key].append(value)

    def _update_scheduler(self, train_results, valid_results):
        """Update learning rate scheduler."""
        if self.scheduler is not None:
            last_lr = self.optimizer.param_groups[0]['lr']
            
            if hasattr(self.scheduler, 'step'):
                # For ReduceLROnPlateau, use validation loss if available
                if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                    metric = valid_results.get('loss', train_results['loss'])
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                if abs(current_lr - last_lr) > 1e-12:
                    self.log(f" > Learning rate changed: {last_lr:.3e} => {current_lr:.3e}")

    def _check_stopping_condition(self, epoch, train_results, valid_results):
        """Check if training should stop early."""
        if self.stopper is not None:
            current_loss = valid_results.get('loss', train_results['loss'])
            
            should_stop = self.stopper(current_loss, self.modeler.model)
            if should_stop:
                self.log(f" > Training stopped by stopper at epoch {epoch}")
                return True
        return False

    def log(self, message, level='info'):
        """Unified logging interface."""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message)

    # ========================================================================
    # Abstract Properties and Methods
    # ========================================================================

    @property
    @abstractmethod
    def trainer_type(self):
        """Return trainer type identifier."""
        pass

    # ========================================================================
    # Model persistence and utilities
    # ========================================================================

    def save_model(self, path):
        """Save model state."""
        if hasattr(self.modeler, 'save_model'):
            self.modeler.save_model(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state."""
        if hasattr(self.modeler, 'load_model'):
            self.modeler.load_model(path)
        else:
            state_dict = torch.load(path, map_location=self.modeler.device)
            self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    pass