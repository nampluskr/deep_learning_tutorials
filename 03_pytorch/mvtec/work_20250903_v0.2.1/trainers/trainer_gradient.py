import torch
from tqdm import tqdm
from time import time

from .trainer_base import BaseTrainer


class GradientTrainer(BaseTrainer):
    """Trainer for gradient-based anomaly detection models (AE, STFPM, DRAEM, DFM)."""
    
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None, **kwargs):
        super().__init__(modeler, scheduler, stopper, logger)
        self.optimizer = optimizer

    @property
    def trainer_type(self):
        return "gradient"

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Train the model with gradient-based optimization."""
        history = self._initialize_history()
        
        self.log("\n > Training started...")

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            
            # Training epoch
            train_results = self.train_epoch(train_loader, epoch, num_epochs)
            
            # Update training history
            self._update_history(history, train_results, 'train')

            # Validation epoch (for early stopping)
            valid_results = {}
            if valid_loader is not None:
                valid_results = self.validate_epoch(valid_loader, epoch, num_epochs)
                self._update_history(history, valid_results, 'valid')

            # Logging
            elapsed_time = time() - start_time
            self._log_epoch_results(epoch, num_epochs, train_results, valid_results, elapsed_time)

            # Learning rate scheduling
            self._update_scheduler(train_results, valid_results)

            # Early stopping check
            if self._check_stopping_condition(epoch, train_results, valid_results):
                break

        # Final fitting for memory-based models (compatibility with existing code)
        if hasattr(self.modeler, 'fit') and hasattr(self.modeler, '_fitted') and not self.modeler._fitted:
            self.log("\n > Fitting model parameters...")
            self.modeler.fit()

        self.log(" > Training completed!")
        return history

    def train_epoch(self, data_loader, epoch, num_epochs):
        """Run a single training epoch."""
        self.modeler.model.train()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.modeler.get_metric_names()}
        num_batches = 0

        desc = f"Train [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Forward pass with backpropagation
                batch_results = self.modeler.train_step(inputs, self.optimizer)

                # Accumulate results
                total_loss += batch_results['loss']
                for metric_name in total_metrics.keys():
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]
                
                num_batches += 1

                # Update progress bar
                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in total_metrics.keys()}
                
                progress_info = {'loss': f"{avg_loss:.3f}"}
                for name, value in avg_metrics.items():
                    if value != 0.0:
                        progress_info[name] = f"{value:.3f}"
                
                pbar.set_postfix(progress_info)

        # Prepare epoch results
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in total_metrics.keys()})
        
        return results

    def validate_epoch(self, data_loader, epoch, num_epochs):
        """Run a single validation epoch (for early stopping)."""
        self.modeler.model.eval()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.modeler.get_metric_names()}
        num_batches = 0

        desc = f"Valid [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Forward pass without backpropagation
                batch_results = self.modeler.validate_step(inputs)

                # Accumulate results
                total_loss += batch_results['loss']
                for metric_name in total_metrics.keys():
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]
                
                num_batches += 1

                # Update progress bar
                avg_loss = total_loss / num_batches
                avg_metrics = {name: total_metrics[name] / num_batches for name in total_metrics.keys()}
                
                progress_info = {'loss': f"{avg_loss:.3f}"}
                for name, value in avg_metrics.items():
                    if value != 0.0:
                        progress_info[name] = f"{value:.3f}"
                
                pbar.set_postfix(progress_info)

        # Prepare epoch results
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in total_metrics.keys()})
        
        return results

    def evaluate_epoch(self, data_loader, desc="Evaluate"):
        """Run a single evaluation epoch (for anomaly detection evaluation)."""
        self.modeler.model.eval()
        
        all_scores = []
        all_labels = []
        score_stats = []
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                # Evaluation step - generates anomaly maps and scores
                batch_results = self.modeler.evaluate_step(inputs)
                
                # Collect scores and labels
                if 'pred_scores' in batch_results:
                    all_scores.append(batch_results['pred_scores'].cpu())
                    all_labels.append(inputs['label'].cpu())
                    
                    # Collect statistics
                    if 'score_mean' in batch_results:
                        score_stats.append({
                            'score_mean': batch_results['score_mean'],
                            'score_std': batch_results['score_std'],
                            'separation': batch_results.get('separation', 0.0)
                        })
                
                num_batches += 1

                # Update progress bar
                if score_stats:
                    avg_separation = sum(s['separation'] for s in score_stats) / len(score_stats)
                    pbar.set_postfix({'separation': f"{avg_separation:.3f}"})

        # Prepare evaluation results
        results = {
            'scores': torch.cat(all_scores, dim=0) if all_scores else torch.tensor([]),
            'labels': torch.cat(all_labels, dim=0) if all_labels else torch.tensor([]),
        }
        
        # Add aggregated statistics
        if score_stats:
            results.update({
                'score_mean': sum(s['score_mean'] for s in score_stats) / len(score_stats),
                'score_std': sum(s['score_std'] for s in score_stats) / len(score_stats),
                'separation': sum(s['separation'] for s in score_stats) / len(score_stats),
            })
        
        return results

    def predict_epoch(self, data_loader, desc="Predict"):
        """Run a single prediction epoch (returns only scores)."""
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

    def evaluate(self, test_loader):
        """Run complete evaluation on test set."""
        self.log(" > Evaluation started...")
        start_time = time()
        
        results = self.evaluate_epoch(test_loader, desc="Evaluate")
        
        elapsed_time = time() - start_time
        if 'separation' in results:
            self.log(f" > Evaluation completed: separation={results['separation']:.3f} ({elapsed_time:.1f}s)")
        else:
            self.log(f" > Evaluation completed ({elapsed_time:.1f}s)")
        
        return results

    @torch.no_grad()
    def predict(self, test_loader):
        """Run prediction on test set."""
        self.log(" > Prediction started...")
        start_time = time()
        
        scores, labels = self.predict_epoch(test_loader, desc="Predict")
        
        elapsed_time = time() - start_time
        self.log(f" > Prediction completed: {len(scores)} samples ({elapsed_time:.1f}s)")
        
        return scores, labels

    def _initialize_history(self):
        """Initialize training history dictionary."""
        history = {'loss': []}
        history.update({name: [] for name in self.modeler.get_metric_names()})
        return history

    def _update_history(self, history, results, prefix=''):
        """Update training history with epoch results."""
        key_prefix = f"{prefix}_" if prefix else ""
        
        for key, value in results.items():
            if key in ['scores', 'labels']:  # Skip evaluation-specific keys
                continue
                
            history_key = f"{key_prefix}{key}"
            if history_key not in history:
                history[history_key] = []
            history[history_key].append(value)

    def _log_epoch_results(self, epoch, num_epochs, train_results, valid_results, elapsed_time):
        """Log epoch results in consistent format."""
        # Format training results
        train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items() 
                               if key not in ['scores', 'labels']])

        if valid_results:
            # Format validation results  
            if 'separation' in valid_results:
                valid_info = f"separation={valid_results['separation']:.3f}"
            else:
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items() 
                                      if key not in ['scores', 'labels']])
            
            self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
        else:
            self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

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
        """Log message using logger or print."""
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message)

    def get_memory_stats(self):
        """Get memory bank statistics if available (for compatibility)."""
        if hasattr(self.modeler, 'get_memory_stats'):
            return self.modeler.get_memory_stats()
        elif hasattr(self.modeler.model, 'memory_bank'):
            if hasattr(self.modeler.model.memory_bank, '__len__') and len(self.modeler.model.memory_bank) == 0:
                return {'total_samples': 0}
            
            if hasattr(self.modeler.model, 'memory_bank'):
                total_samples = sum(emb.shape[0] for emb in self.modeler.model.memory_bank if hasattr(emb, 'shape'))
                return {'total_samples': total_samples}
        
        return {'total_samples': 0}