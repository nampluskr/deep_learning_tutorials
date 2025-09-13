import torch
from tqdm import tqdm
from time import time

from .trainer_base import BaseTrainer


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
        from trainer_base import EarlyStopper
        
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
        from trainer_base import get_scheduler
        
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