import torch
from tqdm import tqdm
from time import time

from trainer_base import BaseTrainer


class MemoryTrainer(BaseTrainer):
    """Trainer for memory-based anomaly detection models (PaDiM, PatchCore, SPADE)"""
    
    def __init__(self, modeler, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, scheduler, stopper, logger)
        self._fitted = getattr(modeler, '_fitted', False)

    @property
    def trainer_type(self):
        return "memory"

    def fit(self, train_loader, num_epochs=None, valid_loader=None):
        """
        Memory-based training: collect features in 1 epoch, then fit statistical model
        num_epochs is ignored for memory-based models (always 1 epoch)
        """
        # Force 1 epoch for memory-based models
        actual_epochs = 1
        
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metric_names)})

        self.log("\n > Training started...")

        # Feature collection phase
        start_time = time()
        train_results = self.run_epoch(train_loader, 1, actual_epochs, mode='train')
        
        train_info = self._format_train_info(train_results)

        # Update training history
        for key, value in train_results.items():
            if key in history:
                history[key].append(value)

        # Fitting phase (critical for memory-based models)
        if hasattr(self.modeler, 'fit') and hasattr(self.modeler, '_fitted') and not self.modeler._fitted:
            self.log(" > Fitting model parameters...")
            self.modeler.fit()
            self._fitted = True

        # Validation phase (if provided)
        valid_results = {}
        if valid_loader is not None:
            valid_results = self.run_epoch(valid_loader, 1, actual_epochs, mode='valid')

            valid_info = self._format_valid_info(valid_results)

            # Update validation history
            for key, value in valid_results.items():
                val_key = f"val_{key}"
                if val_key in history:
                    history[val_key].append(value)

            elapsed_time = time() - start_time
            self.log(f" [{1:2d}/{actual_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
        else:
            elapsed_time = time() - start_time
            self.log(f" [{1:2d}/{actual_epochs}] {train_info} ({elapsed_time:.1f}s)")

        # Final fitting for memory-based models (if not fitted during validation)
        if hasattr(self.modeler, 'fit') and hasattr(self.modeler, '_fitted') and not self.modeler._fitted:
            self.log("\n > Fitting model parameters...")
            self.modeler.fit()
            self._fitted = True

        self.log(" > Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Run a single epoch for memory-based training"""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        
        # Memory-specific accumulators (matching existing trainer.py)
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
                
                # Collect memory-specific metrics (from existing trainer.py)
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
                    pbar.set_postfix({
                        'batches': batch_results['memory_batches'],
                        'samples': batch_results['total_samples'],
                        'emb_mean': f"{batch_results['embedding_mean']:.3f}",
                        'emb_std': f"{batch_results['embedding_std']:.3f}",
                    })
                else:
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        **{name: f"{value:.3f}" for name, value in avg_metrics.items()}
                    })

        # Prepare results (matching existing trainer.py)
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        
        # Add memory-specific results
        if total_samples > 0:
            results['total_samples'] = total_samples
        if separations:
            results['separation'] = sum(separations) / len(separations)
        if embedding_means:
            results['avg_embedding_mean'] = sum(embedding_means) / len(embedding_means)
        
        return results

    def _format_train_info(self, train_results):
        """Format training results for logging (matching existing trainer.py)"""
        if 'total_samples' in train_results:
            return f"collected_samples={train_results['total_samples']}"
        else:
            return ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

    def _format_valid_info(self, valid_results):
        """Format validation results for logging (matching existing trainer.py)"""
        if 'separation' in valid_results:
            return f"score_sep={valid_results['separation']:.3f}"
        else:
            return ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

    def validate_with_test_data(self, test_loader):
        """
        Use test data for validation since normal validation doesn't work for memory-based models
        This solves the PaDiM validation problem (score_sep=0.000)
        """
        if not self._fitted and hasattr(self.modeler, 'fit'):
            self.log("Warning: Model not fitted yet. Fitting now...")
            self.modeler.fit()
            self._fitted = True

        self.log(" > Running validation with test data...")
        start_time = time()
        
        valid_results = self.run_epoch(test_loader, 1, 1, mode='valid')
        elapsed_time = time() - start_time
        
        valid_info = self._format_valid_info(valid_results)
        self.log(f" > Test-based validation: {valid_info} ({elapsed_time:.1f}s)")
        
        return valid_results

    def is_fitted(self):
        """Check if the model is fitted"""
        return self._fitted

    def get_memory_stats(self):
        """Get memory bank statistics if available"""
        if hasattr(self.modeler, 'get_memory_stats'):
            return self.modeler.get_memory_stats()
        elif hasattr(self.model, 'memory_bank'):
            if len(self.model.memory_bank) == 0:
                return {'total_samples': 0}
            
            total_samples = sum(emb.shape[0] for emb in self.model.memory_bank)
            return {'total_samples': total_samples}
        else:
            return {'total_samples': 0}