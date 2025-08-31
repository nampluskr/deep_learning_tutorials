import torch
from tqdm import tqdm
from time import time

from trainer_base import BaseTrainer


class FlowTrainer(BaseTrainer):
    """Trainer for flow-based anomaly detection models (FastFlow, CFlow)"""
    
    def __init__(self, modeler, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, scheduler, stopper, logger)

    @property
    def trainer_type(self):
        return "flow"

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Flow-based training with likelihood maximization (matching existing trainer.py structure)"""
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

        self.log(" > Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Run a single epoch for flow-based training (based on existing trainer.py pattern)"""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        
        # Flow-specific accumulators
        total_log_likelihood = 0.0
        total_jacobian_loss = 0.0
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
                
                # Flow-specific metrics
                if 'log_likelihood' in batch_results:
                    total_log_likelihood += batch_results['log_likelihood']
                if 'jacobian_loss' in batch_results:
                    total_jacobian_loss += batch_results['jacobian_loss']
                
                # Memory-specific metrics (for compatibility)
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

                # Progress bar update (matching existing trainer.py with flow enhancements)
                if 'memory_batches' in batch_results:
                    # For flow models that might use memory banks
                    pbar.set_postfix({
                        'batches': batch_results['memory_batches'],
                        'samples': batch_results['total_samples'],
                        'emb_mean': f"{batch_results['embedding_mean']:.3f}",
                        'emb_std': f"{batch_results['embedding_std']:.3f}",
                    })
                elif total_log_likelihood != 0.0:
                    # Flow-specific progress display
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        'log_prob': f"{total_log_likelihood / num_batches:.3f}",
                        **{name: f"{value:.3f}" for name, value in avg_metrics.items()}
                    })
                else:
                    # Standard progress display
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.3f}",
                        **{name: f"{value:.3f}" for name, value in avg_metrics.items()}
                    })

        # Prepare results (matching existing trainer.py)
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        
        # Add flow-specific results
        if total_log_likelihood != 0.0:
            results['log_likelihood'] = total_log_likelihood / num_batches
        if total_jacobian_loss != 0.0:
            results['jacobian_loss'] = total_jacobian_loss / num_batches
        
        # Add memory-specific results (for compatibility)
        if total_samples > 0:
            results['total_samples'] = total_samples
        if separations:
            results['separation'] = sum(separations) / len(separations)
        if embedding_means:
            results['avg_embedding_mean'] = sum(embedding_means) / len(embedding_means)
        
        return results

    def fit_with_annealing(self, train_loader, num_epochs, valid_loader=None, 
                          initial_lr=1e-3, final_lr=1e-5):
        """Convenience method for training with cosine annealing (common for flow models)"""
        from trainer_base import get_scheduler
        
        original_scheduler = self.scheduler
        self.scheduler = get_scheduler(
            self.optimizer, 
            'cosine', 
            T_max=num_epochs, 
            eta_min=final_lr
        )
        
        try:
            history = self.fit(train_loader, num_epochs, valid_loader)
        finally:
            self.scheduler = original_scheduler
            
        return history

    def fit_with_warmup(self, train_loader, num_epochs, valid_loader=None, warmup_epochs=10):
        """Convenience method for training with warmup (common for flow models)"""
        # Simple warmup implementation
        original_lr = self.optimizer.param_groups[0]['lr']
        warmup_lr = original_lr * 0.1
        
        # Warmup phase
        self.log(f" > Warmup phase: {warmup_epochs} epochs")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr
            
        warmup_history = self.fit(train_loader, warmup_epochs, valid_loader)
        
        # Restore learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = original_lr
            
        # Main training phase
        self.log(f" > Main training phase: {num_epochs - warmup_epochs} epochs")
        main_history = self.fit(train_loader, num_epochs - warmup_epochs, valid_loader)
        
        # Combine histories
        combined_history = {}
        for key in warmup_history:
            combined_history[key] = warmup_history[key] + main_history.get(key, [])
            
        return combined_history