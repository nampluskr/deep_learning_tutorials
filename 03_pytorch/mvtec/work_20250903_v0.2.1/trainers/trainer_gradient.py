import torch
from .trainer_base import BaseTrainer


class GradientTrainer(BaseTrainer):
    """Trainer for gradient-based anomaly detection models (AE, STFPM, DRAEM, VAE, etc.)."""
    
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        """Initialize gradient trainer with modeler and training components."""
        super().__init__(modeler, optimizer, scheduler, stopper, logger)

    @property
    def trainer_type(self):
        """Return trainer type identifier for gradient-based models."""
        return "gradient"

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Gradient-based training with backpropagation."""
        self.log(f"\n > Gradient Training: {self.modeler.learning_type} learning")
        self.log(f" > Model: {type(self.modeler.model).__name__}")
        if hasattr(self.modeler.model, 'teacher_model'):
            self.log(f" > Teacher-Student Architecture (STFPM)")
        
        # Call parent's fit method which uses training_step and validation_step
        history = super().fit(train_loader, num_epochs, valid_loader)
        
        return history

    def fit_with_early_stopping(self, train_loader, max_epochs, valid_loader, patience=10, min_delta=1e-4):
        """Convenience method for training with early stopping."""
        from .trainer_base import EarlyStopper
        
        original_stopper = self.stopper
        self.stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best_weights=True)
        
        try:
            history = self.fit(train_loader, max_epochs, valid_loader)
        finally:
            self.stopper = original_stopper
            
        return history

    def fit_with_lr_scheduling(self, train_loader, num_epochs, valid_loader=None, 
                              scheduler_type='plateau', **scheduler_params):
        """Convenience method for training with learning rate scheduling."""
        import torch.optim.lr_scheduler as lr_scheduler
        
        # Create scheduler
        scheduler_map = {
            'plateau': lr_scheduler.ReduceLROnPlateau,
            'step': lr_scheduler.StepLR,
            'cosine': lr_scheduler.CosineAnnealingLR,
            'exponential': lr_scheduler.ExponentialLR,
        }
        
        if scheduler_type not in scheduler_map:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        original_scheduler = self.scheduler
        
        # Set default parameters
        if scheduler_type == 'plateau':
            params = {'patience': 5, 'factor': 0.5, 'verbose': True}
        elif scheduler_type == 'step':
            params = {'step_size': 30, 'gamma': 0.1}
        elif scheduler_type == 'cosine':
            params = {'T_max': num_epochs}
        elif scheduler_type == 'exponential':
            params = {'gamma': 0.9}
        else:
            params = {}
        
        params.update(scheduler_params)
        self.scheduler = scheduler_map[scheduler_type](self.optimizer, **params)
        
        try:
            history = self.fit(train_loader, num_epochs, valid_loader)
        finally:
            self.scheduler = original_scheduler
            
        return history

    def get_model_stats(self):
        """Get general statistics about the model."""
        stats = {}
        
        # Basic model statistics
        total_params = sum(p.numel() for p in self.modeler.model.parameters())
        trainable_params = sum(p.numel() for p in self.modeler.model.parameters() if p.requires_grad)
        
        stats.update({
            'model_name': type(self.modeler.model).__name__,
            'modeler_name': type(self.modeler).__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'learning_type': self.modeler.learning_type,
            'device': str(self.modeler.device),
        })
        
        # Loss function info
        if self.modeler.loss_fn:
            stats['loss_function'] = type(self.modeler.loss_fn).__name__
        
        # Metrics info
        stats['metrics'] = list(self.modeler.metrics.keys())
        
        # Model-specific stats for STFPM
        if hasattr(self.modeler.model, 'teacher_model') and hasattr(self.modeler.model, 'student_model'):
            teacher_params = sum(p.numel() for p in self.modeler.model.teacher_model.parameters())
            teacher_trainable = sum(p.numel() for p in self.modeler.model.teacher_model.parameters() if p.requires_grad)
            student_params = sum(p.numel() for p in self.modeler.model.student_model.parameters())
            student_trainable = sum(p.numel() for p in self.modeler.model.student_model.parameters() if p.requires_grad)
            
            stats.update({
                'architecture_type': 'teacher_student',
                'teacher_total_params': teacher_params,
                'teacher_trainable_params': teacher_trainable,
                'student_total_params': student_params,
                'student_trainable_params': student_trainable,
                'backbone_architecture': getattr(self.modeler.model, 'backbone', 'unknown'),
            })
        else:
            stats['architecture_type'] = 'single_model'
        
        return stats

    def validate_gradient_setup(self):
        """Validate that gradient-based model is set up correctly."""
        issues = []
        
        # Check that model has trainable parameters
        trainable_params = sum(p.numel() for p in self.modeler.model.parameters() if p.requires_grad)
        if trainable_params == 0:
            issues.append("No trainable parameters in model")
        
        # Check that optimizer has parameters
        if self.optimizer is None:
            issues.append("No optimizer configured")
        elif hasattr(self.optimizer, 'param_groups'):
            optimizer_params = sum(len(group['params']) for group in self.optimizer.param_groups)
            if optimizer_params == 0:
                issues.append("Optimizer has no parameters")
            
        # Check loss function
        if self.modeler.loss_fn is None:
            issues.append("No loss function defined")
            
        # Model-specific validation for teacher-student models (STFPM)
        if hasattr(self.modeler.model, 'teacher_model') and hasattr(self.modeler.model, 'student_model'):
            teacher_trainable = any(p.requires_grad for p in self.modeler.model.teacher_model.parameters())
            student_trainable = any(p.requires_grad for p in self.modeler.model.student_model.parameters())
            
            if teacher_trainable:
                issues.append("Teacher model parameters are trainable (should be frozen for STFPM)")
            if not student_trainable:
                issues.append("Student model parameters are frozen (should be trainable for STFPM)")
            
            # Check optimizer targets student only
            if self.optimizer and hasattr(self.optimizer, 'param_groups'):
                optimizer_param_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
                student_param_ids = {id(p) for p in self.modeler.model.student_model.parameters() if p.requires_grad}
                if optimizer_param_ids != student_param_ids:
                    issues.append("Optimizer is not targeting student model parameters exclusively")
        
        if issues:
            self.log("Gradient Model Setup Issues:")
            for issue in issues:
                self.log(f" - {issue}")
            return False
        else:
            self.log(" > Gradient model setup validated successfully")
            return True

    def analyze_training_progress(self, history):
        """Analyze gradient-based training progress from history."""
        analysis = {}
        
        if 'loss' in history and len(history['loss']) > 0:
            losses = history['loss']
            analysis['loss_trend'] = {
                'initial_loss': losses[0],
                'final_loss': losses[-1],
                'loss_reduction': losses[0] - losses[-1],
                'loss_reduction_ratio': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0.0,
                'min_loss': min(losses),
                'converged': abs(losses[-1] - losses[-2]) < 1e-6 if len(losses) > 1 else False,
            }
        
        # Analyze validation loss if available
        if 'val_loss' in history and len(history['val_loss']) > 0:
            val_losses = history['val_loss']
            analysis['val_loss_trend'] = {
                'initial_val_loss': val_losses[0],
                'final_val_loss': val_losses[-1],
                'min_val_loss': min(val_losses),
                'best_val_epoch': val_losses.index(min(val_losses)) + 1,
                'overfitting': val_losses[-1] > min(val_losses) * 1.1,  # Simple overfitting detection
            }
        
        # Analyze metrics trends
        for metric_name in self.modeler.get_metric_names():
            if metric_name in history and len(history[metric_name]) > 0:
                values = history[metric_name]
                analysis[f'{metric_name}_trend'] = {
                    'initial': values[0],
                    'final': values[-1],
                    'best': max(values) if values else 0,
                    'improved': values[-1] > values[0] if values else False,
                }
            
            # Also check validation metrics
            val_metric_key = f'val_{metric_name}'
            if val_metric_key in history and len(history[val_metric_key]) > 0:
                val_values = history[val_metric_key]
                analysis[f'{val_metric_key}_trend'] = {
                    'initial': val_values[0],
                    'final': val_values[-1],
                    'best': max(val_values) if val_values else 0,
                    'best_epoch': val_values.index(max(val_values)) + 1 if val_values else 0,
                    'improved': val_values[-1] > val_values[0] if val_values else False,
                }
        
        return analysis


if __name__ == "__main__":
    pass