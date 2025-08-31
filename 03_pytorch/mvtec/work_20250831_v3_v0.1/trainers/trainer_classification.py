import torch
from tqdm import tqdm
from time import time

from .trainer_base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification-based anomaly detection models (CutPaste, GeomAD, etc.)"""
    
    def __init__(self, modeler, scheduler=None, stopper=None, logger=None):
        super().__init__(modeler, scheduler, stopper, logger)

    @property
    def trainer_type(self):
        return "classification"

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Two-stage training: Classification training + Post-processing (density modeling)"""
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metric_names)})

        self.log("\n > Stage 1: Classification Training started...")

        # Stage 1: Binary classification training (Normal vs Augmented)
        for epoch in range(1, num_epochs + 1):
            start_time = time()
            train_results = self.run_epoch(train_loader, epoch, num_epochs, mode='train')
            
            # Format training info
            train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

            # Update training history
            for key, value in train_results.items():
                if key in history:
                    history[key].append(value)

            # Validation phase
            valid_results = {}
            if valid_loader is not None:
                valid_results = self.run_epoch(valid_loader, epoch, num_epochs, mode='valid')
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

        # Stage 2: Post-processing (Feature extraction + Density modeling)
        self.log("\n > Stage 2: Post-processing started...")
        start_time = time()
        
        if hasattr(self.modeler, 'fit_density_model'):
            self.modeler.fit_density_model(train_loader)
            elapsed_time = time() - start_time
            self.log(f" > Density model fitting completed ({elapsed_time:.1f}s)")
        elif hasattr(self.modeler, 'fit_post_classifier'):
            self.modeler.fit_post_classifier(train_loader)
            elapsed_time = time() - start_time
            self.log(f" > Post-classification fitting completed ({elapsed_time:.1f}s)")
        else:
            self.log(" > No post-processing required")

        self.log(" > Training completed!")
        return history

    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Run a single epoch for classification training"""
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        
        # Classification-specific metrics
        total_accuracy = 0.0
        total_pos_samples = 0  # Augmented/anomaly samples
        total_neg_samples = 0  # Normal samples
        correct_pos = 0
        correct_neg = 0
        num_batches = 0

        desc = f"{mode.capitalize()} [{epoch}/{num_epochs}]"
        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                if mode == 'train':
                    batch_results = self.modeler.train_step(inputs, self.optimizer)
                else:
                    batch_results = self.modeler.validate_step(inputs)

                total_loss += batch_results['loss']
                
                # Classification accuracy metrics
                if 'accuracy' in batch_results:
                    total_accuracy += batch_results['accuracy']
                if 'pos_samples' in batch_results:
                    total_pos_samples += batch_results['pos_samples']
                    total_neg_samples += batch_results['neg_samples']
                    correct_pos += batch_results['correct_pos']
                    correct_neg += batch_results['correct_neg']
                
                # Standard metrics
                for metric_name in self.modeler.metrics.keys():
                    if metric_name in batch_results:
                        total_metrics[metric_name] += batch_results[metric_name]
                
                num_batches += 1

                avg_loss = total_loss / num_batches
                avg_accuracy = total_accuracy / num_batches if total_accuracy > 0 else 0.0
                avg_metrics = {name: total_metrics[name] / num_batches for name in self.metric_names}

                # Progress bar update
                progress_info = {
                    'loss': f"{avg_loss:.3f}",
                    'acc': f"{avg_accuracy:.3f}",
                }
                
                # Add specific classification metrics
                if total_pos_samples + total_neg_samples > 0:
                    pos_acc = correct_pos / total_pos_samples if total_pos_samples > 0 else 0.0
                    neg_acc = correct_neg / total_neg_samples if total_neg_samples > 0 else 0.0
                    progress_info.update({
                        'pos_acc': f"{pos_acc:.3f}",
                        'neg_acc': f"{neg_acc:.3f}",
                    })
                
                # Add custom metrics
                for name, value in avg_metrics.items():
                    if value != 0.0:
                        progress_info[name] = f"{value:.3f}"
                
                pbar.set_postfix(progress_info)

        # Prepare results
        results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        results.update({name: total_metrics[name] / num_batches for name in self.metric_names})
        
        # Add classification-specific results
        if total_accuracy > 0:
            results['accuracy'] = total_accuracy / num_batches
        
        if total_pos_samples + total_neg_samples > 0:
            results.update({
                'pos_accuracy': correct_pos / total_pos_samples if total_pos_samples > 0 else 0.0,
                'neg_accuracy': correct_neg / total_neg_samples if total_neg_samples > 0 else 0.0,
                'total_pos_samples': total_pos_samples,
                'total_neg_samples': total_neg_samples,
            })
        
        return results

    def fit_with_early_stopping(self, train_loader, max_epochs, valid_loader, patience=10, min_delta=1e-4):
        """Convenience method for training with early stopping"""
        from .trainer_base import EarlyStopper
        
        original_stopper = self.stopper
        self.stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best_weights=True)
        
        try:
            history = self.fit(train_loader, max_epochs, valid_loader)
        finally:
            self.stopper = original_stopper
            
        return history

    def fit_with_lr_scheduling(self, train_loader, num_epochs, valid_loader=None, 
                              scheduler_type='step', **scheduler_params):
        """Convenience method for training with learning rate scheduling"""
        from .trainer_base import get_scheduler
        
        original_scheduler = self.scheduler
        self.scheduler = get_scheduler(self.optimizer, scheduler_type, **scheduler_params)
        
        try:
            history = self.fit(train_loader, num_epochs, valid_loader)
        finally:
            self.scheduler = original_scheduler
            
        return history

    def get_classification_stats(self):
        """Get classification training statistics"""
        if hasattr(self.modeler, 'get_classification_stats'):
            return self.modeler.get_classification_stats()
        else:
            return {'classification_stats': 'not_available'}

    def get_density_model_stats(self):
        """Get density model statistics if available"""
        if hasattr(self.modeler, 'get_density_stats'):
            return self.modeler.get_density_stats()
        elif hasattr(self.modeler, 'get_gmm_stats'):
            return self.modeler.get_gmm_stats()
        else:
            return {'density_model': 'not_fitted'}

    def evaluate_classifier_only(self, test_loader):
        """Evaluate only the classification performance (without density modeling)"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        desc = "Evaluate Classifier"
        with tqdm(test_loader, desc=desc, leave=False, ascii=True) as pbar:
            for inputs in pbar:
                with torch.no_grad():
                    if hasattr(self.modeler, 'predict_classification'):
                        predictions = self.modeler.predict_classification(inputs)
                        labels = inputs.get('augmentation_label', inputs.get('label'))
                        
                        all_predictions.append(predictions.cpu())
                        all_labels.append(labels.cpu())

        if all_predictions:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            labels_tensor = torch.cat(all_labels, dim=0)
            return predictions_tensor, labels_tensor
        else:
            return None, None

    def set_augmentation_params(self, **params):
        """Set parameters for data augmentation during training"""
        if hasattr(self.modeler, 'set_augmentation_params'):
            self.modeler.set_augmentation_params(**params)

    def get_feature_extractor(self):
        """Get feature extractor for analysis"""
        if hasattr(self.modeler, 'get_feature_extractor'):
            return self.modeler.get_feature_extractor()
        else:
            return None

    def extract_features(self, data_loader):
        """Extract features from data using trained classifier"""
        if hasattr(self.modeler, 'extract_features'):
            return self.modeler.extract_features(data_loader)
        else:
            self.log("Warning: Feature extraction not available for this model")
            return None

    def validate_two_stage_learning(self):
        """Validate that the model supports two-stage learning"""
        stage1_supported = hasattr(self.modeler, 'train_step') and hasattr(self.modeler, 'validate_step')
        stage2_supported = hasattr(self.modeler, 'fit_density_model') or hasattr(self.modeler, 'fit_post_classifier')
        
        if not stage1_supported:
            raise ValueError("Model does not support classification training (Stage 1)")
        
        if not stage2_supported:
            self.log("Warning: Model does not support post-processing (Stage 2)")
        
        return stage1_supported and stage2_supported

    def get_augmentation_stats(self):
        """Get statistics about data augmentation if available"""
        if hasattr(self.modeler, 'get_augmentation_stats'):
            return self.modeler.get_augmentation_stats()
        else:
            return {'augmentation_stats': 'not_available'}