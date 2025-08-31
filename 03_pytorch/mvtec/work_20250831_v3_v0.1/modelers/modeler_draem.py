import torch
from torch import optim

from .modeler_base import BaseModeler


class DraemModeler(BaseModeler):
    def __init__(self, model, loss_fn, metrics=None, device=None, enable_sspcab=False, sspcab_lambda=0.1):
        super().__init__(model, loss_fn, metrics, device)
        
        # SSPCAB support
        self.enable_sspcab = enable_sspcab
        self.sspcab_lambda = sspcab_lambda
        self.sspcab_activations = {}
        
        if self.enable_sspcab:
            self.setup_sspcab_hooks()
            self.sspcab_loss = torch.nn.MSELoss()

    def setup_sspcab_hooks(self):
        """Set up SSPCAB forward hooks for activation capture."""
        def get_activation(name: str):
            def hook(module, input, output):
                self.sspcab_activations[name] = output
            return hook

        # Register hooks on the reconstructive subnetwork
        if hasattr(self.model, 'reconstructive_subnetwork'):
            # Hook on mp4 (input to block5)
            self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(
                get_activation("input")
            )
            # Hook on block5 (output)
            self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(
                get_activation("output")
            )

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        optimizer.zero_grad()
        
        # Forward pass - model handles anomaly generation in training mode
        outputs = self.model(inputs['image'])
        
        # DRAEM training mode returns dict with multiple outputs
        loss = self.loss_fn(
            outputs['input_image'],
            outputs['reconstruction'], 
            outputs['anomaly_mask'],
            outputs['prediction']
        )
        
        # Add SSPCAB loss if enabled
        if self.enable_sspcab and 'input' in self.sspcab_activations and 'output' in self.sspcab_activations:
            sspcab_loss = self.sspcab_loss(
                self.sspcab_activations['input'],
                self.sspcab_activations['output']
            )
            loss += self.sspcab_lambda * sspcab_loss
        
        loss.backward()
        optimizer.step()

        results = {'loss': loss.item()}
        
        # Additional training metrics
        with torch.no_grad():
            # Reconstruction quality metrics
            reconstruction = outputs['reconstruction']
            input_image = outputs['input_image']
            
            # Ensure tensors are in correct range [0, 1] and format
            reconstruction = torch.clamp(reconstruction, 0, 1)
            input_image = torch.clamp(input_image, 0, 1)
            
            for metric_name, metric_fn in self.metrics.items():
                if metric_name in ['ssim', 'psnr', 'lpips']:
                    try:
                        metric_value = metric_fn(reconstruction, input_image)
                        # Check for NaN or infinity
                        if torch.isnan(torch.tensor(metric_value)) or torch.isinf(torch.tensor(metric_value)):
                            results[metric_name] = 0.0
                        else:
                            results[metric_name] = float(metric_value)
                    except Exception as e:
                        results[metric_name] = 0.0
            
            # Anomaly detection accuracy on synthetic data
            if 'prediction' in outputs and 'anomaly_mask' in outputs:
                prediction = outputs['prediction']
                anomaly_mask = outputs['anomaly_mask']
                
                # Convert to prediction probabilities
                pred_probs = torch.softmax(prediction, dim=1)
                pred_anomaly_map = pred_probs[:, 1, ...]  # anomaly class
                
                # Convert masks to binary
                binary_mask = (anomaly_mask.squeeze(1) > 0.5).float()
                binary_pred = (pred_anomaly_map > 0.5).float()
                
                # Calculate pixel-wise accuracy
                pixel_accuracy = ((binary_mask == binary_pred).float()).mean()
                results['pixel_accuracy'] = pixel_accuracy.item()
                
                # Calculate anomaly region coverage
                if binary_mask.sum() > 0:
                    anomaly_coverage = (binary_pred * binary_mask).sum() / binary_mask.sum()
                    results['anomaly_coverage'] = anomaly_coverage.item()
                
                # SSPCAB loss if enabled
                if self.enable_sspcab and 'input' in self.sspcab_activations:
                    results['sspcab_loss'] = sspcab_loss.item() if 'sspcab_loss' in locals() else 0.0

        return results

    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        # Inference mode forward pass
        predictions = self.model(inputs['image'])
        
        results = {'loss': 0.0}  # DRAEM doesn't have validation loss in inference mode
        
        if hasattr(predictions, 'pred_score'):
            scores = predictions.pred_score
            labels = inputs['label']
            
            # Score distribution analysis
            normal_mask = labels == 0
            anomaly_mask = labels == 1

            normal_scores = scores[normal_mask] if normal_mask.any() else torch.tensor([0.0])
            anomaly_scores = scores[anomaly_mask] if anomaly_mask.any() else torch.tensor([0.0])

            results.update({
                'score_mean': scores.mean().item(),
                'score_std': scores.std().item(),
                'normal_mean': normal_scores.mean().item(),
                'anomaly_mean': anomaly_scores.mean().item(),
                'separation': (anomaly_scores.mean() - normal_scores.mean()).item() if anomaly_mask.any() and normal_mask.any() else 0.0,
            })
            
            # Anomaly map statistics
            if hasattr(predictions, 'anomaly_map'):
                anomaly_map = predictions.anomaly_map
                results.update({
                    'anomaly_map_mean': anomaly_map.mean().item(),
                    'anomaly_map_max': anomaly_map.max().item(),
                    'anomaly_map_min': anomaly_map.min().item(),
                })

        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        predictions = self.model(inputs['image'])
        
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback if inference mode not working properly
            return torch.zeros(inputs['image'].shape[0], device=self.device)

    def compute_anomaly_scores(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            predictions = self.model(inputs['image'])
            
            if hasattr(predictions, 'anomaly_map') and hasattr(predictions, 'pred_score'):
                return {
                    'anomaly_maps': predictions.anomaly_map,
                    'pred_scores': predictions.pred_score
                }
            else:
                # Fallback computation
                batch_size = inputs['image'].shape[0]
                image_size = inputs['image'].shape[-2:]
                
                return {
                    'anomaly_maps': torch.zeros(batch_size, *image_size, device=self.device),
                    'pred_scores': torch.zeros(batch_size, device=self.device)
                }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler like in original DRAEM."""
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.0001,
        )
    
    def configure_optimizers_with_scheduler(self):
        """Configure optimizer with MultiStepLR scheduler."""
        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=0.0001,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[400, 600], 
            gamma=0.1
        )
        return optimizer, scheduler

    @property
    def learning_type(self):
        return "one_class"

    @property
    def trainer_arguments(self):
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }
    
    def get_sspcab_stats(self):
        """Get SSPCAB activation statistics if enabled."""
        if not self.enable_sspcab:
            return {'sspcab_enabled': False}
        
        stats = {'sspcab_enabled': True}
        if 'input' in self.sspcab_activations:
            input_act = self.sspcab_activations['input']
            stats.update({
                'input_act_mean': input_act.mean().item(),
                'input_act_std': input_act.std().item(),
            })
        
        if 'output' in self.sspcab_activations:
            output_act = self.sspcab_activations['output']
            stats.update({
                'output_act_mean': output_act.mean().item(),
                'output_act_std': output_act.std().item(),
            })
        
        return stats
    
    def get_anomaly_generation_stats(self):
        """Get statistics about synthetic anomaly generation."""
        if hasattr(self.model, 'anomaly_generator'):
            return {
                'generator_type': 'SimpleAnomalyGenerator',
                'beta_range': self.model.anomaly_generator.beta_range,
                'image_size': self.model.anomaly_generator.image_size,
            }
        return {}
    
    def set_anomaly_generation_params(self, **params):
        """Set parameters for anomaly generation."""
        if hasattr(self.model, 'anomaly_generator'):
            for key, value in params.items():
                if hasattr(self.model.anomaly_generator, key):
                    setattr(self.model.anomaly_generator, key, value)
    
    def get_model_components(self):
        """Get individual model components for analysis."""
        components = {}
        if hasattr(self.model, 'reconstructive_subnetwork'):
            components['reconstructive'] = self.model.reconstructive_subnetwork
        if hasattr(self.model, 'discriminative_subnetwork'):
            components['discriminative'] = self.model.discriminative_subnetwork
        if hasattr(self.model, 'anomaly_generator'):
            components['anomaly_generator'] = self.model.anomaly_generator
        return components
    
    def freeze_reconstructive(self):
        """Freeze reconstructive network weights."""
        if hasattr(self.model, 'reconstructive_subnetwork'):
            for param in self.model.reconstructive_subnetwork.parameters():
                param.requires_grad = False
    
    def unfreeze_reconstructive(self):
        """Unfreeze reconstructive network weights."""
        if hasattr(self.model, 'reconstructive_subnetwork'):
            for param in self.model.reconstructive_subnetwork.parameters():
                param.requires_grad = True
    
    def freeze_discriminative(self):
        """Freeze discriminative network weights."""
        if hasattr(self.model, 'discriminative_subnetwork'):
            for param in self.model.discriminative_subnetwork.parameters():
                param.requires_grad = False
    
    def unfreeze_discriminative(self):
        """Unfreeze discriminative network weights."""
        if hasattr(self.model, 'discriminative_subnetwork'):
            for param in self.model.discriminative_subnetwork.parameters():
                param.requires_grad = True
    
    def get_reconstruction_quality(self, inputs):
        """Get reconstruction quality metrics on normal images."""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            # Only use normal images
            normal_mask = inputs['label'] == 0
            if not normal_mask.any():
                return {}
            
            normal_images = inputs['image'][normal_mask]
            
            # Get reconstruction (need to access subnetwork directly for inference)
            reconstruction = self.model.reconstructive_subnetwork(normal_images)
            
            quality_metrics = {}
            for metric_name, metric_fn in self.metrics.items():
                if metric_name in ['ssim', 'psnr', 'lpips']:
                    metric_value = metric_fn(reconstruction, normal_images)
                    quality_metrics[f'normal_{metric_name}'] = float(metric_value)
            
            return quality_metrics
    
    def enable_sspcab_mode(self, enable=True, lambda_val=0.1):
        """Enable/disable SSPCAB training mode."""
        self.enable_sspcab = enable
        self.sspcab_lambda = lambda_val
        
        if enable and not hasattr(self, 'sspcab_loss'):
            self.sspcab_loss = torch.nn.MSELoss()
            self.setup_sspcab_hooks()
    
    def clear_sspcab_activations(self):
        """Clear stored SSPCAB activations to free memory."""
        self.sspcab_activations.clear()