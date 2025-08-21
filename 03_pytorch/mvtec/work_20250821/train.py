import torch
import torch.nn as nn
from tqdm import tqdm
import sys
from pytorch_msssim import ssim


def train_epoch(model, train_loader, optimizer, loss_fn, metrics={}):
    """Train model for one epoch"""
    device = next(model.parameters()).device
    model.train()

    # Initialize results tracking
    loss_names = ['total']
    metric_names = list(metrics.keys())
    results = {name: 0.0 for name in loss_names + metric_names}

    with tqdm(train_loader, desc="Training", leave=False, file=sys.stdout,
              dynamic_ncols=True, ncols=120, ascii=True) as pbar:

        for cnt, data in enumerate(pbar):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            # Use only normal samples for training in anomaly detection
            normal_mask = labels == 0
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]

            optimizer.zero_grad()
            outputs = model(normal_images)
            losses = model.compute_loss(outputs, loss_fn_dict)
            total_loss = losses['total']

            total_loss.backward()
            optimizer.step()

            for loss_name, loss_value in losses.items():
                if loss_name in results:
                    results[loss_name] += loss_value.item()

            for metric_name, metric_fn in metrics.items():
                if metric_name in results:
                    metric_value = metric_fn(outputs)
                    results[metric_name] += metric_value.item()

            # Update progress bar
            pbar.set_postfix({k: f"{v/(cnt + 1):.3f}" for k, v in results.items()})

    return {k: v/len(train_loader) for k, v in results.items()}


@torch.no_grad()
def validate_epoch(model, valid_loader, loss_fn_dict, metrics={}):
    """Validate model for one epoch"""
    device = next(model.parameters()).device
    model.eval()

    # Initialize results tracking
    loss_names = list(loss_fn_dict.keys()) + ['total']
    metric_names = list(metrics.keys())
    results = {name: 0.0 for name in loss_names + metric_names}

    with tqdm(valid_loader, desc="Validation", leave=False, file=sys.stdout,
              dynamic_ncols=True, ncols=120, ascii=True) as pbar:

        for cnt, data in enumerate(pbar):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            # Use only normal samples for validation in anomaly detection
            normal_mask = labels == 0
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]

            # Forward pass - get dictionary output
            outputs = model(normal_images)

            # Compute losses using model's compute_loss method
            losses = model.compute_loss(outputs, loss_fn_dict)

            # Update loss results
            for loss_name, loss_value in losses.items():
                if loss_name in results:
                    results[loss_name] += loss_value.item()

            # Calculate metrics
            for metric_name, metric_fn in metrics.items():
                if metric_name in results:
                    metric_value = metric_fn(outputs)
                    results[metric_name] += metric_value.item() if torch.is_tensor(metric_value) else metric_value

            # Update progress bar
            pbar.set_postfix({k: f"{v/(cnt + 1):.3f}" for k, v in results.items()})

    return {k: v/len(valid_loader) for k, v in results.items()}


class Trainer:
    """Trainer wrapper class for anomaly detection models"""

    def __init__(self, model, optimizer, loss_fn_dict, metrics={}, logger=None):
        """
        Initialize trainer

        Args:
            model: The model to train
            optimizer: Optimizer for training
            loss_fn_dict: Dictionary of loss functions with weights
            metrics: Dictionary of metric functions
            logger: Logger instance for logging (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn_dict = loss_fn_dict
        self.metrics = metrics
        self.logger = logger
        self.device = next(model.parameters()).device

        # Initialize history tracking
        loss_names = list(loss_fn_dict.keys()) + ['total']
        metric_names = list(metrics.keys())
        self.history = {name: [] for name in loss_names + metric_names}
        self.history.update({f"val_{k}": [] for k in self.history.keys()})

    def log(self, message, level='info'):
        """
        Log message using logger or print to console

        Args:
            message: Message to log
            level: Log level ('info', 'debug', 'warning', 'error')
        """
        if self.logger:
            # Log to file with level and timestamp (StreamHandler disabled for clean terminal output)
            if level == 'info':
                self.logger.info(message)
            elif level == 'debug':
                self.logger.debug(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            else:
                self.logger.info(message)

            # Print clean message to terminal (without timestamp and level)
            print(message)
        else:
            # Print to console with clean message only
            print(message)

    def fit(self, train_loader, num_epochs, valid_loader=None):
        """
        Fit the model using external train/validate functions

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            valid_loader: Validation data loader (optional)

        Returns:
            Dictionary containing training history
        """
        self.log(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            # Training using external function
            train_results = train_epoch(self.model, train_loader, self.optimizer,
                                      self.loss_fn_dict, self.metrics)

            # Store training results
            for name, value in train_results.items():
                if name in self.history:
                    self.history[name].append(value)

            # Validation using external function
            if valid_loader is not None:
                valid_results = validate_epoch(self.model, valid_loader,
                                             self.loss_fn_dict, self.metrics)

                # Store validation results
                for name, value in valid_results.items():
                    val_name = f"val_{name}"
                    if val_name in self.history:
                        self.history[val_name].append(value)

                # Log epoch summary
                train_summary = self._format_results(train_results)
                valid_summary = self._format_results(valid_results, prefix="val_")
                self.log(f"[Epoch {epoch:2d}/{num_epochs}] {train_summary} | {valid_summary}")
            else:
                # Log epoch summary (training only)
                train_summary = self._format_results(train_results)
                self.log(f"[Epoch {epoch:2d}/{num_epochs}] {train_summary}")

        self.log("Training completed!")
        return self.history

    def train_single_epoch(self, train_loader):
        """Train for a single epoch using external function"""
        return train_epoch(self.model, train_loader, self.optimizer,
                          self.loss_fn_dict, self.metrics)

    def validate_single_epoch(self, valid_loader):
        """Validate for a single epoch using external function"""
        return validate_epoch(self.model, valid_loader,
                            self.loss_fn_dict, self.metrics)

    def _format_results(self, results, prefix=""):
        """Format results for printing"""
        model_type = getattr(self.model, 'model_type', 'unknown')

        if model_type == 'vanilla_ae':
            key_metrics = ['total', 'ssim', 'psnr']
        elif model_type == 'vae':
            key_metrics = ['total', 'reconstruction', 'kl_divergence', 'ssim']
        elif model_type == 'fastflow':
            key_metrics = ['total', 'nll', 'log_prob']
        elif model_type == 'stfpm':
            key_metrics = ['total', 'feature_matching', 'feature_mag']
        elif model_type == 'padim':
            key_metrics = ['total', 'feature_mag']
        else:
            key_metrics = ['total', 'ssim', 'psnr']

        formatted = []
        for k, v in results.items():
            if k in key_metrics:
                formatted.append(f"{prefix}{k}={v:.3f}")
        return ', '.join(formatted)


# Metric functions that work with dictionary outputs
def compute_ssim_metric(outputs):
    """Compute SSIM between reconstructed and input images"""
    if ('reconstructed' in outputs and 'input' in outputs and
        isinstance(outputs['reconstructed'], torch.Tensor) and
        isinstance(outputs['input'], torch.Tensor)):
        try:
            return ssim(outputs['reconstructed'], outputs['input'], data_range=1.0)
        except Exception as e:
            print(f"Warning: SSIM computation failed: {e}")
            return torch.tensor(0.0)
    else:
        return torch.tensor(0.0)


def compute_psnr_metric(outputs):
    """Compute PSNR between reconstructed and input images"""
    if ('reconstructed' in outputs and 'input' in outputs and
        isinstance(outputs['reconstructed'], torch.Tensor) and
        isinstance(outputs['input'], torch.Tensor)):
        try:
            mse = nn.MSELoss()(outputs['reconstructed'], outputs['input'])
            if mse == 0:
                return torch.tensor(float('inf'))
            return 10 * torch.log10(1.0 ** 2 / mse)
        except Exception as e:
            print(f"Warning: PSNR computation failed: {e}")
            return torch.tensor(0.0)
    else:
        return torch.tensor(0.0)


def compute_reconstruction_error(outputs):
    """Compute reconstruction error (MSE)"""
    if ('reconstructed' in outputs and 'input' in outputs and
        isinstance(outputs['reconstructed'], torch.Tensor) and
        isinstance(outputs['input'], torch.Tensor)):
        try:
            return nn.MSELoss()(outputs['reconstructed'], outputs['input'])
        except Exception as e:
            print(f"Warning: Reconstruction error computation failed: {e}")
            return torch.tensor(0.0)
    else:
        return torch.tensor(0.0)


def compute_feature_magnitude(outputs):
    """Compute average magnitude of features"""
    if 'features' in outputs:
        features = outputs['features']

        # Handle different feature types
        if isinstance(features, list):
            # Multi-scale features (e.g., FastFlow, STFPM)
            total_magnitude = 0
            total_elements = 0

            for feat in features:
                if isinstance(feat, torch.Tensor):
                    magnitude = torch.mean(torch.abs(feat))
                    total_magnitude += magnitude
                    total_elements += 1

            if total_elements > 0:
                return total_magnitude / total_elements
            else:
                return torch.tensor(0.0)

        elif isinstance(features, torch.Tensor):
            # Single feature tensor (e.g., VAE, Vanilla AE)
            return torch.mean(torch.abs(features))
        else:
            return torch.tensor(0.0)
    else:
        return torch.tensor(0.0)


def compute_log_prob_metric(outputs):
    """Compute average log probability for flow-based models"""
    if 'log_probs' in outputs and isinstance(outputs['log_probs'], list):
        try:
            total_log_prob = 0
            total_elements = 0

            for log_prob in outputs['log_probs']:
                if isinstance(log_prob, torch.Tensor):
                    # Average over spatial dimensions
                    avg_log_prob = torch.mean(log_prob)
                    total_log_prob += avg_log_prob
                    total_elements += 1

            if total_elements > 0:
                return total_log_prob / total_elements
            else:
                return torch.tensor(0.0)

        except Exception as e:
            print(f"Warning: Log probability computation failed: {e}")
            return torch.tensor(0.0)
    else:
        return torch.tensor(0.0)


def compute_nll_metric(outputs):
    """Compute negative log likelihood for flow-based models"""
    log_prob = compute_log_prob_metric(outputs)
    return -log_prob


def compute_latent_magnitude(outputs):
    """Compute average magnitude of latent representation"""
    # Handle different latent representations
    latent_keys = ['latent', 'z', 'mu']

    for key in latent_keys:
        if key in outputs:
            latent = outputs[key]

            if isinstance(latent, list):
                # Multi-scale latent features
                total_magnitude = 0
                total_elements = 0

                for lat in latent:
                    if isinstance(lat, torch.Tensor):
                        magnitude = torch.mean(torch.abs(lat))
                        total_magnitude += magnitude
                        total_elements += 1

                if total_elements > 0:
                    return total_magnitude / total_elements

            elif isinstance(latent, torch.Tensor):
                # Single latent tensor
                return torch.mean(torch.abs(latent))

    return torch.tensor(0.0)
    """Compute average magnitude of latent representation"""
    # Handle different latent representations
    latent_keys = ['latent', 'z', 'mu']

    for key in latent_keys:
        if key in outputs:
            latent = outputs[key]

            if isinstance(latent, list):
                # Multi-scale latent features
                total_magnitude = 0
                total_elements = 0

                for lat in latent:
                    if isinstance(lat, torch.Tensor):
                        magnitude = torch.mean(torch.abs(lat))
                        total_magnitude += magnitude
                        total_elements += 1

                if total_elements > 0:
                    return total_magnitude / total_elements

            elif isinstance(latent, torch.Tensor):
                # Single latent tensor
                return torch.mean(torch.abs(latent))

    return torch.tensor(0.0)


# Utility functions for creating loss configurations
def create_vanilla_ae_loss_config(reconstruction_loss='mse', reconstruction_weight=1.0):
    """Create loss configuration for vanilla autoencoder"""
    from model import ReconstructionLoss

    return {
        'reconstruction': {
            'fn': ReconstructionLoss(reconstruction_loss),
            'weight': reconstruction_weight
        }
    }


def create_vae_loss_config(reconstruction_loss='mse', reconstruction_weight=1.0, kl_weight=0.1):
    """Create loss configuration for VAE"""
    from model import ReconstructionLoss, KLDivergenceLoss

    return {
        'reconstruction': {
            'fn': ReconstructionLoss(reconstruction_loss),
            'weight': reconstruction_weight
        },
        'kl_divergence': {
            'fn': KLDivergenceLoss(),
            'weight': kl_weight
        }
    }


def create_memory_bank_loss_config(distance_weight=1.0):
    """Create loss configuration for memory bank models"""
    from model import MemoryDistanceLoss

    return {
        'memory_distance': {
            'fn': MemoryDistanceLoss(),
            'weight': distance_weight
        }
    }


def create_padim_loss_config():
    """Create loss configuration for PaDiM (no gradient-based training)"""
    return {
        'dummy': {
            'fn': lambda x, y: torch.tensor(0.0, requires_grad=True),
            'weight': 1.0
        }
    }


def create_fastflow_loss_config(nll_weight=1.0):
    """Create loss configuration for FastFlow"""
    # FastFlow uses its own compute_loss method, so we provide a dummy config
    return {
        'nll': {
            'fn': lambda pred, target: torch.tensor(0.0, requires_grad=True),
            'weight': nll_weight
        }
    }


def create_stfpm_loss_config(feature_weights=None):
    """Create loss configuration for STFPM"""
    from stfpm_model import STFPMLoss

    return {
        'feature_matching': {
            'fn': STFPMLoss(feature_weights),
            'weight': 1.0
        }
    }


# Utility function for creating metric configurations
def create_standard_metrics():
    """Create standard metrics for anomaly detection models"""
    return {
        'ssim': compute_ssim_metric,
        'psnr': compute_psnr_metric,
        'recon_error': compute_reconstruction_error,
        'feature_mag': compute_feature_magnitude,
        'latent_mag': compute_latent_magnitude,
        'log_prob': compute_log_prob_metric,
        'nll': compute_nll_metric
    }
