from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration class for anomaly detection experiments"""
    
    # Data configuration
    data_dir: str = '/mnt/d/datasets/mvtec'
    categories: List[str] = field(default_factory=lambda: ['bottle', 'cable', 'capsule'])
    img_size: int = 256
    batch_size: int = 32
    valid_ratio: float = 0.2
    seed: int = 42
    
    # Model configuration
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512
    
    # Training configuration
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer_type: str = 'adamw'
    scheduler_type: str = 'reduce_plateau'
    
    # Loss configuration
    loss_type: str = 'combined'
    mse_weight: float = 0.7
    ssim_weight: float = 0.3
    beta: float = 1.0  # For VAE
    
    # Early stopping configuration
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    
    # Metric configuration
    metric_names: List[str] = field(default_factory=lambda: ['mse', 'ssim', 'psnr'])


if __name__ == "__main__":
    pass