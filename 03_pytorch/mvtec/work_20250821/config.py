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
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.valid_ratio < 0 or self.valid_ratio > 1:
            raise ValueError("valid_ratio must be between 0 and 1")
        
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not self.categories:
            raise ValueError("categories list cannot be empty")
    
    def get_experiment_name(self):
        """Generate experiment name based on configuration"""
        categories_str = '_'.join(self.categories)
        return f"{categories_str}_{self.loss_type}_{self.img_size}_{self.latent_dim}"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'data_dir': self.data_dir,
            'categories': self.categories,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'valid_ratio': self.valid_ratio,
            'seed': self.seed,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'latent_dim': self.latent_dim,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'loss_type': self.loss_type,
            'mse_weight': self.mse_weight,
            'ssim_weight': self.ssim_weight,
            'beta': self.beta,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'restore_best_weights': self.restore_best_weights,
            'metric_names': self.metric_names,
        }


@dataclass
class VAEConfig(Config):
    """Specialized configuration for VAE experiments"""
    
    # Override defaults for VAE
    loss_type: str = 'vae'
    beta: float = 1.0
    learning_rate: float = 1e-4
    optimizer_type: str = 'adam'
    scheduler_type: str = 'cosine'
    metric_names: List[str] = field(default_factory=lambda: ['vae', 'mse', 'ssim'])


@dataclass
class FastFlowConfig(Config):
    """Specialized configuration for FastFlow experiments"""
    loss_type: str = 'fastflow'
    learning_rate: float = 1e-3
    optimizer_type: str = 'adam'
    scheduler_type: str = 'reduce_plateau'  # <-- step → reduce_plateau 로 변경
    metric_names: List[str] = field(default_factory=lambda: ['fastflow_log_prob', 'fastflow_anomaly_score'])
    
    # FastFlow specific parameters
    backbone: str = 'resnet18'
    layers: List[str] = field(default_factory=lambda: ['layer2', 'layer3'])
    flow_steps: int = 8
    hidden_dim: int = 512
    weights_path: str = None


@dataclass  
class SingleCategoryConfig(Config):
    """Configuration for single category experiments"""
    
    # Override for single category
    categories: List[str] = field(default_factory=lambda: ['bottle'])
    batch_size: int = 16  # Smaller batch for single category
    num_epochs: int = 30


def get_default_configs():
    """Get a list of default configurations for different experiments"""
    
    configs = {
        'vanilla_ae_multi': Config(
            categories=['bottle', 'cable', 'capsule'],
            loss_type='combined',
            mse_weight=0.7,
            ssim_weight=0.3,
            num_epochs=50
        ),
        
        'vanilla_ae_single': SingleCategoryConfig(
            categories=['bottle'],
            num_epochs=30
        ),
        
        'vae_multi': VAEConfig(
            categories=['bottle', 'cable', 'capsule'],
            beta=1.0,
            num_epochs=50
        ),
        
        'vae_single': VAEConfig(
            categories=['bottle'],
            beta=0.5,
            num_epochs=30
        ),
        
        'fastflow_multi': FastFlowConfig(
            categories=['bottle', 'cable', 'capsule'],
            backbone='resnet18',
            layers=['layer2', 'layer3'],
            flow_steps=8,
            num_epochs=50
        ),
        
        'fastflow_single': FastFlowConfig(
            categories=['bottle'],
            backbone='resnet18',
            layers=['layer2', 'layer3'],
            flow_steps=6,
            num_epochs=30
        ),
        
        'quick_test': Config(
            categories=['bottle'],
            img_size=128,
            batch_size=8,
            num_epochs=5,
            latent_dim=256
        ),
        
        'fastflow_quick_test': FastFlowConfig(
            categories=['bottle'],
            img_size=128,
            batch_size=8,
            num_epochs=5,
            backbone='resnet18',
            layers=['layer2'],
            flow_steps=4,
            hidden_dim=256
        )
    }
    
    return configs


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration classes...")
    
    # Default config
    config = Config()
    print(f"Default config: {config}")
    print(f"Experiment name: {config.get_experiment_name()}")
    
    # VAE config
    vae_config = VAEConfig()
    print(f"\nVAE config: {vae_config}")
    print(f"Experiment name: {vae_config.get_experiment_name()}")
    
    # Get all default configs
    configs = get_default_configs()
    print(f"\nAvailable default configs: {list(configs.keys())}")
    
    # Test config validation
    try:
        invalid_config = Config(valid_ratio=1.5)  # Should raise error
    except ValueError as e:
        print(f"\nValidation working: {e}")
    
    print("\nConfiguration test completed!")