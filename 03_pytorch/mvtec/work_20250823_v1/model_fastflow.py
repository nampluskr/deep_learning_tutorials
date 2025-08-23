from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration class for anomaly detection experiments"""
    
    # Experiment configuration
    experiment_name: str = "default_experiment"
    
    # Data configuration
    data_dir: str = '/mnt/d/datasets/mvtec'
    categories: List[str] = field(default_factory=lambda: ['bottle', 'cable', 'capsule'])
    img_size: int = 256
    batch_size: int = 32
    valid_ratio: float = 0.2
    seed: int = 42
    
    # Model configuration
    model_name: str = 'vanilla_ae'  # vanilla_ae, vae, fastflow
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512
    
    # FastFlow specific configuration
    flow_steps: int = 8
    hidden_dim: int = 512
    backbone: str = 'resnet18'
    layers: List[str] = field(default_factory=lambda: ['layer2', 'layer3'])
    weights_path: str = None
    
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
        # Auto-adjust metrics based on model type
        if self.model_name.lower() == 'vae' and 'vae' not in self.metric_names:
            self.metric_names.append('vae')
        elif self.model_name.lower() == 'fastflow':
            self.metric_names = ['fastflow_log_prob', 'fastflow_anomaly_score']
        
        # Auto-adjust loss type based on model
        if self.model_name.lower() == 'vae' and self.loss_type == 'combined':
            self.loss_type = 'vae'
        elif self.model_name.lower() == 'fastflow':
            self.loss_type = 'fastflow'
        
        # Validate categories
        valid_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'
        ]
        
        for category in self.categories:
            if category not in valid_categories:
                raise ValueError(f"Invalid category: {category}. Valid categories: {valid_categories}")
        
        # Generate experiment name if not provided
        if self.experiment_name == "default_experiment":
            categories_str = "_".join(self.categories[:2]) + ("_etc" if len(self.categories) > 2 else "")
            self.experiment_name = f"{self.model_name}_{categories_str}_{self.loss_type}"


def get_default_configs():
    """Get predefined experiment configurations"""
    configs = {}
    
    # Vanilla AutoEncoder configurations
    configs['vanilla_ae_basic'] = Config(
        experiment_name="vanilla_ae_basic",
        model_name="vanilla_ae",
        categories=['bottle'],
        loss_type="mse",
        metric_names=['mse', 'psnr'],
        num_epochs=30,
        learning_rate=1e-4
    )
    
    configs['vanilla_ae_combined'] = Config(
        experiment_name="vanilla_ae_combined",
        model_name="vanilla_ae",
        categories=['bottle'],
        loss_type="combined",
        mse_weight=0.7,
        ssim_weight=0.3,
        metric_names=['mse', 'ssim', 'psnr'],
        num_epochs=50,
        learning_rate=1e-4
    )
    
    # VAE configurations
    configs['vae_basic'] = Config(
        experiment_name="vae_basic",
        model_name="vae",
        categories=['bottle'],
        loss_type="vae",
        beta=1.0,
        mse_weight=1.0,
        metric_names=['mse', 'ssim', 'psnr', 'vae'],
        num_epochs=50,
        learning_rate=1e-4
    )
    
    configs['vae_beta_high'] = Config(
        experiment_name="vae_beta_high",
        model_name="vae",
        categories=['bottle'],
        loss_type="vae",
        beta=4.0,
        mse_weight=1.0,
        metric_names=['mse', 'ssim', 'psnr', 'vae'],
        num_epochs=50,
        learning_rate=1e-4
    )
    
    # FastFlow configurations  
    configs['fastflow_basic'] = Config(
        experiment_name="fastflow_basic",
        model_name="fastflow",
        categories=['bottle'],
        loss_type="fastflow",
        flow_steps=8,
        hidden_dim=512,
        backbone='resnet18',
        layers=['layer2', 'layer3'],
        metric_names=['fastflow_log_prob', 'fastflow_anomaly_score'],
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-3
    )
    
    configs['fastflow_deep'] = Config(
        experiment_name="fastflow_deep",
        model_name="fastflow",
        categories=['bottle'],
        loss_type="fastflow",
        flow_steps=16,
        hidden_dim=1024,
        backbone='resnet18',
        layers=['layer1', 'layer2', 'layer3'],
        metric_names=['fastflow_log_prob', 'fastflow_anomaly_score'],
        num_epochs=50,
        batch_size=8,
        learning_rate=5e-4
    )
    
    # Multi-category configurations
    configs['vanilla_ae_multi'] = Config(
        experiment_name="vanilla_ae_multi_category",
        model_name="vanilla_ae",
        categories=['bottle', 'cable', 'capsule'],
        loss_type="combined",
        mse_weight=0.7,
        ssim_weight=0.3,
        metric_names=['mse', 'ssim', 'psnr'],
        num_epochs=40,
        batch_size=64,
        learning_rate=1e-4
    )
    
    # High resolution configurations
    configs['vanilla_ae_hires'] = Config(
        experiment_name="vanilla_ae_hires",
        model_name="vanilla_ae",
        categories=['bottle'],
        img_size=512,
        loss_type="combined",
        latent_dim=1024,
        metric_names=['mse', 'ssim', 'psnr'],
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-4
    )
    
    return configs


if __name__ == "__main__":
    # Test configuration creation and validation
    print("Testing configuration creation...")
    
    # Test basic config
    config = Config()
    print(f"Default config: {config.experiment_name}")
    print(f"Model: {config.model_name}, Loss: {config.loss_type}")
    print(f"Metrics: {config.metric_names}")
    
    # Test VAE config
    vae_config = Config(
        model_name="vae",
        categories=['bottle'],
        beta=2.0
    )
    print(f"\nVAE config: {vae_config.experiment_name}")
    print(f"Model: {vae_config.model_name}, Loss: {vae_config.loss_type}")
    print(f"Metrics: {vae_config.metric_names}")
    
    # Test FastFlow config
    fastflow_config = Config(
        model_name="fastflow",
        categories=['cable'],
        flow_steps=12
    )
    print(f"\nFastFlow config: {fastflow_config.experiment_name}")
    print(f"Model: {fastflow_config.model_name}, Loss: {fastflow_config.loss_type}")
    print(f"Metrics: {fastflow_config.metric_names}")
    print(f"Flow steps: {fastflow_config.flow_steps}")
    
    # Test predefined configs
    print(f"\n{'-'*50}")
    print("Testing predefined configurations...")
    configs = get_default_configs()
    
    for name, config in configs.items():
        print(f"{name}: {config.model_name} - {config.loss_type}")
        if config.model_name == 'fastflow':
            print(f"  Flow steps: {config.flow_steps}, Hidden dim: {config.hidden_dim}")
        elif config.model_name == 'vae':
            print(f"  Beta: {config.beta}")
    
    print("\nConfiguration testing completed!")