import torch
import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration for dataset loading and preprocessing"""
    
    # Dataset paths
    data_dir: str = '/mnt/d/datasets/mvtec'
    category: Union[str, List[str]] = 'bottle'
    
    # Data loading
    batch_size: int = 32
    valid_ratio: float = 0.2
    num_workers: int = 4
    
    # Image preprocessing
    img_size: int = 256
    normalize: bool = True
    augmentation_level: str = 'medium'  # 'light', 'medium', 'heavy', 'oled'
    
    # Advanced options
    load_masks: bool = False  # Load ground truth masks for test set
    cache_images: bool = False  # Cache images in memory for faster loading
    multi_category_training: bool = False  # Enable multi-category training
    
    # Custom normalization (computed from dataset if None)
    custom_mean: Optional[List[float]] = None  # [R, G, B] channel means
    custom_std: Optional[List[float]] = None   # [R, G, B] channel stds


@dataclass 
class ModelConfig:
    """Model architecture configuration"""
    
    # Basic model settings
    model_type: str = 'vanilla_ae'  # See get_model() for available types
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512
    
    # Pretrained model settings (for ResNet/VGG/EfficientNet based models)
    backbone_arch: str = 'resnet50'  # resnet50, vgg16, efficientnet_b0, etc.
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # VAE specific settings
    beta: float = 4.0  # β parameter for β-VAE
    lambda_reg: float = 10.0  # λ parameter for WAE
    
    # Advanced model options
    use_attention: bool = False  # Add attention mechanisms
    use_skip_connections: bool = True  # For UNet-style models
    dropout_rate: float = 0.0  # Dropout probability
    
    # Model initialization
    init_type: str = 'normal'  # 'normal', 'xavier', 'kaiming', 'orthogonal'
    init_gain: float = 0.02


@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Basic training settings
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Loss function
    loss_type: str = 'combined'  # See metrics_functional.py for available types
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'l1': 0.7, 'ssim': 0.3})
    
    # Optimizer settings
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    momentum: float = 0.9  # For SGD
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])  # For Adam/AdamW
    
    # Learning rate scheduling
    scheduler_type: str = 'plateau'  # 'plateau', 'step', 'cosine', 'exponential'
    lr_patience: int = 5  # For ReduceLROnPlateau
    lr_factor: float = 0.5  # LR reduction factor
    step_size: int = 10  # For StepLR
    gamma: float = 0.1  # For StepLR and ExponentialLR
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    
    # Training stability
    gradient_clipping: bool = False
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Validation frequency
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection specific configuration"""
    
    # Evaluation methods
    evaluation_methods: List[str] = field(default_factory=lambda: ['mse', 'l1', 'ssim'])
    threshold_percentiles: List[float] = field(default_factory=lambda: [90, 95, 99])
    
    # Reconstruction error computation
    default_method: str = 'mse'  # Default method for anomaly scoring
    normalize_scores: bool = True
    score_normalization: str = 'minmax'  # 'minmax', 'zscore', 'robust'
    
    # Advanced evaluation
    pixel_level_evaluation: bool = False  # Requires ground truth masks
    compute_auroc: bool = True
    compute_aupr: bool = True
    compute_f1: bool = True
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_method: str = 'percentile'  # 'percentile', 'otsu', 'gmm'
    
    # Ensemble evaluation (future use)
    ensemble_methods: List[str] = field(default_factory=list)
    ensemble_weights: List[float] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Experiment management configuration"""
    
    # Experiment identification
    experiment_name: str = 'mvtec_anomaly_detection'
    run_name: Optional[str] = None  # Auto-generated if None
    tags: List[str] = field(default_factory=list)
    
    # Paths
    output_dir: str = './experiments'
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    
    # Logging
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Checkpointing
    save_model: bool = True
    save_best_only: bool = True
    save_last: bool = True
    save_optimizer: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False  # CUDNN benchmark mode
    
    # Visualization
    save_plots: bool = True
    plot_format: str = 'png'  # 'png', 'pdf', 'svg'
    save_reconstructions: bool = True
    num_reconstruction_samples: int = 10
    
    # Progress tracking
    use_tqdm: bool = True
    tqdm_ncols: int = 100


@dataclass
class OLEDConfig:
    """OLED display specific configuration"""
    
    # OLED characteristics
    display_type: str = 'oled'  # 'oled', 'lcd', 'general'
    
    # Resolution settings
    native_resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    support_multi_resolution: bool = True
    min_resolution: int = 256
    max_resolution: int = 4096
    
    # OLED-specific preprocessing
    gamma_correction: bool = True
    gamma_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    
    # Defect characteristics
    defect_types: List[str] = field(default_factory=lambda: [
        'mura', 'stain', 'bright_spot', 'dark_spot', 'line_defect'
    ])
    
    # Sensitivity settings
    sensitivity_level: str = 'high'  # 'low', 'medium', 'high', 'ultra'
    detection_threshold_adjustment: float = 0.9  # Multiplier for threshold
    
    # Post-processing
    apply_morphological_ops: bool = True
    morphological_kernel_size: int = 3
    min_defect_area: int = 10  # Minimum defect area in pixels


@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    oled: OLEDConfig = field(default_factory=OLEDConfig)
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_ids: List[int] = field(default_factory=lambda: [0] if torch.cuda.is_available() else [])
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Auto-generate run name if not provided
        if self.experiment.run_name is None:
            self.experiment.run_name = f"{self.model.model_type}_{self.data.category}_{self.experiment.experiment_name}"
        
        # Validate model-specific settings
        self._validate_model_config()
        
        # Setup paths
        self._setup_paths()
        
        # Validate consistency
        self._validate_config()
    
    def _validate_model_config(self):
        """Validate model-specific configuration"""
        # VAE models need special loss handling
        if 'vae' in self.model.model_type.lower():
            if self.training.loss_type == 'combined':
                print("Warning: VAE models typically use reconstruction + KL loss")
        
        # Pretrained models should use appropriate input size
        pretrained_models = ['resnet_ae', 'vgg_ae', 'efficientnet_ae']
        if self.model.model_type in pretrained_models:
            if self.data.img_size < 224:
                print(f"Warning: {self.model.model_type} typically works better with img_size >= 224")
    
    def _setup_paths(self):
        """Setup experiment paths"""
        base_path = Path(self.experiment.output_dir) / self.experiment.run_name
        
        # Create directories
        self.experiment.log_dir = str(base_path / 'logs')
        self.experiment.checkpoint_dir = str(base_path / 'checkpoints')
        
        # Create directories if they don't exist
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.checkpoint_dir, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration consistency"""
        # Check data consistency
        if isinstance(self.data.category, list) and not self.data.multi_category_training:
            print("Warning: Multiple categories specified but multi_category_training is False")
        
        # Check OLED-specific settings
        if self.oled.display_type == 'oled':
            if self.data.augmentation_level != 'oled':
                print("Info: Consider using augmentation_level='oled' for OLED displays")
        
        # Check evaluation settings
        if self.anomaly_detection.pixel_level_evaluation and not self.data.load_masks:
            print("Warning: Pixel-level evaluation requires load_masks=True")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save(self, path: str, format: str = 'yaml'):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        if format.lower() == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        path = Path(path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Create config object from dictionary
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        # Extract sub-configs
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        anomaly_config = AnomalyDetectionConfig(**config_dict.get('anomaly_detection', {}))
        experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))
        oled_config = OLEDConfig(**config_dict.get('oled', {}))
        
        # Create main config
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            anomaly_detection=anomaly_config,
            experiment=experiment_config,
            oled=oled_config,
            device=config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            gpu_ids=config_dict.get('gpu_ids', [0] if torch.cuda.is_available() else [])
        )
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                if hasattr(getattr(self, key), '__dict__'):
                    # Update sub-config
                    sub_config = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    # Update main config attribute
                    setattr(self, key, value)


def get_preset_config(preset_name: str) -> Config:
    """Get predefined configuration presets"""
    
    if preset_name == 'vanilla_baseline':
        return Config(
            model=ModelConfig(model_type='vanilla_ae', latent_dim=512),
            training=TrainingConfig(num_epochs=20, learning_rate=1e-3),
            data=DataConfig(category='bottle', img_size=256, normalize=True)
        )
    
    elif preset_name == 'unet_advanced':
        return Config(
            model=ModelConfig(model_type='unet_ae', latent_dim=512, use_attention=True),
            training=TrainingConfig(num_epochs=30, learning_rate=1e-4, use_amp=True),
            data=DataConfig(category='bottle', img_size=512, augmentation_level='medium')
        )
    
    elif preset_name == 'vae_disentangled':
        return Config(
            model=ModelConfig(model_type='beta_vae', latent_dim=256, beta=4.0),
            training=TrainingConfig(num_epochs=50, learning_rate=5e-4),
            data=DataConfig(category=['bottle', 'cable'], multi_category_training=True)
        )
    
    elif preset_name == 'resnet_pretrained':
        return Config(
            model=ModelConfig(
                model_type='resnet_ae', backbone_arch='resnet50', 
                pretrained=True, freeze_backbone=False
            ),
            training=TrainingConfig(num_epochs=25, learning_rate=1e-4),
            data=DataConfig(img_size=224, normalize=True)
        )
    
    elif preset_name == 'oled_optimized':
        return Config(
            model=ModelConfig(model_type='unet_ae', latent_dim=512),
            training=TrainingConfig(num_epochs=40, learning_rate=1e-4),
            data=DataConfig(
                img_size=512, normalize=False, 
                augmentation_level='oled', cache_images=True
            ),
            oled=OLEDConfig(
                display_type='oled', 
                native_resolution=[1920, 1080],
                sensitivity_level='high'
            ),
            anomaly_detection=AnomalyDetectionConfig(
                evaluation_methods=['mse', 'ssim'],
                threshold_percentiles=[95, 98, 99.5]
            )
        )
    
    elif preset_name == 'multi_category_ensemble':
        return Config(
            model=ModelConfig(model_type='vanilla_ae', latent_dim=1024),
            training=TrainingConfig(num_epochs=35, learning_rate=1e-3),
            data=DataConfig(
                category=['bottle', 'cable', 'capsule', 'carpet', 'grid'],
                multi_category_training=True,
                img_size=256
            ),
            anomaly_detection=AnomalyDetectionConfig(
                evaluation_methods=['mse', 'l1', 'ssim'],
                ensemble_methods=['mse', 'ssim'],
                ensemble_weights=[0.7, 0.3]
            )
        )
    
    elif preset_name == 'debug_fast':
        return Config(
            model=ModelConfig(model_type='vanilla_ae', latent_dim=128),
            training=TrainingConfig(num_epochs=3, learning_rate=1e-3),
            data=DataConfig(category='bottle', img_size=128, batch_size=8),
            experiment=ExperimentConfig(save_model=False, save_plots=False)
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: "
                        f"vanilla_baseline, unet_advanced, vae_disentangled, "
                        f"resnet_pretrained, oled_optimized, multi_category_ensemble, debug_fast")


def print_config(config: Config, sections: Optional[List[str]] = None):
    """Print configuration in a formatted way"""
    
    if sections is None:
        sections = ['data', 'model', 'training', 'anomaly_detection', 'experiment', 'oled']
    
    print("=" * 80)
    print(f"Configuration: {config.experiment.run_name}")
    print("=" * 80)
    
    for section in sections:
        if hasattr(config, section):
            section_config = getattr(config, section)
            print(f"\n[{section.upper()}]")
            print("-" * 40)
            
            for key, value in section_config.__dict__.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"{key:<25}: {value[:3]}... ({len(value)} items)")
                elif isinstance(value, dict) and len(value) > 3:
                    print(f"{key:<25}: {list(value.keys())[:3]}... ({len(value)} keys)")
                else:
                    print(f"{key:<25}: {value}")
    
    print("\n" + "=" * 80)


def create_experiment_configs(base_config: Config, param_grid: Dict[str, List[Any]]) -> List[Config]:
    """Create multiple experiment configurations from parameter grid"""
    
    configs = []
    
    def _recursive_grid(grid, current_params=None):
        if current_params is None:
            current_params = {}
        
        if not grid:
            # Create config with current parameters
            config = Config(**asdict(base_config))
            config.update_from_dict(current_params)
            configs.append(config)
            return
        
        # Get next parameter
        param_name = list(grid.keys())[0]
        param_values = grid[param_name]
        remaining_grid = {k: v for k, v in grid.items() if k != param_name}
        
        for value in param_values:
            new_params = current_params.copy()
            
            # Handle nested parameters (e.g., 'model.latent_dim')
            if '.' in param_name:
                parts = param_name.split('.')
                if parts[0] not in new_params:
                    new_params[parts[0]] = {}
                new_params[parts[0]][parts[1]] = value
            else:
                new_params[param_name] = value
            
            _recursive_grid(remaining_grid, new_params)
    
    _recursive_grid(param_grid)
    return configs


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Test basic config creation
    print("\n1. Testing basic config:")
    config = Config()
    print_config(config, sections=['data', 'model'])
    
    # Test preset configs
    print("\n2. Testing preset configs:")
    preset_names = ['vanilla_baseline', 'oled_optimized', 'debug_fast']
    
    for preset in preset_names:
        print(f"\n--- {preset.upper()} ---")
        preset_config = get_preset_config(preset)
        print(f"Model: {preset_config.model.model_type}")
        print(f"Epochs: {preset_config.training.num_epochs}")
        print(f"Image size: {preset_config.data.img_size}")
        print(f"Category: {preset_config.data.category}")
    
    # Test save/load
    print("\n3. Testing save/load:")
    config = get_preset_config('vanilla_baseline')
    config.save('test_config.yaml')
    loaded_config = Config.load('test_config.yaml')
    print(f"Original run name: {config.experiment.run_name}")
    print(f"Loaded run name: {loaded_config.experiment.run_name}")
    
    # Clean up
    if os.path.exists('test_config.yaml'):
        os.remove('test_config.yaml')
    
    # Test parameter grid
    print("\n4. Testing parameter grid:")
    base_config = get_preset_config('debug_fast')
    param_grid = {
        'model.latent_dim': [128, 256, 512],
        'training.learning_rate': [1e-3, 1e-4],
        'data.img_size': [128, 256]
    }
    
    experiment_configs = create_experiment_configs(base_config, param_grid)
    print(f"Generated {len(experiment_configs)} experiment configurations")
    
    for i, exp_config in enumerate(experiment_configs[:3]):  # Show first 3
        print(f"  Config {i+1}: latent_dim={exp_config.model.latent_dim}, "
              f"lr={exp_config.training.learning_rate}, "
              f"img_size={exp_config.data.img_size}")
    
    print("\nConfiguration system test completed!")