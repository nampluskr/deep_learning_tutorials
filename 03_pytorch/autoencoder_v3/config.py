import torch
from dataclasses import dataclass, asdict
import json
import yaml

@dataclass
class Config:
    """Configuration class for autoencoder training parameters"""
    
    # Data configuration
    data_dir: str = '/mnt/d/datasets/mvtec'
    category: str = 'bottle'
    batch_size: int = 32
    img_size: int = 256
    normalize: bool = True
    valid_ratio: float = 0.2

    # Model configuration
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512
    model_type: str = 'unet_ae'  # 'vanilla_ae' or 'unet_ae'

    # Training configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42                   # random seed for reproducibility
    num_epochs: int = 10             # number of training epochs
    learning_rate: float = 1e-3      # optimizer learning rate
    weight_decay: float = 1e-5       # optimizer weight decay
    save_model: bool = False         # whether to save the model after training
    save_path: str = 'model.pth'     # path to save the trained model

    def __post_init__(self):
        """Initialize model_params after other attributes are set"""
        self.model_params = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'latent_dim': self.latent_dim
        }

    def __init__(self, **kwargs):
        # Set default values first
        self.data_dir = '/mnt/d/datasets/mvtec'
        self.category = 'bottle'
        self.batch_size = 32
        self.img_size = 256
        self.normalize = True
        self.valid_ratio = 0.2
        self.in_channels = 3
        self.out_channels = 3
        self.latent_dim = 512
        self.model_type = 'unet_ae'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_epochs = 10
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.save_model = False
        self.save_path = 'model.pth'
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'.")
        
        # Initialize model_params after all attributes are set
        self.model_params = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'latent_dim': self.latent_dim
        }



    def update_model_params(self):
        """Update model_params dictionary when model configuration changes"""
        self.model_params = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'latent_dim': self.latent_dim
        }


def print_config(config):
    """Print configuration settings in a formatted table"""
    print("=" * 40)
    print("Config Settings")
    print("=" * 40)
    for key, value in config.__dict__.items():
        if key != 'model_params':  # Skip model_params to avoid redundancy
            print(f"{key:<15}: {value}")
    print("=" * 40)


if __name__ == "__main__":
    # Example usage
    config = Config(model_type='unet_ae', num_epochs=5)
    print_config(config)

    # Save to JSON
    with open('config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)

    # Save to YAML
    with open('config.yaml', 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)