import torch
from dataclasses import dataclass, asdict
import json
import yaml

@dataclass
class Config:
    """Configuration class for autoencoder training parameters"""
    
    # =====================================================================
    # Data configuration
    # =====================================================================
    data_dir: str = '/mnt/d/datasets/mvtec'
    category: str = 'bottle'
    batch_size: int = 32
    img_size: int = 256
    normalize: bool = True
    valid_ratio: float = 0.2

    # =====================================================================
    # Model configuration
    # =====================================================================
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512
    model_type: str = 'unet_ae'  # 'vanilla_ae' or 'unet_ae'

    # =====================================================================
    # Training configuration
    # =====================================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42                   # random seed for reproducibility
    num_epochs: int = 10             # number of training epochs
    learning_rate: float = 1e-3      # optimizer learning rate
    weight_decay: float = 1e-5       # optimizer weight decay
    loss_type: str = 'anomaly_detection'  # loss function type
    save_model: bool = False         # whether to save the model after training
    save_path: str = 'model.pth'     # path to save the trained model


def print_config(config):
    """Print configuration settings in a formatted table"""
    print("=" * 40)
    print("Config Settings")
    print("=" * 40)
    for key, value in config.__dict__.items():
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