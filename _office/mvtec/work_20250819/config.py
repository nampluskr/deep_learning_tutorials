"""
Configuration management for anomaly detection experiments
Handles experiment configuration, saving, and loading
"""

import torch
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional
import json
import yaml


@dataclass
class Config:
    """Configuration class for autoencoder training parameters"""

    # =====================================================================
    # Data configuration
    # =====================================================================
    # data_dir: str = '/home/namu/myspace/NAMU/datasets/mvtec'
    data_dir: str = r"E:\datasets\MVTec_AD\archive"
    category: str = 'bottle'
    batch_size: int = 32
    img_size: int = 256
    valid_ratio: float = 0.2

    # =====================================================================
    # Model configuration
    # =====================================================================
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512
    model_type: str = ""  # (ex) 'vanilla_ae' or 'unet_ae'

    # =====================================================================
    # Training configuration
    # =====================================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42                      # random seed for reproducibility
    num_epochs: int = 10                # number of training epochs
    learning_rate: float = 1e-3         # optimizer learning rate
    weight_decay: float = 1e-5          # optimizer weight decay
    loss_type: str = "combined"         # loss function type

    # =====================================================================
    # Saving configuration
    # =====================================================================
    save_model: bool = False            # whether to save the model after training
    save_log: bool = False              # whether to save the training log
    model_path: str = ""                # path to save the trained model
    config_path: str = ""               # path to save the configuration

    # =====================================================================
    # Early Stopping configuration
    # =====================================================================
    fine_tuning: bool = False           # whether to fine-tune the model
    early_stop: bool = False        # enable/disable early stopping
    early_stop_patience: int = 5    # patience for early stopping
    evaluation: bool = False            # whether to evaluate the model after training

    include_keys: Dict[str, str] = None
    exclude_keys: List[str] = None

    def __post_init__(self):
        if self.include_keys is None:
            self.include_keys = {              
                "batch_size":    "batch",
                "img_size":      "size",
                "latent_dim":    "latent",
                "in_channels":   "in",
                "out_channels":  "out",
                "seed":          "seed",
                "learning_rate": "lr",
                "weight_decay":  "decay",
                "loss_type":     "loss",
                "category":      "cat",
                "valid_ratio":   "val",
            }
            self.exclude_keys = [
                "model_path", 
                "config_path", 
                "include_keys",
                "exclude_keys"
            ]


def load_config(config_path):
    """Load configuration file (json)"""
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    return Config(**cfg_dict)


def show_config(config, show_all=False):
    """Print configuration settings in a formatted table"""
    header = "All Config Settings" if show_all else "Non-default Config Settings"
    print("=" * 40)
    print(header)
    print("=" * 40)
    # create a default instance for comparison
    default_config = Config()

    for f in fields(config):
        key = f.name
        user_value = getattr(config, key)
        default_value = getattr(default_config, key)

        if show_all or user_value != default_value:
            if key not in config.exclude_keys:
                print(f"{key:<20}: {user_value}")

    print("=" * 40)


def get_config_prefix(config):
    """Generate prefix string with non-default config values (except save options)"""
    default_config = Config()
    parts = []
    for f in fields(config):
        key = f.name
        if key in config.include_keys.keys():
            user_value = getattr(config, key)
            default_value = getattr(default_config, key)
            if user_value != default_value:
                parts.append(f"{config.include_keys[key]}-{user_value}")

    suffix = "_".join(parts) if parts else "preset"
    return f"{config.model_type}_{config.category}_{suffix}"


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
