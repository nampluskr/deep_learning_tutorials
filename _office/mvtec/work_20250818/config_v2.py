"""
Configuration management for anomaly detection experiments
Handles experiment configuration, saving, and loading
"""

from dataclasses import dataclass, replace
import json
import os
from datetime import datetime


@dataclass
class Config:
    """Configuration class for anomaly detection experiments"""
    # Data parameters
    data_dir: str = "/path/to/mvtec"
    category: str = "bottle"
    img_size: int = 256
    batch_size: int = 16
    valid_ratio: float = 0.2
    normalize: bool = False
    num_workers: int = 4

    # Model parameters
    model_type: str = "vanilla_ae"  # ["vanilla_ae", "unet_ae"]
    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512

    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    loss_type: str = "mse"  # ["mse", "combined", "ssim", "bce"]
    metric_names: List[str] = ["psnr", "ssim"]

    # Evaluation parameters
    anomaly_method: str = "mse"  # ["mse", "ssim", "combined"]
    threshold_percentile: int = 95

    # System parameters
    device: str = "cuda"
    seed: int = 42

    # Experiment parameters
    save_model: bool = True
    save_log: bool = True
    model_save_dir: str = "./results"
    log_save_dir: str = "./results"
    config_save_dir: str = "./results"
    fine_tuning: bool = False
    evaluation: bool = True
    experiment_name: str = ""

    # predefined key list
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
                "loss_type":     "loss",
                "category":      "cat",
                "valid_ratio":   "val",
            }
            self.exclude_keys = [
                "model_save_dir",
                "config_save_dir",
                "log_save_dir",
                "include_keys",
                "exclude_keys"
            ]


def print_config(config, show_all=False):
    """Print configuration parameters"""
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


def save_config(config, config_path):
    """Save configuration to JSON file"""
    os.makedirs(config.config_save_dir, exist_ok=True)
    with open('config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    return Config(**cfg_dict)


def get_experiment_name(config):
    """Generate experiment name based on config and timestamp"""
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
