import torch
from dataclasses import dataclass, asdict, fields
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
    fine_tuning: bool = False            # whether to fine-tune the model
    early_stopping: bool = False         # enable/disable early stopping
    early_stopping_patience: int = 5    # patience for early stopping
    evaluation: bool = False            # whether to evaluate the model after training

    include_keys = {
        "batch_size":    "batch", 
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

def print_config(config, show_all=False):
    """Print configuration settings in a formatted table"""
    header = "All Config Settings" if show_all else "Non-default Config Settings"
    print("=" * 40)
    print(header)
    print("=" * 40)

    default_config = Config()  # create a default instance for comparison

    for f in fields(config):
        key = f.name
        user_value = getattr(config, key)
        default_value = getattr(default_config, key)

        if show_all or user_value != default_value:
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

    suffix = "_".join(parts) if parts else "default"
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