import torch
from dataclasses import dataclass, asdict
import json
import yaml

@dataclass
class Config:
    ## Data
    data_dir = '/mnt/d/datasets/mvtec'
    category = 'bottle'
    batch_size = 32
    img_size = 256
    normalize = True
    valid_ratio = 0.2

    ## Modeling
    in_channels = 3
    out_channels = 3
    latent_dim = 512

    ## Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42                   # random seed for reproducibility
    model_type = 'unet_ae'      # 'vanilla_ae' or 'unet_ae'
    num_epochs = 10             # number of training epochs
    learning_rate = 1e-3        # optimizer learning rate
    weight_decay = 1e-5         # optimizer weight decay

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'.")


def print_config(config):
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