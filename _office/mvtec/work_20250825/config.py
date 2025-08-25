import os
import torch
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Any
import json
import yaml


@dataclass
class DataConfig:
    # data_dir: str = '/mnt/d/datasets/mvtec'
    data_dir: str = '/home/namu/myspace/NAMU/datasets/mvtec'
    categories: List[str] = field(default_factory=lambda: ['bottle'])
    img_size: int = 256
    batch_size: int = 32
    valid_ratio: float = 0.2


@dataclass
class ModelConfig:
    model_type: str = "vanilla_ae"
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "in_channels": 3,
        "out_channels": 3,
        "latent_dim": 512,
        "img_size": 256,
    })


@dataclass
class TrainConfig:
    num_epochs: int = 50

    # optimizer
    optimizer_type: str = 'adamw'
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-4,
        "weight_decay": 1e-5
    })

    # scheduler
    scheduler_type: str = 'plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "min",
        "factor": 0.5,
        "patience": 5
    })

    # stopper
    stopper_type: str = 'stop'
    stopper_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_epoch": 50
    })


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./experiments"
    save_model: bool = False
    save_config: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def print_config(config):

    def _print_dict(d, prefix="", indent=2):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                _print_dict(value, prefix + " " * indent)
            else:
                print(f"{prefix}{key}: {value}")

    if is_dataclass(config):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("config must be a dataclass or dict")

    print("\n===== Experiment Config =====")
    _print_dict(config_dict, indent=2)
    print("=============================")


def save_config(config, output_dir, filename):
    """Save config dataclass to json or yaml in output_dir/filename"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    if is_dataclass(config):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("Config must be a dataclass or dict")

    def serialize(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        return obj

    if filename.endswith(".json"):
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4, default=serialize)
    elif filename.endswith((".yml", ".yaml")):
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError("Unsupported file extension. Use .json or .yaml.")

    print(f"[INFO] Config saved to {path}")


def load_config(output_dir, filename):
    """Load config file (json/yaml) from output_dir/filename and return Config dataclass"""
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.endswith(".json"):
        with open(path, "r") as f:
            cfg_dict = json.load(f)
    elif path.endswith((".yml", ".yaml")):
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .yaml")

    config = Config(
        seed=cfg_dict.get("seed", 42),
        device=cfg_dict.get("device", "cuda"),
        output_dir=cfg_dict.get("output_dir", "./experiments"),
        data=DataConfig(**cfg_dict.get("data", {})),
        model=ModelConfig(**cfg_dict.get("model", {})),
        train=TrainConfig(**cfg_dict.get("train", {})),
    )

    print(f"[INFO] Config loaded from {path}")
    return config


if __name__ == "__main__":
    cfg = Config()
    print_config(TrainConfig())
