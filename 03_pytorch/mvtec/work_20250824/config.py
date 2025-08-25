import torch
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Any


@dataclass
class DataConfig:
    data_dir: str = '/mnt/d/datasets/mvtec'
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
        "latent_dim": 256,
        "img_size": 256,
    })


@dataclass
class TrainConfig:
    num_epochs: int = 50

    # optimizer
    optimizer_type: str = 'adamw'
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-3,
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

    print("===== Experiment Config =====")
    _print_dict(config_dict, indent=2)
    print("=============================")

if __name__ == "__main__":
    cfg = Config()
    print_config(TrainConfig())
