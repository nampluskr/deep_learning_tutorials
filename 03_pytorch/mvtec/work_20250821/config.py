from dataclasses import dataclass, field

@dataclass
class Config:

    # Data configuration
    data_dir: str = '/mnt/d/datasets/mvtec'
    categories: list = field(default_factory=lambda: ['bottle', 'cable', 'capsule'])
    img_size: int = 256
    batch_size: int = 32
    valid_ratio: float = 0.2
    seed: int = 42

    in_channels: int = 3
    out_channels: int = 3
    latent_dim: int = 512


if __name__ == "__main__":
    config = Config()
    print(config)