```python
from dataloader import get_dataloaders
from types import SimpleNamespace
import os

def get_config(category="tile"):
    config = SimpleNamespace(
        # data_root="/home/namu/myspace/NAMU/datasets/mvtec",
        data_root=r"E:\datasets\mvtec",
        category=category,
        img_size=256,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,

        model_type="autoencoder",
        backbone_dir=r"D:\Non_Documents\2025\1_project\1_image_processing\modeling\mvtec_office\backbones",
        num_epochs=100,
        learning_rate= 1e-4,
        latent_dim=512,
        output_dir="./experiments"
    )
    config.weight_path = os.path.join(config.output_dir,
        f"model_{config.category}_{config.model_type}_epochs-{config.num_epochs}.pth")
    os.makedirs(os.path.dirname(config.weight_path), exist_ok=True)
    return config
```

```python
from model_ae import AutoEncoder
from trainer import BaseTrainer

config = get_config()
train_loader, test_loader = get_dataloaders(config)

model = AutoEncoder(backbone="resnet18")
trainer = BaseTrainer(model)
history = trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader, output_dir=config.output_dir)
trainer.evaluate(test_loader)
```
