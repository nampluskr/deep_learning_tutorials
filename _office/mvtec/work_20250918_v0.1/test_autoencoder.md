## 신규 코드 테스트
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


## 이전 모델 테스트

```python
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from autoencoder_ref import get_dataloaders, AutoEncoder
from autoencoder_ref import SSIMMetric, validate, get_thresholds, evaluate_thresholds
from sklearn.metrics import roc_auc_score, average_precision_score

def train(model, loader, loss_fn, optimizer, metric_fn, device):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    for inputs in loader:
        images = inputs['image'].to(device)

        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = loss_fn(reconstructed, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            total_metric += metric_fn(reconstructed, images) * images.size(0)

    return total_loss / len(loader.dataset), total_metric / len(loader.dataset)


def run_experiement(config):
    print(f"\n*** RUN EXPERIMENT: {config.model_type.upper()} - {config.category.upper()}")

    train_loader, test_loader = get_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(latent_dim=config.latent_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    metric_fn = SSIMMetric()

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_metric = train(model, train_loader, loss_fn, optimizer, metric_fn, device)
        scores, labels = validate(model, test_loader, device)

        auroc = roc_auc_score(labels, scores)
        aupr  = average_precision_score(labels, scores)

        print(f"Epoch [{epoch}/{config.num_epochs}] loss={train_loss:.4f}, ssim={train_metric:.4f}"
              f" | (val) auroc={auroc:.4f}, aupr={aupr:.4f}")

    print("\n...Training finished.")

    scores, labels = validate(model, test_loader, device)
    thresholds = get_thresholds(scores, labels)
    results = evaluate_thresholds(scores, labels, thresholds)
    print(results)
```

```python
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
        num_epochs=10,
        learning_rate= 1e-4,
        latent_dim=512,
        output_dir="./experiments"
    )
    config.weight_path = os.path.join(config.output_dir,
        f"model_{config.category}_{config.model_type}_epochs-{config.num_epochs}.pth")
    os.makedirs(os.path.dirname(config.weight_path), exist_ok=True)
    return config

config = get_config(category="tile")
run_experiement(config)
```

```python
from trainer import BaseTrainer

train_loader, test_loader = get_dataloaders(config)
model = AutoEncoder(latent_dim=config.latent_dim)
trainer = BaseTrainer(model)

trainer.fit(train_loader, num_epochs=10)
# trainer.evaluate(test_loader) # Error
```

```python
scores, labels = validate(model, test_loader, device=torch.device("cuda"))
thresholds = get_thresholds(scores, labels)
results = evaluate_thresholds(scores, labels, thresholds)
print(results)
```
