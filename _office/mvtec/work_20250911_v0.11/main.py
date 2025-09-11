import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim


from dataloaders import get_dataloaders
from model_ae import VanillaAE
from trainer import BaseTrainer
from utils import show_evaluation, show_statistics


def get_config():
    config = SimpleNamespace(
        # data_dir="/mnt/d/datasets/mvtec",   # WSL
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category="grid",
        model_type="none",
        batch_size=8,
        img_size=512,
        latent_dim=1024,
        num_epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    config.output_dir = f"results_{config.category}"
    os.makedirs(config.output_dir, exist_ok=True)
    return config


if __name__ == "__main__":

    config = get_config()

    # 1. Data loaders    
    train_loader, test_loader = get_dataloaders(
        config.data_dir, config.category, config.batch_size, config.img_size)

    # 2. Model
    model = VanillaAE(latent_dim=config.latent_dim).to(config.device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = BaseTrainer(model, optimizer, loss_fn)
    history = trainer.fit(train_loader, num_epochs=config.num_epochs)

    scores, labels = trainer.predict(test_loader)
    show_evaluation(scores, labels)
    show_statistics(scores, labels)
