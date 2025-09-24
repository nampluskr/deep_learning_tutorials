import os
import random
import numpy as np
from types import SimpleNamespace

import torch
import torch.optim as optim

from dataloader import get_dataloaders
from model_stfpm import STFPM, STFPMTrainer
from model_autoencoder import AutoEncoder, AutoEncoderTrainer
from model_stmv1 import STMV1, STMTrainer
from model_efficientad import EfficientAD, EfficientADTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def show_model_info(model):
    print()
    print(f" > Total params.:     "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f" > Trainable params.: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def get_config(model_type, category, num_epochs=20):
    config = SimpleNamespace(
        data_root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=category,
        img_size=256,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
        seed=42,

        model_type=model_type,
        num_epochs=num_epochs,
        img_name=f"img_{category}_{model_type}_epochs-{num_epochs}",
        weight_name=f"model_{category}_{model_type}_epochs-{num_epochs}.pth",
        output_dir=os.path.join(".", category, model_type),
    )
    config.weight_path=os.path.join(config.output_dir, config.weight_name)
    return config


def run_experiment(trainer, config):
    print("\n" + "="*50)
    print(f"RUN EXPERIMENT: {config.model_type} / {config.category}")
    print("="*50 + "\n")

    train_loader, test_loader = get_dataloaders(config)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader, 
        weight_path=config.weight_path)
    # trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type, 
    #     skip_normal=True, num_max=20)
    # trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type, 
    #     skip_anomaly=True, num_max=20)


if __name__ == "__main__":

    category = "grid"
    num_epochs = 5

    # config = get_config("stfpm-resnet50", category, num_epochs)
    # config.batch_size=32
    # set_seed(seed=config.seed)
    # model=STFPM(backbone="resnet50", layers=["layer1", "layer2", "layer3"])
    # trainer = STFPMTrainer(model)
    # run_experiment(trainer, config)

    # config = get_config("autoencoder", category, num_epochs)
    # config.batch_size = 16
    # set_seed(seed=config.seed)
    # model=AutoEncoder(latent_dim=256, img_size=config.img_size)
    # trainer = AutoEncoderTrainer(model)
    # run_experiment(trainer, config)

    # config = get_config("stmv1", category, num_epochs)
    # config.batch_size=8
    # model = STMV1(out_channels=128)
    # trainer = STMTrainer(model)
    # run_experiment(trainer, config)

    config = get_config("efficientad-small", category, num_epochs)
    config.batch_size=8
    train_loader, test_loader = get_dataloaders(config)

    model = EfficientAD(model_size="small")
    trainer = EfficientADTrainer(model)
    trainer.fit(train_loader, num_epochs=config.num_epochs)

    # config = get_config("efficientad-medium", category, num_epochs)
    # config.batch_size=8
    # train_loader, test_loader = get_dataloaders(config)

    # model = EfficientAD(model_size="medium")
    # show_model_info(model)
    # trainer = EfficientADTrainer(model)
    # trainer.fit(train_loader, num_epochs=config.num_epochs)
