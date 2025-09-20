import os
import random
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from dataloader import get_dataloaders
from model_autoencoder import Baseline, AutoEncoder, SSIMMetric, AELoss, AECombinedLoss
from trainer import AutoEncoderTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_config(model_type, category, num_epochs, output_dir, latent_dim=512):
    config = SimpleNamespace(
        data_root="/mnt/d/datasets/mvtec",
        category=category,
        img_size=256,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,

        model_type=model_type,
        num_epochs=num_epochs,
        learning_rate= 1e-4,
        latent_dim=latent_dim,
        output_dir=output_dir,
        seed=42,
        img_name=f"img_{category}_{model_type}_latent-{latent_dim}_epochs-{num_epochs}",
        weight_path=os.path.join(output_dir, f"model_{category}_{model_type}_latent-{latent_dim}_epochs-{num_epochs}.pth"),
    )
    return config


def run_experiment(model, config):
    print("\n" + "="*50)
    print(f"RUN EXPERIMENT: {config.category.upper()} - {config.model_type.upper()} MODEL")
    print("="*50 + "\n")

    train_loader, test_loader = get_dataloaders(config)
    show_model_info(model)
    
    loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3)
    metrics = {'mse': AELoss(), 'ssim': SSIMMetric()}
    trainer = AutoEncoderTrainer(model, loss_fn=loss_fn, metrics=metrics)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader, weight_path=config.weight_path)

    trainer.test(test_loader, output_dir=config.output_dir, show_image=False, img_name=config.img_name)
    print(trainer.evaluate_image_level(test_loader, method="roc"))
    print(trainer.evaluate_pixel_level(test_loader, percentile=95))


def show_model_info(model):
    print()
    print(f" > Total params.:     "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f" > Trainable params.: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":

    category = "grid"
    num_epochs = 100
    output_dir = f"./results/{category}"

    set_seed()
    config = get_config("baseline", category, num_epochs, output_dir, latent_dim=1024)
    run_experiment(Baseline(latent_dim=1024), config)

    set_seed()
    config = get_config("baseline", category, num_epochs, output_dir, latent_dim=512)
    run_experiment(Baseline(latent_dim=512), config)

    # set_seed()
    # config = get_config("ae_none", category, num_epochs, output_dir)
    # run_experiment(AutoEncoder(backbone="none", latent_dim=512), config)

    # set_seed()
    # config = get_config("ae_resnet18", category, num_epochs, output_dir)
    # run_experiment(AutoEncoder(backbone="resnet18", latent_dim=512), config)

    # set_seed()
    # config = get_config("ae_resnet34", category, num_epochs, output_dir)
    # run_experiment(AutoEncoder(backbone="resnet34", latent_dim=512), config)

    # set_seed()
    # config = get_config("ae_resnet50", category, num_epochs, output_dir)
    # run_experiment(AutoEncoder(backbone="resnet50", latent_dim=512), config)