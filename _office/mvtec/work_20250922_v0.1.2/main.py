import os
import random
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_config(model_type, category, num_epochs, output_dir):
    config = SimpleNamespace(
        # data_root="/mnt/d/datasets/mvtec",
        data_root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=category,
        img_size=256,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,

        model_type=model_type,
        num_epochs=num_epochs,
        learning_rate= 1e-4,
        latent_dim=512,
        output_dir=output_dir,
        seed=42,
        img_name=f"img_{category}_{model_type}_epochs-{num_epochs}",
        weight_path=os.path.join(output_dir, f"model_{category}_{model_type}_epochs-{num_epochs}.pth"),
        save_test=True
    )
    return config


def run_experiment(trainer, config):
    from dataloader import get_dataloaders

    print("\n" + "="*50)
    print(f"RUN EXPERIMENT: {config.category.upper()} - {config.model_type.upper()} MODEL")
    print("="*50 + "\n")

    train_loader, test_loader = get_dataloaders(config)
    show_model_info(trainer.model)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader, weight_path=config.weight_path)

    if config.save_test:
        trainer.test(test_loader, output_dir=config.output_dir, 
            show_image=False, img_prefix=f"{config.category}_{config.model_type}", 
            skip_normal=False, skip_anomaly=False, num_max=10)



def show_model_info(model):
    print()
    print(f" > Total params.:     "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f" > Trainable params.: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":

    # category = "grid"
    # num_epochs = 50
    # output_dir = f"./results/{category}"

    # from model_autoencoder import AutoEncoder, ConvMemAE, ConvDenoisingAE, AELoss, SSIMMetric
    # from trainer import AutoEncoderTrainer

    # set_seed()
    # config = get_config("autoencoder", category, num_epochs, output_dir)
    # metrics = {'mse': AELoss(), 'ssim': SSIMMetric()}
    # trainer = AutoEncoderTrainer(AutoEncoder(latent_dim=config.latent_dim), metrics=metrics)
    # run_experiment(trainer, config)

    # set_seed()
    # config = get_config("memory-ae", category, num_epochs, output_dir)
    # metrics = {'mse': AELoss(), 'ssim': SSIMMetric()}
    # trainer = AutoEncoderTrainer(AutoEncoder(latent_dim=config.latent_dim), metrics=metrics)
    # run_experiment(ConvMemAE(latent_dim=latent_dim), config)

    # set_seed()
    # config = get_config("denoising-ae", category, num_epochs, output_dir)
    # metrics = {'mse': AELoss(), 'ssim': SSIMMetric()}
    # trainer = AutoEncoderTrainer(AutoEncoder(latent_dim=config.latent_dim), metrics=metrics)
    # run_experiment(ConvDenoisingAE(latent_dim=latent_dim), config)

    # from model_stfpm import STFPM
    # from trainer import STFPMTrainer

    # set_seed()
    # config = get_config("stfpm-resnet18", category, num_epochs, output_dir)
    # trainer = STFPMTrainer(STFPM(backbone="resnet18", pretrained=True))
    # run_experiment(trainer, config)

    # set_seed()
    # config = get_config("stfpm-resnet50", category, num_epochs, output_dir)
    # trainer = STFPMTrainer(STFPM(backbone="resnet50", pretrained=True))
    # run_experiment(trainer, config)


    # from model_efficientad import EfficientAD
    # from trainer import EfficientADTrainer

    # set_seed()
    # config = get_config("efficientad-resnet18", category, num_epochs, output_dir)
    # trainer = EfficientADTrainer(EfficientAD(backbone="resnet18", pretrained=True, img_size=256))
    # run_experiment(trainer, config)

    # set_seed()
    # config = get_config("efficientad-resnet50", category, num_epochs, output_dir)
    # trainer = EfficientADTrainer(EfficientAD(backbone="resnet50", pretrained=True, img_size=256))
    # run_experiment(trainer, config)

    from model_manual import ManualEfficientAD
    from trainer import ManualTrainer

    category = "grid"
    num_epochs = 20
    output_dir = f"./results/{category}"
    config = get_config("manual", category, num_epochs, output_dir)

    BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
    teacher_backbone_path = os.path.join(BACKBONE_DIR, "efficientnet_b7_lukemelas-c5b4e57e.pth")
    student_backbone_path = os.path.join(BACKBONE_DIR, "wide_resnet101_2-32ee1156.pth")
    trainer = ManualTrainer(ManualEfficientAD(
        teacher_backbone_path=teacher_backbone_path,
        student_backbone_path=student_backbone_path,
        out_channels=128))
    run_experiment(trainer, config)

    # from model_stfpmv2 import STFPMV2, STFPMV2Loss, STFPMV2Metric
    # from model_efficientadv2 import EfficientADV2, EfficientADV2Loss, EfficientADV2Metric
    # from trainerv2 import STFPMV2Trainer, EfficientADV2Trainer

    # category = "grid"
    # num_epochs = 20
    # output_dir = f"./results/{category}"
    
    # set_seed()
    # config = get_config("stfpmv2", category, num_epochs, output_dir)
    # trainer = STFPMV2Trainer(
    #     model = STFPMV2(backbone="resnet50", pretrained=True, out_channels=128), 
    #     use_amp=True, pretrain_penalty=False)
    # run_experiment(trainer, config)

    # set_seed()
    # config = get_config("efficientadv2", category, num_epochs, output_dir)
    # trainer = EfficientADV2Trainer(
    #     model = EfficientADV2(backbone="resnet50", pretrained=True, out_channels=128, img_size=256),
    #     use_amp=True, pretrain_penalty=False)
    # run_experiment(trainer, config)
