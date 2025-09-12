import os
from types import SimpleNamespace
import logging

import torch
import torch.nn as nn
import torch.optim as optim


from dataloader import get_dataloaders
from utils import show_evaluation, show_statistics
from model_base import TimmFeatureExtractor, set_backbone_dir, check_backbone_files


BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
    "vgg16": "vgg16-397923af.pth",
    "alexnet": "alexnet-owt-7be5be79.pth",
    "squeezenet1_1": "squeezenet1_1-b8a52dc0.pth",
    # EfficientAD weights
    "efficientad_teacher_small": "pretrained_teacher_small.pth",
    "efficientad_teacher_medium": "pretrained_teacher_medium.pth",
    # LPIPS weights
    "lpips_alex": "lpips_alex.pth",
    "lpips_vgg": "lpips_vgg.pth",
    "lpips_squeeze": "lpips_squeeze.pth"
}

def get_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'experiment.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_config():
    config = SimpleNamespace(
        # data_dir="/mnt/d/datasets/mvtec",   # WSL
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category="grid",

        backbone_dir="/home/namu/myspace/NAMU/project_2025/backbones",
        model_type="none",
        batch_size=8,
        img_size=256,
        latent_dim=512,
        num_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-5,

        seed=42,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    config.output_dir = f"results_{config.category}"
    os.makedirs(config.output_dir, exist_ok=True)
    return config


def run_autoencoder(config):
    from model_ae import VanillaAE
    from trainer import AutoEncoderTrainer
    from metrics import PSNRMetric, SSIMMetric

    logger = get_logger(config.output_dir)

    train_loader, test_loader = get_dataloaders(config.data_dir, config.category,
        batch_size=config.batch_size, img_size=config.img_size)

    model = VanillaAE(latent_dim=config.latent_dim).to(config.device)
    loss_fn = nn.MSELoss()
    metrics = {'psnr': PSNRMetric(), 'ssim': SSIMMetric()}
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = AutoEncoderTrainer(model, optimizer, loss_fn, metrics=metrics, logger=logger)
    history = trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader)

    scores, labels = trainer.predict(test_loader)
    show_evaluation(scores, labels)
    show_statistics(scores, labels)


def run_stfpm(config):
    from model_stfpm import STFPMModel, STFPMLoss
    from trainer import STFPMTrainer
    from metrics import FeatureSimilarityMetric

    set_backbone_dir(config.backbone_dir)
    check_backbone_files(config.backbone_dir)
    logger = get_logger(config.output_dir)

    train_loader, test_loader = get_dataloaders(config.data_dir, config.category,
        batch_size=config.batch_size, img_size=config.img_size)

    model = STFPMModel(backbone='resnet18', layers=['layer1', 'layer2', 'layer3']).to(config.device)
    # model = STFPMModel(backbone='resnet50', layers=['layer1', 'layer2', 'layer3']).to(config.device)
    loss_fn = STFPMLoss()
    # metrics = {'fsim': FeatureSimilarityMetric(similarity_fn='cosine')}
    metrics = {}
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = STFPMTrainer(model, optimizer, loss_fn, metrics=metrics, logger=logger)
    history = trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader)

    scores, labels = trainer.predict(test_loader)
    show_evaluation(scores, labels)
    show_statistics(scores, labels)


def run_efficientad(config):
    from model_efficientad import EfficientADModel
    from trainer import EfficientADTrainer

    set_backbone_dir(config.backbone_dir)
    check_backbone_files(config.backbone_dir)
    logger = get_logger(config.output_dir)

    train_loader, test_loader = get_dataloaders(config.data_dir, config.category,
        batch_size=config.batch_size, img_size=config.img_size)

    model = EfficientADModel(model_size="small",
            teacher_out_channels=384,
            padding=False,
            pad_maps=True,
            use_imagenet_penalty=True,).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = EfficientADTrainer(model, optimizer, logger=logger)
    history = trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader)

    scores, labels = trainer.predict(test_loader)
    show_evaluation(scores, labels)
    show_statistics(scores, labels)


if __name__ == "__main__":

    # config = get_config()
    # run_autoencoder(config)

    config = get_config()
    run_stfpm(config)

    # config = get_config()
    # config.batch_size = 4
    # run_efficientad(config)

