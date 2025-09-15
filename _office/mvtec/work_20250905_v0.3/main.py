import os
import random
import numpy as np
import torch
import torch.optim as optim
from types import SimpleNamespace

from utils import show_model_info, show_results_old, show_results_new
from utils import get_thresholds, evaluate_thresholds
from dataloader import get_dataloaders
from trainer import AutoEncoderTrainer, STFPMTrainer, EfficientADTrainer
from model_ae import VanillaAE, UNetAE, AECombinedLoss, SSIMMetric
from model_stfpm import STFPMModel, STFPMLoss, FeatureSimilarityMetric
from model_efficientad import EfficientADModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logging.info("Logger initialized in main.py")

CATEGORY = "tile"


def get_config():
    config = SimpleNamespace(
        # Dataset Configuration
        data_type="mvtec",
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=CATEGORY,
        batch_size=4,
        img_size=256,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,

        # Training Configuration
        num_epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-2,
        optimizer_type="adamw",  # adam, adamw, sgd
    )
    return config


def run_autoencoder(num_epochs=10, output_dir=None, model_type="ae"):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)

    train_loader, test_loader = get_dataloaders(
        root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=CATEGORY, batch_size=4, img_size=256)
    device = get_device()
    model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512).to(device)
    show_model_info(model)

    loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean')
    metrics = {"ssim": SSIMMetric(data_range=1.0)}
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    trainer = AutoEncoderTrainer(model, optimizer, loss_fn, metrics=metrics)

    trainer.fit(train_loader, num_epochs=num_epochs)
    scores, labels = trainer.predict(test_loader)
    thresholds = get_thresholds(scores, labels)
    results = evaluate_thresholds(scores, labels, thresholds)
    print(results)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(os.path.join(output_dir, f"model_{model_type}_epochs-{num_epochs}.pth"))


def run_unet(num_epochs=10, output_dir=None, model_type="unet"):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: UNET\n" + "="*50)

    train_loader, test_loader = get_dataloaders(
        root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=CATEGORY, batch_size=4, img_size=256)
    device = get_device()
    model = UNetAE(in_channels=3, out_channels=3, latent_dim=512).to(device)
    show_model_info(model)

    loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean')
    metrics = {"ssim": SSIMMetric(data_range=1.0)}
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    trainer = AutoEncoderTrainer(model, optimizer, loss_fn, metrics=metrics)

    trainer.fit(train_loader, num_epochs=num_epochs)
    scores, labels = trainer.predict(test_loader)
    thresholds = get_thresholds(scores, labels)
    results = evaluate_thresholds(scores, labels, thresholds)
    print(results)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(os.path.join(output_dir, f"model_{model_type}_epochs-{num_epochs}.pth"))


def run_stfpm(num_epochs=10, output_dir=None, model_type="stfpm"):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)

    train_loader, test_loader = get_dataloaders(
        root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=CATEGORY, batch_size=4, img_size=256)
    device = get_device()
    # model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18").to(device)
    model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet50").to(device)
    show_model_info(model)

    loss_fn = STFPMLoss()
    metrics = {"feat_sim": FeatureSimilarityMetric()}
    optimizer = optim.AdamW(model.student_model.parameters(), lr=1e-3, weight_decay=1e-2)
    trainer = STFPMTrainer(model, optimizer, loss_fn, metrics=metrics)

    trainer.fit(train_loader, num_epochs=num_epochs)
    scores, labels = trainer.predict(test_loader)
    thresholds = get_thresholds(scores, labels)
    results = evaluate_thresholds(scores, labels, thresholds)
    print(results)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(os.path.join(output_dir, f"model_{model_type}_epochs-{num_epochs}.pth"))


def run_efficientad(model_size="small", num_epochs=30, output_dir=None, model_type="efficientad"):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: EFFICIENTAD\n" + "="*50)

    train_loader, test_loader = get_dataloaders(
        root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=CATEGORY, batch_size=8, img_size=256)

    device = get_device()
    if model_size == "small":
        model = EfficientADModel(teacher_out_channels=384, model_size="small",
            padding=False, pad_maps=True, use_imagenet_penalty=True).to(device)
        optimizer = optim.AdamW(list(model.student.parameters()) + list(model.ae.parameters()),
            lr=1e-4, weight_decay=1e-3, betas=(0.9, 0.999))
        # num_epochs = 30
    elif model_size == "medium":
        model = EfficientADModel(teacher_out_channels=384, model_size="medium",
            padding=False, pad_maps=True, use_imagenet_penalty=True).to(device)
        optimizer = optim.AdamW(list(model.student.parameters()) + list(model.ae.parameters()),
            lr=1e-4, weight_decay=1e-3, betas=(0.9, 0.999))
        # num_epochs = 50
    show_model_info(model)

    loss_fn = None  # Loss is computed internally by the model
    # metrics = {"distance": EfficientADMetric()}
    metrics = None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode='min', factor=0.5, patience=10, verbose=True)
    trainer = EfficientADTrainer(model, optimizer, loss_fn, metrics=metrics, scheduler=scheduler)
    trainer.fit(train_loader, num_epochs=num_epochs)
    scores, labels = trainer.predict(test_loader)
    thresholds = get_thresholds(scores, labels)
    results = evaluate_thresholds(scores, labels, thresholds)
    print(results)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(os.path.join(output_dir, f"model_{model_type}_epochs-{num_epochs}.pth"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    set_seed(seed=42)
    OUTPUT_DIR = "./experiments"

    # run_autoencoder(num_epochs=50, output_dir=OUTPUT_DIR, model_type="ae")
    # run_unet(num_epochs=50, output_dir=OUTPUT_DIR, model_type="unet")
    # run_stfpm(num_epochs=50, output_dir=OUTPUT_DIR, model_type="stfpm")
    # run_efficientad(model_size="small", num_epochs=5, output_dir=OUTPUT_DIR, model_type="effad-small")
    run_efficientad(model_size="medium", num_epochs=50, output_dir=OUTPUT_DIR, model_type="effad-medium")
