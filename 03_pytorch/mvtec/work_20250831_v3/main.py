import os
from datasets.dataset_mvtec import MVTecDataloader

from models.model_ae import UNetAE, AECombinedLoss
from modelers.modeler_ae import AEModeler
from trainers.trainer_gradient import GradientTrainer
from metrics.metrics_gradient import PSNRMetric, SSIMMetric

from models.model_padim import PadimModel
from modelers.modeler_padim import PadimModeler
from trainers.trainer_memory import MemoryTrainer
from metrics.metrics_memory import *

from models.model_stfpm import STFPMModel, STFPMLoss
from modelers.modeler_stfpm import STFPMModeler
from trainers.trainer_gradient import GradientTrainer
from metrics.metrics_gradient import PSNRMetric, SSIMMetric, FeatureSimilarityMetric
from metrics.metrics_gradient import LPIPSMetric

from models.model_base import set_backbone_dir
from utils import show_data_info, show_modeler_info, show_trainer_info, show_results


BACKBONE_DIR = os.path.abspath(os.path.join("..", "..", "backbones"))

import torch
from torchvision.transforms import v2

class TrainTransform:
    def __init__(self, img_size=256, **params):
        flip_prob = params.get('flip_prob', 0.5)
        rotation_degrees = params.get('rotation_degrees', 15)
        brightness = params.get('brightness', 0.1)
        contrast = params.get('contrast', 0.1)
        saturation = params.get('saturation', 0.1)
        hue = params.get('hue', 0.05)

        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.RandomHorizontalFlip(p=flip_prob),
            v2.RandomVerticalFlip(p=flip_prob),
            v2.RandomRotation(degrees=rotation_degrees),
            v2.ColorJitter(brightness=brightness, contrast=contrast,
                          saturation=saturation, hue=hue),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestTransform:
    def __init__(self, img_size=256, **params):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, image):
        return self.transform(image)

def get_data_for_gradient(categories=["bottle"]):
    return MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=categories,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        train_batch_size=32,
        test_batch_size=16,
        valid_ratio=0.2,
        num_workers=4,
        pin_memory=True
    )


def get_data_for_memory(categories=["bottle"]):
    return MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=categories,
        train_batch_size=4,
        test_batch_size=8,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        valid_ratio=0.0,  # No validation for memory models
        num_workers=2,
        pin_memory=False,
    )


def get_data_for_flow(categories=["bottle"]):
    return MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=categories,
        train_batch_size=4,
        test_batch_size=8,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        valid_ratio=0.1,
        num_workers=2,
        pin_memory=False,
    )


def run_unet_ae(verbose=True):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)

    categories=["carpet", "leather", "tile"]
    data = get_data_for_gradient(categories)

    modeler = AEModeler(
        model = UNetAE(in_channels=3, out_channels=3, latent_dim=512, img_size=256),
        loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean'),
        metrics = {
            # "psnr": PSNRMetric(max_val=1.0), 
            "ssim": SSIMMetric(data_range=1.0),
            "lpips": LPIPSMetric(net='vgg'),
        },
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)
    
    trainer.fit(data.train_loader(), num_epochs=10, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_padim(verbose=True):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: PADIM\n" + "="*50)

    categories=["carpet", "leather", "tile"]
    data = get_data_for_memory(categories)

    modeler = PadimModeler(
        model = PadimModel(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True, n_features=100),
        loss_fn = None,
        metrics = {},
    )
    trainer = MemoryTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)
    
    trainer.fit(data.train_loader(), num_epochs=1, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_stfpm(verbose=True):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)

    categories=["carpet", "leather", "tile"]
    # data = get_data_for_gradient(categories)
    data = get_data_for_memory(categories)

    modeler = STFPMModeler(
        model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18"),
        loss_fn = STFPMLoss(),
        metrics = {"feature_sim": FeatureSimilarityMetric(similarity_fn='cosine')},
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)
    
    trainer.fit(data.train_loader(), num_epochs=10)
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def check_backbones(backbone_dir):
    required_files = [
        "resnet18-f37072fd.pth",
        "resnet50-0676ba61.pth",
        "wide_resnet50_2-95faca4d.pth",
        "efficientnet_b0_ra-3dd342df.pth",
        "lpips_alex.pth",
        "lpips_vgg.pth",
        "lpips_squeeze.pth",
    ]

    if not os.path.exists(backbone_dir):
        print(f"Warning: Backbone directory not found: {backbone_dir}")
        print("Continuing with random initialization...")
        return False

    missing_files = []
    for file in required_files:
        full_path = os.path.join(backbone_dir, file)
        if not os.path.exists(full_path):
            missing_files.append(file)

    if missing_files:
        print(f"Warning: Missing backbone files: {missing_files}")
        print("Continuing with random initialization...")
        return False

    print(f"All backbone weights verified in: {backbone_dir}")
    return True



if __name__ == "__main__":

    backbone_available = check_backbones(BACKBONE_DIR)
    if backbone_available:
        set_backbone_dir(BACKBONE_DIR)

    verbose = True
    run_unet_ae(verbose)
    # run_stfpm(verbose)
    # run_padim(verbose)
