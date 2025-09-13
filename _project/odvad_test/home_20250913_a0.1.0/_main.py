import os
from datasets.dataset_mvtec import MVTecDataloader
from models.model_base import set_backbone_dir
from utils import show_data_info, show_modeler_info, show_trainer_info, show_results_old, show_results_new


BACKBONE_DIR = '/mnt/d/backbones'

import torch
from torchvision.transforms import v2

class TrainTransform:
    def __init__(self, img_size=256, **params):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestTransform:
    def __init__(self, img_size=256, **params):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.transform(image)


def run_vanilla_ae(verbose=True):
    from models.model_ae import VanillaAE, AECombinedLoss
    from modelers.modeler_ae import AEModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import SSIMMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)
    categories=["grid"]
    data = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=categories,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        train_batch_size=4,
        test_batch_size=2,
        valid_ratio=0.2,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    modeler = AEModeler(
        model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512),
        loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean'),
        metrics = {"ssim": SSIMMetric(data_range=1.0)},
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=10, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results_old(scores, labels)
    show_results_new(scores, labels)


def run_unet_ae(verbose=True):
    from models.model_ae import UNetAE, AECombinedLoss
    from modelers.modeler_ae import AEModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import SSIMMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)
    categories=["grid"]
    data = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=categories,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        train_batch_size=4,
        test_batch_size=2,
        valid_ratio=0.2,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    modeler = AEModeler(
        model = UNetAE(in_channels=3, out_channels=3, latent_dim=512, img_size=256),
        loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean'),
        metrics = {"ssim": SSIMMetric(data_range=1.0)},
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=10, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results_old(scores, labels)
    show_results_new(scores, labels)


def run_stfpm(verbose=True):
    from models.model_stfpm import STFPMModel, STFPMLoss
    from modelers.modeler_stfpm import STFPMModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import FeatureSimilarityMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)
    data = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["grid"],
        train_batch_size=4,
        test_batch_size=2,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        valid_ratio=0.2,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

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

    trainer.fit(data.train_loader(), num_epochs=10, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results_old(scores, labels)
    show_results_new(scores, labels)


def check_backbones(backbone_dir):
    required_files = [
        "resnet18-f37072fd.pth",
        "resnet34-b627a593.pth",
        "resnet50-0676ba61.pth",
        "wide_resnet50_2-95faca4d.pth",
        "efficientnet_b0_ra-3dd342df.pth",
        "vgg16-397923af.pth",
        "alexnet-owt-7be5be79.pth",
        "squeezenet1_1-b8a52dc0.pth",
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

    run_vanilla_ae()
    # run_unet_ae()
    # run_stfpm()
    # run_padim()
    # run_fastflow()
    # run_patchcore()
    # run_draem()
    # run_cutpaste()
    # run_efficientad()



