import os

from models.model_base import set_backbone_dir
from utils import show_data_info, show_modeler_info, show_trainer_info, show_results

import torch
from torchvision.transforms import v2


###########################################################
# Global Confuguration
###########################################################

DATASET_DIR = "/home/namu/myspace/NAMU/datasets"
BACKBONE_DIR = os.path.abspath(os.path.join("..", "..", "backbones"))
OUTPUT_DIR = "./experiments"

DATASETS = ["mvtec", "visa", "btad"]
MODELS = ["vanilla_ae", "unet_ae", 
    "stfpm", "padim", "fastflow", "patchcore", "draem",
    # "cutpaste", "efficientad"
]
TRAINING = ["gradient", "memory", "flow", "classification"]


###########################################################
# Transforms
###########################################################

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


###########################################################
# Load datasets
###########################################################

def get_data_for_gradient(categories=["pcb1"]):
    from datasets.dataset_visa import VisADataloader

    return VisADataloader(
        data_dir=os.path.join(DATASET_DIR, "visa"),
        categories=categories,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        train_batch_size=32,
        test_batch_size=16,
        valid_ratio=0.2,
        num_workers=8,
        pin_memory=True
    )

def get_data_for_memory(categories=["pcb1"]):
    from datasets.dataset_visa import VisADataloader

    return VisADataloader(
        data_dir='/home/namu/myspace/NAMU/datasets/visa',
        categories=categories,
        train_batch_size=4,
        test_batch_size=8,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        valid_ratio=0.0,  # No validation for memory models
        num_workers=2,
        pin_memory=False,
    )


def get_data_for_flow(categories=["pcb1"]):
    from datasets.dataset_visa import VisADataloader
    
    return VisADataloader(
        data_dir='/home/namu/myspace/NAMU/datasets/visa',
        categories=categories,
        train_batch_size=4,
        test_batch_size=8,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        valid_ratio=0.1,
        num_workers=2,
        pin_memory=False,
    )


###########################################################
# Run experiments
###########################################################

def run_vanilla_ae(verbose=True):
    from models.model_ae import VanillaAE, AECombinedLoss
    from modelers.modeler_ae import AEModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import PSNRMetric, SSIMMetric
    from metrics.metrics_gradient import LPIPSMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: VANILLA AUTOENCODER\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_gradient(categories)

    modeler = AEModeler(
        model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512, img_size=256),
        loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean'),
        metrics = {
            "psnr": PSNRMetric(max_val=1.0),
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


def run_unet_ae(verbose=True):
    from models.model_ae import UNetAE, AECombinedLoss
    from modelers.modeler_ae import AEModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import PSNRMetric, SSIMMetric
    from metrics.metrics_gradient import LPIPSMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: UNET-STYLE AUTOENCODER\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_gradient(categories)

    modeler = AEModeler(
        model = UNetAE(in_channels=3, out_channels=3, latent_dim=512, img_size=256),
        loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean'),
        metrics = {
            "psnr": PSNRMetric(max_val=1.0),
            "ssim": SSIMMetric(data_range=1.0),
            "lpips": LPIPSMetric(net='vgg'),
        },
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=5, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)

    modeler.save_model(os.path.join("./experiments", "__weight.pth"))


def run_padim(verbose=True):
    from models.model_padim import PadimModel
    from modelers.modeler_padim import PadimModeler
    from trainers.trainer_memory import MemoryTrainer

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: PADIM\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
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
    from models.model_stfpm import STFPMModel, STFPMLoss
    from modelers.modeler_stfpm import STFPMModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import FeatureSimilarityMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    # data = get_data_for_gradient(categories)
    data = get_data_for_memory(categories)

    modeler = STFPMModeler(
        model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet50"),
        loss_fn = STFPMLoss(),
        metrics = {"feature_sim": FeatureSimilarityMetric(similarity_fn='cosine')},
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=5)
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_fastflow(verbose=True):
    from models.model_fastflow import FastflowModel, FastflowLoss
    from modelers.modeler_fastflow import FastflowModeler
    from trainers.trainer_flow import FlowTrainer

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: FASTFLOW\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_flow(categories)

    modeler = FastflowModeler(
        model = FastflowModel(input_size=(256, 256),
                              backbone="wide_resnet50_2", ## resnet18 (default)
                              pre_trained=True, flow_steps=8),
        loss_fn = FastflowLoss(),
        metrics = {},
    )
    trainer = FlowTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=10, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_patchcore(verbose=True):
    from models.model_patchcore import PatchcoreModel
    from modelers.modeler_patchcore import PatchcoreModeler
    from trainers.trainer_memory import MemoryTrainer

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: PATCHCORE\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_memory(categories)

    modeler = PatchcoreModeler(
        model = PatchcoreModel(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            num_neighbors=9
        ),
        loss_fn = None,  # PatchCore doesn't use loss function
        metrics = {},
        coreset_sampling_ratio=0.1,
    )
    trainer = MemoryTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=1, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_draem(verbose=True):
    from models.model_draem import DraemModel, DraemLoss
    from modelers.modeler_draem import DraemModeler
    from metrics.metrics_gradient import PSNRMetric, SSIMMetric, LPIPSMetric
    from trainers.trainer_gradient import GradientTrainer

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: DRAEM\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_gradient(categories)

    modeler = DraemModeler(
        model = DraemModel(sspcab=False),
        loss_fn = DraemLoss(),
        metrics = {
            "ssim": SSIMMetric(data_range=1.0),
            "psnr": PSNRMetric(max_val=1.0),
            "lpips": LPIPSMetric(net='vgg'),
        },
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=5, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_cutpaste(verbose=True):
    from models.model_cutpaste import CutPasteModel, CutPasteLoss
    from modelers.modeler_cutpaste import CutPasteModeler
    from trainers.trainer_classification import ClassificationTrainer
    from metrics.metrics_base import AccuracyMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: CUTPASTE\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_gradient(categories)

    modeler = CutPasteModeler(
        model = CutPasteModel(
            backbone="resnet18",
            cut_size_ratio=(0.02, 0.15),  # 패치 크기 비율
            paste_number_range=(1, 4)     # 패치 개수
        ),
        loss_fn = CutPasteLoss(),
        metrics = {"accuracy": AccuracyMetric()},
    )
    trainer = ClassificationTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    # Two-stage training: Classification (30 epochs) + GMM fitting
    trainer.fit(data.train_loader(), num_epochs=5, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


def run_efficientad(verbose=False):
    from models.model_efficientad import EfficientAdModel, EfficientAdLoss
    from modelers.modeler_efficientad import EfficientAdModeler
    from trainers.trainer_gradient import GradientTrainer

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: EFFICIENTAD\n" + "="*50)
    categories=["pcb1", "pcb2", "pcb3"]
    data = get_data_for_gradient(categories)

    modeler = EfficientAdModeler(
        model = EfficientAdModel(
            teacher_out_channels=384,
            model_size="s",  # or "m" for medium
            padding=False,
            pad_maps=True
        ),
        loss_fn = EfficientAdLoss(),
        metrics = {},  # EfficientAd doesn't use standard reconstruction metrics
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=5, valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)


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

    ## 모델 개발 순서
    # run_vanilla_ae()    # manual
    run_unet_ae()       # manual
    # run_stfpm()         # anomalib
    # run_padim()         # anomalib
    # run_fastflow()      # anomalib
    # run_patchcore()     # anomalib
    # run_draem()         # anomalib



