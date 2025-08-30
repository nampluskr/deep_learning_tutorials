import os
from dataset_factory import get_transform
from dataset_mvtec import MVTecDataloader
from model_ae import VanillaAE, UNetAE, AECombinedLoss
from modeler_ae import AEModeler

from model_padim import PadimModel
from modeler_padim import PadimModeler

from model_stfpm import STFPMModel, STFPMLoss
from modeler_stfpm import STFPMModeler

from metrics import get_metric
from trainer import Trainer

from model_base import set_backbone_dir as set_model_backbone_dir
# from metrics import set_backbone_dir as set_metrics_backbone_dir


BACKBONE_DIR = os.path.abspath(os.path.join("..", "..", "backbones"))


def run_autoencoder():
    train_loader, valid_loader, test_loader = get_dataloaders()
    modeler = AEModeler(
        model=UNetAE(),
        loss_fn=AECombinedLoss(),
        metrics={"psnr": get_metric("psnr"),
                 "ssim": get_metric("ssim"),
        },
    )
    trainer = Trainer(modeler)
    trainer.fit(train_loader, num_epochs=10, valid_loader=valid_loader)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")

def run_padim():
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        # categories=["carpet", "leather", "tile", "wood"],
        categories=["carpet", "leather", "tile"],
        train_batch_size=4,
        test_batch_size=8,
        train_transform = get_transform("train"),
        test_transform = get_transform("test"),
        valid_ratio=0.0,
        seed=42,
        num_workers= 4,
        pin_memory=False,
        # persistent_workers=True,
    )
    train_loader = mvtec.train_loader()
    test_loader = mvtec.test_loader()

    modeler = PadimModeler(
        model=PadimModel(),
    )
    trainer = Trainer(modeler)
    trainer.fit(train_loader, num_epochs=1)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")


def run_stfpm():
    train_loader, valid_loader, test_loader = get_dataloaders()
    modeler = STFPMModeler(
        model=STFPMModel(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"]
        ),
        loss_fn = STFPMLoss(),
    )
    trainer = Trainer(modeler)
    trainer.fit(train_loader, num_epochs=10)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")


def get_dataloaders():
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        # categories=["carpet", "leather", "tile", "wood"],
        categories=["carpet", "leather", "tile"],
        train_batch_size=4,
        test_batch_size=8,
        train_transform = get_transform("train"),
        test_transform = get_transform("test"),
        valid_ratio=0.0,
        seed=42,
        num_workers= 2,
        pin_memory=False,
        # persistent_workers=True,
    )
    train_loader = mvtec.train_loader()
    valid_loader = mvtec.valid_loader()
    test_loader = mvtec.test_loader()

    print(f" > Train dataset: {len(train_loader.dataset)}")
    # print(f" > Valid dataset: {len(valid_loader.dataset)}")
    print(f" > Test dataset:  {len(test_loader.dataset)}")
    return train_loader, valid_loader, test_loader


def show_results(scores, labels, method="roc"):
    auroc_metric = get_metric("auroc")
    aupr_metric = get_metric("aupr")
    threshold_metric = get_metric("threshold", method=method)

    auroc = auroc_metric(labels, scores)
    aupr = aupr_metric(labels, scores)
    optimal_threshold = threshold_metric(labels, scores)

    print()
    print(f" > AUROC:     {auroc:.4f}")
    print(f" > AUPR:      {aupr:.4f}")
    print(f" > Threshold: {optimal_threshold:.4f}")


def verify_backbone_setup(backbone_dir):
    """Verify backbone directory and required files"""
    required_files = [
        "resnet18-f37072fd.pth",
        "resnet50-0676ba61.pth",
        "wide_resnet50_2-95faca4d.pth",
        "efficientnet_b0_ra-3dd342df.pth",
        # "lpips_alex.pth",
        # "lpips_vgg.pth",
        # "lpips_squeeze.pth"
    ]

    if not os.path.exists(backbone_dir):
        raise FileNotFoundError(f"Backbone directory not found: {backbone_dir}")

    missing_files = []
    for file in required_files:
        full_path = os.path.join(backbone_dir, file)
        if not os.path.exists(full_path):
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"Missing backbone files in {backbone_dir}: {missing_files}")

    print(f"All backbone weights verified in: {backbone_dir}")
    return True

if __name__ == "__main__":

    if verify_backbone_setup(BACKBONE_DIR):
        set_model_backbone_dir(BACKBONE_DIR)
        # set_metrics_backbone_dir(BACKBONE_DIR)

    # run_autoencoder()
    run_padim()
    # run_stfpm()