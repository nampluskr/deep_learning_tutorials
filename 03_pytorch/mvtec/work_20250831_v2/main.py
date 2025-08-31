import os
from datasets.dataset_factory import get_transform
from datasets.dataset_mvtec import MVTecDataloader
from models.model_ae import VanillaAE, UNetAE, AECombinedLoss
from modelers.modeler_ae import AEModeler

from models.model_padim import PadimModel
from modelers.modeler_padim import PadimModeler  # Use fixed version

from models.model_stfpm import STFPMModel, STFPMLoss
from modelers.modeler_stfpm import STFPMModeler

from metrics.metrics_factory import get_metric
from trainers.trainer_factory import get_trainer, Trainer  # New trainer system

from models.model_base import set_backbone_dir as set_model_backbone_dir


BACKBONE_DIR = os.path.abspath(os.path.join("..", "..", "backbones"))


def run_autoencoder():
    """Run autoencoder anomaly detection experiment"""
    print("\n" + "="*50)
    print("RUNNING AUTOENCODER EXPERIMENT")
    print("="*50)

    train_loader, valid_loader, test_loader = get_dataloaders_for_gradient()
    modeler = AEModeler(
        model=UNetAE(),
        loss_fn=AECombinedLoss(),
        metrics={"psnr": get_metric("psnr"),
                 "ssim": get_metric("ssim"),
        },
    )

    # Auto-detect trainer (will be GradientTrainer)
    trainer = get_trainer(modeler)
    print(f" > Using {trainer.trainer_type} trainer")

    trainer.fit(train_loader, num_epochs=10, valid_loader=valid_loader)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")


def run_padim():
    """Run PaDiM experiment"""
    print("\n" + "="*50)
    print("RUNNING PADIM EXPERIMENT")
    print("="*50)

    # Create dataloader optimized for memory-based models
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["carpet", "leather", "tile"],
        train_batch_size=4,
        test_batch_size=8,
        train_transform=get_transform("train"),
        test_transform=get_transform("test"),
        valid_ratio=0.0,  # CRITICAL: No validation split for memory models
        seed=42,
        num_workers=4,
        pin_memory=False,
    )
    train_loader = mvtec.train_loader()
    test_loader = mvtec.test_loader()

    modeler = PadimModeler(
        model=PadimModel(),
    )

    # Auto-detect trainer (will be MemoryTrainer)
    trainer = get_trainer(modeler)
    print(f" > Using {trainer.trainer_type} trainer")

    # Memory-based training: 1 epoch feature collection + fitting
    print(f" > Model fitted before: {trainer.is_fitted()}")
    trainer.fit(train_loader)
    print(f" > Model fitted after: {trainer.is_fitted()}")

    # FIXED: Use test data for validation (solves score_sep=0.000 problem!)
    print(" > Running test-based validation...")
    validation_results = trainer.validate_with_test_data(test_loader)

    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")

    # Display the separation results
    print(f"\n > VALIDATION RESULTS:")
    print(f"   Score separation: {validation_results.get('separation', 0.0):.4f}")
    print(f"   Normal score mean: {validation_results.get('normal_mean', 0.0):.4f}")
    print(f"   Anomaly score mean: {validation_results.get('anomaly_mean', 0.0):.4f}")


def run_stfpm():
    """Run STFPM experiment"""
    print("\n" + "="*50)
    print("RUNNING STFPM EXPERIMENT")
    print("="*50)

    train_loader, valid_loader, test_loader = get_dataloaders_for_gradient()
    modeler = STFPMModeler(
        model=STFPMModel(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"]
        ),
        loss_fn=STFPMLoss(),
    )

    # Auto-detect trainer (will be GradientTrainer despite having memory monitoring)
    trainer = get_trainer(modeler)
    print(f" > Using {trainer.trainer_type} trainer")

    trainer.fit(train_loader, num_epochs=10)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")


def get_dataloaders_for_memory():
    """Create dataloaders optimized for memory-based models (no validation split)"""
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["carpet", "leather", "tile"],
        train_batch_size=4,
        test_batch_size=8,
        train_transform=get_transform("train"),
        test_transform=get_transform("test"),
        valid_ratio=0.0,  # No validation for memory models
        seed=42,
        num_workers=2,
        pin_memory=False,
    )

    train_loader = mvtec.train_loader()
    test_loader = mvtec.test_loader()

    print(f" > Train dataset: {len(train_loader.dataset)}")
    print(f" > Test dataset:  {len(test_loader.dataset)}")
    return train_loader, None, test_loader


def get_dataloaders_for_gradient():
    """Create dataloaders optimized for gradient-based models (with validation split)"""
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["carpet", "leather", "tile"],
        train_batch_size=4,
        test_batch_size=8,
        train_transform=get_transform("train"),
        test_transform=get_transform("test"),
        valid_ratio=0.2,  # 20% validation for gradient models
        seed=42,
        num_workers=2,
        pin_memory=False,
    )

    train_loader = mvtec.train_loader()
    valid_loader = mvtec.valid_loader()
    test_loader = mvtec.test_loader()

    print(f" > Train dataset: {len(train_loader.dataset)}")
    print(f" > Valid dataset: {len(valid_loader.dataset)}")
    print(f" > Test dataset:  {len(test_loader.dataset)}")
    return train_loader, valid_loader, test_loader


def show_results(scores, labels, method="roc"):
    """Display evaluation results (original function)"""
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
    """Verify backbone directory and required files (original function)"""
    required_files = [
        "resnet18-f37072fd.pth",
        "resnet50-0676ba61.pth",
        "wide_resnet50_2-95faca4d.pth",
        "efficientnet_b0_ra-3dd342df.pth",
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


def demo_trainer_system():
    """Demonstrate the new trainer system capabilities"""
    print("\n" + "="*60)
    print("TRAINER SYSTEM DEMONSTRATION")
    print("="*60)

    # Show auto-detection capabilities
    modelers = [
        ("PaDiM", PadimModeler(model=PadimModel(n_features=100))),
        ("Autoencoder", AEModeler(model=UNetAE(img_size=256), loss_fn=AECombinedLoss())),
        ("STFPM", STFPMModeler(model=STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18"), loss_fn=STFPMLoss())),
    ]

    print(" > Auto-detection results:")
    for name, modeler in modelers:
        trainer = get_trainer(modeler)
        print(f"   {name:12} -> {trainer.trainer_type:8} trainer")

    # Show backward compatibility
    print(f"\n > Backward compatibility:")
    padim_modeler = PadimModeler(model=PadimModel(n_features=100))
    old_style_trainer = Trainer(padim_modeler)  # Old interface
    new_style_trainer = get_trainer(padim_modeler)  # New interface
    print(f"   Old style: {old_style_trainer.trainer_type} trainer")
    print(f"   New style: {new_style_trainer.trainer_type} trainer")
    print(f"   Same type: {old_style_trainer.trainer_type == new_style_trainer.trainer_type}")


if __name__ == "__main__":

    backbone_available = verify_backbone_setup(BACKBONE_DIR)
    if backbone_available:
        set_model_backbone_dir(BACKBONE_DIR)

    # run_autoencoder()
    # run_stfpm()
    run_padim()
