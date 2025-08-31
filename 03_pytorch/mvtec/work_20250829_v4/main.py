#!/usr/bin/env python3
"""
Updated main.py using the new modularized trainer architecture
Fixes PaDiM validation issues and optimizes training for each model type
"""

import os
from dataset_factory import get_transform
from dataset_mvtec import MVTecDataloader
from model_ae import VanillaAE, UNetAE, AECombinedLoss
from modeler_ae import AEModeler

from model_padim import PadimModel
from modeler_padim import PadimModeler  # Use fixed version

from model_stfpm import STFPMModel, STFPMLoss
from modeler_stfpm import STFPMModeler

from metrics import get_metric
from trainer_factory import get_trainer, Trainer  # New trainer system

from model_base import set_backbone_dir as set_model_backbone_dir


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
    """Run PaDiM experiment with FIXED validation"""
    print("\n" + "="*50)
    print("RUNNING PADIM EXPERIMENT (FIXED)")
    print("="*50)
    
    # FIXED: Create dataloader optimized for memory-based models
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
    
    # Display the FIXED separation results
    print(f"\n > FIXED VALIDATION RESULTS:")
    print(f"   Score separation: {validation_results.get('separation', 0.0):.4f} (Previously 0.000!)")
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


def run_comparison_demo():
    """Run side-by-side comparison of old vs new trainer behavior"""
    print("\n" + "="*60)
    print("OLD vs NEW TRAINER COMPARISON")
    print("="*60)
    
    # Create simple dataset
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["carpet"],
        train_batch_size=4,
        test_batch_size=8,
        train_transform=get_transform("train"),
        test_transform=get_transform("test"),
        valid_ratio=0.0,
        seed=42,
        num_workers=2,
        pin_memory=False,
    )
    
    train_loader = mvtec.train_loader()
    test_loader = mvtec.test_loader()
    
    print(f" > Dataset: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    
    # Test with PaDiM (the problematic case)
    print(f"\n > Testing PaDiM validation fix:")
    
    modeler = PadimModeler(model=PadimModel())
    
    # Use backward compatible interface (auto-detects MemoryTrainer)
    trainer = Trainer(modeler)
    print(f"   Detected trainer type: {trainer.trainer_type}")
    
    # Training
    history = trainer.fit(train_loader)
    
    # The key fix: proper validation with test data
    validation_results = trainer.validate_with_test_data(test_loader)
    
    # Final prediction
    scores, labels = trainer.predict(test_loader)
    
    # Results
    auroc = get_metric("auroc")(labels, scores)
    aupr = get_metric("aupr")(labels, scores)
    
    print(f"\n   RESULTS:")
    print(f"   AUROC: {auroc:.4f}")
    print(f"   AUPR: {aupr:.4f}")
    print(f"   Score separation: {validation_results.get('separation', 0.0):.4f}")
    
    if validation_results.get('separation', 0.0) > 0.001:
        print("   ✓ VALIDATION PROBLEM FIXED!")
    else:
        print("   ✗ Validation still showing low separation")


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
        
    run_autoencoder()
    run_stfpm()
    run_padim()
    
    # print("OLED Anomaly Detection Framework")
    # print("Updated Main Script with Fixed Trainer Architecture")
    # print("=" * 80)
    
    # # Setup backbone weights
    # backbone_available = verify_backbone_setup(BACKBONE_DIR)
    # if backbone_available:
    #     set_model_backbone_dir(BACKBONE_DIR)
    
    # # Demonstrate trainer system
    # demo_trainer_system()
    
    # # Run comparison to show the fix
    # run_comparison_demo()
    
    # # Run main experiments
    # print("\n" + "="*60)
    # print("RUNNING MAIN EXPERIMENTS")
    # print("="*60)
    
    # # PaDiM with FIXED validation
    # run_padim()
    
    # # Uncomment to run other experiments
    # # run_autoencoder()
    # # run_stfpm()
    
    # print("\n" + "="*80)
    # print("EXPERIMENT COMPLETED!")
    # print("Key improvements:")
    # print("  ✓ PaDiM validation problem FIXED (score_sep now > 0)")
    # print("  ✓ Memory models use optimized 1-epoch training")
    # print("  ✓ Gradient models use proper multi-epoch training")
    # print("  ✓ Full backward compatibility maintained")
    # print("  ✓ Easy to extend for new model types")
    # print("="*80)