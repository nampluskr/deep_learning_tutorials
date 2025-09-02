# config.py
# Configuration definitions for all models, datasets, and combinations

from types import SimpleNamespace

# ===================================================================
# BASE CONFIGURATION
# ===================================================================

BASE_CONFIGS = SimpleNamespace(
    img_size=256,
    train_batch_size=16,
    test_batch_size=16,
    num_workers=0,
    pin_memory=False,
)

# ===================================================================
# DATASET CONFIGURATIONS
# ===================================================================

DATASET_CONFIGS = {
    "mvtec": SimpleNamespace(
        dataset_type="mvtec",
        dataset_params=dict(),
        categories=["bottle", "cable", "capsule", "carpet", "grid", 
                   "hazelnut", "leather", "metal_nut", "pill", "screw", 
                   "tile", "toothbrush", "transistor", "wood", "zipper"]
    ),
    "btad": SimpleNamespace(
        dataset_type="btad",
        dataset_params=dict(),
        categories=["01", "02", "03"]
    ),
    "visa": SimpleNamespace(
        dataset_type="visa",
        dataset_params=dict(),
        categories=["candle", "capsules", "cashew", "chewinggum", "fryum",
                   "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
    ),
}

# ===================================================================
# MODEL CONFIGURATIONS
# ===================================================================

MODEL_CONFIGS = {
    "unet_ae": SimpleNamespace(
        model_type="unet_ae",
        model_params=dict(latent_dim=512, backbone="resnet18"),
        loss_type="ae",
        loss_params=dict(),
        modeler_type="unet_ae",
        modeler_params=dict(),
        trainer_type="gradient",
        trainer_params=dict(epochs=100, lr=1e-3),
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_type="step",
        scheduler_params=dict(step_size=30, gamma=0.1),
        stopper_type="early",
        stopper_params=dict(patience=10, min_delta=1e-4),
    ),
    "stfpm": SimpleNamespace(
        model_type="stfpm",
        model_params=dict(backbone="resnet18", layers=["layer1", "layer2", "layer3"]),
        loss_type="stfpm",
        loss_params=dict(),
        modeler_type="stfpm",
        modeler_params=dict(),
        trainer_type="gradient",
        trainer_params=dict(epochs=100, lr=1e-3),
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_type="step",
        scheduler_params=dict(step_size=30, gamma=0.1),
        stopper_type="early",
        stopper_params=dict(patience=15, min_delta=1e-4),
    ),
    "patchcore": SimpleNamespace(
        model_type="patchcore",
        model_params=dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], 
                         n_neighbors=9, coreset_sampling_ratio=0.1),
        loss_type="none",
        loss_params=dict(),
        modeler_type="patchcore",
        modeler_params=dict(),
        trainer_type="memory",
        trainer_params=dict(),
        optimizer_type="none",
        optimizer_params=dict(),
        scheduler_type="none",
        scheduler_params=dict(),
        stopper_type="none",
        stopper_params=dict(),
    ),
    "padim": SimpleNamespace(
        model_type="padim",
        model_params=dict(backbone="resnet18", layers=["layer2", "layer3"]),
        loss_type="none",
        loss_params=dict(),
        modeler_type="padim",
        modeler_params=dict(),
        trainer_type="memory",
        trainer_params=dict(),
        optimizer_type="none",
        optimizer_params=dict(),
        scheduler_type="none",
        scheduler_params=dict(),
        stopper_type="none",
        stopper_params=dict(),
    ),
    "fastflow": SimpleNamespace(
        model_type="fastflow",
        model_params=dict(backbone="resnet18", flow_steps=8, hidden_ratio=1.0),
        loss_type="fastflow",
        loss_params=dict(),
        modeler_type="fastflow",
        modeler_params=dict(),
        trainer_type="flow",
        trainer_params=dict(epochs=500, lr=1e-3),
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_type="cosine",
        scheduler_params=dict(T_max=500, eta_min=1e-6),
        stopper_type="early",
        stopper_params=dict(patience=50, min_delta=1e-5),
    ),
    "draem": SimpleNamespace(
        model_type="draem",
        model_params=dict(backbone="resnet18", anomaly_source_path="dtd"),
        loss_type="draem",
        loss_params=dict(l_rec_weight=1.0, l_ssim_weight=1.0),
        modeler_type="draem",
        modeler_params=dict(),
        trainer_type="gradient",
        trainer_params=dict(epochs=700, lr=1e-3),
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_type="step",
        scheduler_params=dict(step_size=200, gamma=0.5),
        stopper_type="early",
        stopper_params=dict(patience=100, min_delta=1e-4),
    ),
}

# ===================================================================
# COMBINATION-SPECIFIC OVERRIDES
# ===================================================================

COMBINATION_OVERRIDES = {
    # PatchCore needs smaller batches for memory efficiency
    ("patchcore", "btad"): SimpleNamespace(
        train_batch_size=4, 
        test_batch_size=4,
        model_params=dict(n_neighbors=5)
    ),
    ("patchcore", "visa"): SimpleNamespace(
        train_batch_size=6, 
        test_batch_size=6
    ),
    ("patchcore", "mvtec"): SimpleNamespace(
        train_batch_size=8, 
        test_batch_size=8
    ),
    
    # FastFlow optimization with smaller batches
    ("fastflow", "btad"): SimpleNamespace(
        train_batch_size=4, 
        test_batch_size=4,
        trainer_params=dict(epochs=300, lr=5e-4)
    ),
    ("fastflow", "visa"): SimpleNamespace(
        train_batch_size=6, 
        test_batch_size=6
    ),
    ("fastflow", "mvtec"): SimpleNamespace(
        train_batch_size=8, 
        test_batch_size=8
    ),
    
    # UNet AutoEncoder works well with larger batches on simple datasets
    ("unet_ae", "mvtec"): SimpleNamespace(
        train_batch_size=32, 
        test_batch_size=32,
        trainer_params=dict(epochs=150, lr=2e-3)
    ),
    
    # STFPM optimizations
    ("stfpm", "mvtec"): SimpleNamespace(
        train_batch_size=24, 
        test_batch_size=24
    ),
    
    # DRAEM needs specific tuning
    ("draem", "mvtec"): SimpleNamespace(
        trainer_params=dict(epochs=500)
    ),
}

# ===================================================================
# CONFIGURATION UTILITIES
# ===================================================================

def merge_configs(destination, source):
    """Merge config objects with smart dict merging for *_params"""
    for key, value in source.__dict__.items():
        if key.endswith("_params") and hasattr(destination, key):
            existing_params = getattr(destination, key)
            if isinstance(existing_params, dict) and isinstance(value, dict):
                existing_params.update(value)
            else:
                setattr(destination, key, value)
        else:
            setattr(destination, key, value)
    return destination

def build_config(dataset_type, model_type, overrides=None):
    """Build final config: BASE_CONFIGS <- DATASET <- MODEL <- COMBINATION <- overrides"""
    if dataset_type not in DATASET_CONFIGS:
        available_datasets = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Available: {available_datasets}")
    if model_type not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model_type: {model_type}. Available: {available_models}")
    
    config = SimpleNamespace()
    merge_configs(config, BASE_CONFIGS)
    merge_configs(config, DATASET_CONFIGS[dataset_type])
    merge_configs(config, MODEL_CONFIGS[model_type])
    
    # Apply combination-specific overrides
    combination_key = (model_type, dataset_type)
    if combination_key in COMBINATION_OVERRIDES:
        merge_configs(config, COMBINATION_OVERRIDES[combination_key])
    
    # Normalize dataloader section
    config.dataloader = dict(
        img_size=config.img_size,
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    # Apply user overrides last
    if overrides:
        for key, value in overrides.items():
            if key.endswith("_params") and hasattr(config, key):
                existing_params = getattr(config, key)
                if isinstance(existing_params, dict) and isinstance(value, dict):
                    existing_params.update(value)
                else:
                    setattr(config, key, value)
            else:
                setattr(config, key, value)
    
    return config

# ===================================================================
# DEBUG UTILITIES
# ===================================================================

def print_config(config):
    print(f"\n=== Config: {config.dataset_type} + {config.model_type} ===")
    print(f"  Batch: train={config.train_batch_size}, test={config.test_batch_size}")
    print(f"  Image size: {config.img_size}")
    print(f"  Epochs: {config.trainer_params.get('epochs', 'N/A')}")
    print(f"  Learning rate: {config.trainer_params.get('lr', 'N/A')}")
    print(f"  Optimizer: {config.optimizer_type}")
    print(f"  Scheduler: {config.scheduler_type}")
    print(f"  Stopper: {config.stopper_type}")
    print(f"  Model params: {config.model_params}")
    print(f"  Modeler params: {config.modeler_params}")
    print("=" * 50)


def get_available_types():
    """Get available dataset/model types"""
    available_datasets = list(DATASET_CONFIGS.keys())
    available_models = list(MODEL_CONFIGS.keys())
    return available_datasets, available_models


def print_all_combinations():
    """Print all possible combinations"""
    available_datasets, available_models = get_available_types()
    
    print("=" * 60)
    print("ALL POSSIBLE COMBINATIONS")
    print("=" * 60)
    
    total_combinations = 0
    for dataset_type in available_datasets:
        for model_type in available_models:
            total_combinations += 1
            print(f"{total_combinations:2d}. {dataset_type:8s} + {model_type:10s}")
    
    print("=" * 60)
    print(f"Total combinations: {total_combinations}")
    
    # Show combination-specific overrides
    if COMBINATION_OVERRIDES:
        print("\nCombination-specific overrides:")
        for (model, dataset), override in COMBINATION_OVERRIDES.items():
            print(f"  {dataset:8s} + {model:10s}: {list(override.__dict__.keys())}")
    
    print("=" * 60)


def validate_all_configs():
    available_datasets, available_models = get_available_types()
    
    print("=" * 60)
    print("VALIDATING ALL CONFIGURATIONS")
    print("=" * 60)
    
    valid_count = 0
    invalid_count = 0
    
    for dataset_type in available_datasets:
        for model_type in available_models:
            try:
                config = build_config(dataset_type, model_type)
                print(f"[OK] {dataset_type:8s} + {model_type:10s}")
                valid_count += 1
            except Exception as e:
                print(f"[FAIL] {dataset_type:8s} + {model_type:10s} -> {e}")
                invalid_count += 1
    
    print("=" * 60)
    print(f"Valid configurations: {valid_count}")
    print(f"Invalid configurations: {invalid_count}")
    print("=" * 60)

# ===================================================================
# MAIN (FOR TESTING)
# ===================================================================


if __name__ == "__main__":
    # Test configuration system
    print_all_combinations()
    validate_all_configs()
    
    # Test specific configurations
    print_config("mvtec", "unet_ae")
    print_config("patchcore", "btad")