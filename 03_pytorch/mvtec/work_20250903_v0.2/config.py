import os
from types import SimpleNamespace

# ===================================================================
# BASE CONFIGURATION
# ===================================================================

BASE_CONFIGS = SimpleNamespace(
    backbone_dir="/mnt/d/github/deep_learning_tutorials/03_pytorch/backbones",
    current_dir=os.getcwd(),
    output_dir="./experiments",
    num_epochs=5,
    seed=42,
    verbose=True,
)

# ===================================================================
# DATASET CONFIGURATIONS
# ===================================================================

DATA_CONFIGS = {
    "mvtec": SimpleNamespace(
        dataloader_type="mvtec",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/mvtec",
            categories=["bottle", "cable", "capsule"],
            train_batch_size=16, test_batch_size=8,
            test_ratio=0.2, valid_ratio=0.2,
            num_workers=8, pin_memory=True, persistent_workers=True,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
    "btad": SimpleNamespace(
        dataloader_type="btad",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/btad",
            categories=["01", "02", "03"],
            train_batch_size=8, test_batch_size=8,
            test_ratio=0.2, valid_ratio=0.2,
            num_workers=8, pin_memory=True, persistent_workers=True,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
    "visa": SimpleNamespace(
        dataloader_type="visa",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/visa",
            categories=["candle", "capsules", "cashew", "chewinggum", "fryum",
                        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
            train_batch_size=4, test_batch_size=4,
            test_ratio=0.2, valid_ratio=0.2,
            num_workers=8, pin_memory=True, persistent_workers=True,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
}

# ===================================================================
# MODEL CONFIGURATIONS
# ===================================================================

MODEL_CONFIGS = {
    "vanilla_ae": SimpleNamespace(
        modeler_type="ae",
        model_type="vanilla_ae",
        model_params=dict(),
        loss_type="combined",
        loss_params=dict(),
        metric_list=[("psnr", dict()), ("ssim", dict())],   # (metric_type, matric_params)
    ),
    "unet_ae": SimpleNamespace(
        modeler_type="ae",
        model_type="unet_ae",
        model_params=dict(),
        loss_type="combined",
        loss_params=dict(),
        metric_list=[("psnr", dict()), ("ssim", dict())],   # (metric_type, matric_params)
    ),
    "stfpm": SimpleNamespace(
        modeler_type="stfpm",
        model_type="stfpm",
        model_params=dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"]),
        loss_type="stfpm",
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],   # (metric_type, matric_params)
    ),
    "padim": SimpleNamespace(
        modeler_type="padim",
        model_type="padim",
        model_params=dict(backbone="resnet18", layers=["layer2", "layer3"]),
        loss_type="none",
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],   # (metric_type, matric_params)
    ),
}

# ===================================================================
# MODEL CONFIGURATIONS
# ===================================================================

TRAIN_CONFIS = {
    "gradient": SimpleNamespace(
        trainer_type="gradient",
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_type="step",
        scheduler_params=dict(),
        stopper_type="early",
        stopper_params=dict(patience=5, min_delta=1e-4),
    ),
    "memory": SimpleNamespace(
        trainer_type="memory",
        trainer_params=dict(),
        optimizer_type="none",
        optimizer_params=dict(),
        scheduler_type="none",
        scheduler_params=dict(),
        stopper_type="none",
        stopper_params=dict(),
    ),
}


def show_config(config):
    print(f"\n=== Config: {config.dataloader_type} + {config.model_type} + {config.trainer_type} ===")
    for key, value in config.__dict__.items():
        if key in ("dataloader_type", "model_type", "trainer_type"):
            print()
        print(f"  {key}: {value}")
    print("=" * 50)


def merge_configs(destination, source):
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


def build_config(data_type, model_type, train_type):
    config = SimpleNamespace()
    config = merge_configs(config, BASE_CONFIGS)
    config = merge_configs(config, DATA_CONFIGS[data_type])
    config = merge_configs(config, MODEL_CONFIGS[model_type])
    config = merge_configs(config, TRAIN_CONFIS[train_type])

    if model_type in ["patchcore", "padim"]:    # memory-based models
        config.num_epochs = 1
        config.dataloader_params["train_batch_size"] = 4
        config.dataloader_params["test_batch_size"] = 2
        config.dataloader_params["valid_ratio"] = 0.0
        config.dataloader_params["train_shuffle"] = False
        config.dataloader_params["test_shuffle"] = False
        config.dataloader_params["train_drop_last"] = False
        config.dataloader_params["test_drop_last"] = False

    return config


def get_available_types():
    """Get available dataset/model types"""
    available_data = list(DATA_CONFIGS.keys())
    available_models = list(MODEL_CONFIGS.keys())
    available_trainers = list(TRAIN_CONFIS.keys())
    return available_data, available_models, available_trainers


if __name__ == "__main__":
    pass