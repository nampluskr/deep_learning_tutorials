from types import SimpleNamespace
from datetime import datetime


# ===================================================================
# Global Configuration
# ===================================================================

BASE_CONFIGS = SimpleNamespace(
    backbone_dir="/mnt/d/backbones",
    output_dir="./experiments",
    log_name="experiment.log",
    num_epochs=10,
    seed=42,

    verbose=True,
    show_dataloader=True,
    show_modeler=True,
    show_trainer=True,
    show_memory=True,

    save_log=True,
    save_config=True,
    save_model=True,
    save_history=True,
    save_results=True,

)


# ===================================================================
# Data Configuration
# ===================================================================

DATA_CONFIGS = {
    "mvtec": SimpleNamespace(
        dataloader_type="mvtec",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/mvtec",
            categories=["bottle", "cable", "capsule"],
            train_batch_size=4, test_batch_size=2,
            test_ratio=0.15, valid_ratio=0.15,
            num_workers=4, pin_memory=True, persistent_workers=True,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
}


# ===================================================================
# Model Configuration
# ===================================================================

MODEL_CONFIGS = {
    "vanilla_ae": SimpleNamespace(
        modeler_type="ae",
        trainer_type="reconstruction",

        model_type="vanilla_ae",
        model_params=dict(),
        loss_type="ae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "unet_ae": SimpleNamespace(
        modeler_type="ae",
        trainer_type="reconstruction",

        model_type="unet_ae",
        model_params=dict(),
        loss_type="ae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "vanilla_vae": SimpleNamespace(
        modeler_type="vae",
        trainer_type="reconstruction",

        model_type="vanilla_vae",
        model_params=dict(),
        loss_type="vae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "unet_vae": SimpleNamespace(
        modeler_type="vae",
        trainer_type="reconstruction",

        model_type="unet_vae",
        model_params=dict(),
        loss_type="vae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "stfpm": SimpleNamespace(
        modeler_type="stfpm",
        trainer_type="reconstruction",

        model_type="stfpm",
        model_params=dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"]),
        loss_type="stfpm",
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],   # (metric_type, matric_params)
    ),
    "fastflow": SimpleNamespace(
        modeler_type="fastflow",
        trainer_type="reconstruction",

        model_type="fastflow",
        model_params=dict(
            backbone="resnet18",
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0
        ),
        loss_type="fastflow",
        loss_params=dict(),
        metric_list=[("likelihood", dict())],
    ),
    "draem": SimpleNamespace(
        modeler_type="draem",
        trainer_type="reconstruction",
        model_type="draem",
        model_params=dict(sspcab=False),
        loss_type="draem",
        loss_params=dict(),
        metric_list=[("auroc", dict()), ("aupr", dict())],
    ),
    "efficientad": SimpleNamespace(
        modeler_type="efficientad",
        trainer_type="reconstruction",

        model_type="efficientad",
        model_params=dict(
            model_size="small",  # "small" or "medium"
            teacher_out_channels=384,
            padding=False,
            pad_maps=True,
            use_imagenet_penalty=True,
        ),
        loss_type=None,  # EfficientAD는 자체 loss 계산
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],
    ),
}


# ===================================================================
# Trainer Configuration
# ===================================================================


TRAIN_CONFIGS = {
    "reconstruction": SimpleNamespace(
        trainer_type="reconstruction",
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-4, weight_decay=1e-5),
        scheduler_type="plateau",
        scheduler_params=dict(),
        stopper_type="early",
        stopper_params=dict(patience=5, min_delta=1e-4),
    ),
}


# ===================================================================
# Config Utilities
# ===================================================================


def build_config(data_type, model_type):
    config = SimpleNamespace()
    config = merge_configs(config, BASE_CONFIGS)
    config = merge_configs(config, DATA_CONFIGS[data_type])

    model_config = MODEL_CONFIGS[model_type]
    config = merge_configs(config, model_config)

    trainer_type = model_config.trainer_type
    config = merge_configs(config, TRAIN_CONFIGS[trainer_type])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = os.path.join(config.output_dir, f"{timestamp}_{data_type.lower()}_{model_type.lower()}")
    return config


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
