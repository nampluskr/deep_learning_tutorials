import os
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
    "default": SimpleNamespace(
        dataloader_params=dict(
            test_ratio=0.2, valid_ratio=0.15,
            num_workers=4, pin_memory=True, persistent_workers=True,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
    "mvtec": SimpleNamespace(
        dataloader_type="mvtec",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/mvtec",
            categories=["bottle", "cable", "capsule", "carpet", "grid",
                       "hazelnut", "leather", "metal_nut", "pill", "screw",
                       "tile", "toothbrush", "transistor", "wood", "zipper"],
            train_batch_size=8, test_batch_size=4,
        ),
    ),
    "visa": SimpleNamespace(
        dataloader_type="visa",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/visa",
            categories=["candle", "cashew", "chewing_gum", "fryum", "macaroni1",
                       "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
            train_batch_size=12, test_batch_size=6,
        ),
    ),
    "btad": SimpleNamespace(
        dataloader_type="btad",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/btad",
            categories=["01", "02", "03"],
            train_batch_size=6, test_batch_size=3,
        ),
    ),
}


# ===================================================================
# Model Configuration (Unified: MODEL_CONFIGS + TRAIN_CONFIGS)
# ===================================================================

MODEL_CONFIGS = {
    "default": SimpleNamespace(
        trainer_type="reconstruction",
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-2),
        scheduler_type="plateau",
        scheduler_params=dict(factor=0.5, patience=3, min_lr=1e-7),
        stopper_type="early",
        stopper_params=dict(patience=5, min_delta=1e-4),
        num_epochs=10,  # Default for testing
        metric_list=[("psnr", dict())],
        model_params=dict(),
        loss_params=dict(),
    ),

    # AutoEncoder family - 충분한 epochs + early stopping
    "vanilla_ae": SimpleNamespace(
        modeler_type="ae",
        model_type="vanilla_ae",
        loss_type="ae",
        num_epochs=100,  # 50 → 100 (early stopping으로 실제로는 30-40에서 종료 예상)
        optimizer_type="adamw",
        optimizer_params=dict(lr=5e-4, weight_decay=1e-3),
        scheduler_type="step",
        scheduler_params=dict(step_size=15, gamma=0.5),
        stopper_type="early",
        stopper_params=dict(patience=8, min_delta=1e-4),
    ),

    "unet_ae": SimpleNamespace(
        modeler_type="ae",
        model_type="unet_ae",
        loss_type="ae",
        num_epochs=120,  # 60 → 120 (UNet은 좀 더 복잡하므로)
        optimizer_type="rmsprop",
        optimizer_params=dict(lr=1e-3, alpha=0.99, weight_decay=1e-5),
        scheduler_type="multistep",
        scheduler_params=dict(milestones=[30, 60, 90], gamma=0.3),  # milestone도 조정
        stopper_type="early",
        stopper_params=dict(patience=10, min_delta=1e-4),
    ),

    # VAE family - KL divergence 안정화를 위해 더 많은 epochs
    "vanilla_vae": SimpleNamespace(
        modeler_type="vae",
        model_type="vanilla_vae",
        loss_type="vae",
        num_epochs=150,  # 80 → 150 (VAE는 더 긴 훈련 필요할 수 있음)
        optimizer_type="adam_clip",
        optimizer_params=dict(lr=5e-4, max_grad_norm=1.0),
        scheduler_type="cosine",
        scheduler_params=dict(T_max=150, eta_min=1e-6),  # T_max도 조정
        stopper_type="early",
        stopper_params=dict(patience=15, min_delta=5e-5),  # patience도 증가
    ),

    "unet_vae": SimpleNamespace(
        modeler_type="vae",
        model_type="unet_vae",
        loss_type="vae",
        num_epochs=180,  # 100 → 180
        optimizer_type="adam_clip",
        optimizer_params=dict(lr=3e-4, max_grad_norm=1.0),
        scheduler_type="cosine",
        scheduler_params=dict(T_max=180, eta_min=1e-6),
        stopper_type="early",
        stopper_params=dict(patience=20, min_delta=5e-5),
    ),

    # Feature distillation - Teacher-Student 안정화를 위해 충분한 epochs
    "stfpm": SimpleNamespace(
        modeler_type="stfpm",
        model_type="stfpm",
        loss_type="stfpm",
        model_params=dict(
            backbone="resnet50",
            layers=["layer1", "layer2", "layer3"]
        ),
        num_epochs=200,  # 100 → 200
        optimizer_type="sgd",
        optimizer_params=dict(lr=1e-2, momentum=0.9, weight_decay=1e-4),
        scheduler_type="multistep",
        scheduler_params=dict(milestones=[60, 120, 160], gamma=0.1),  # milestone 조정
        stopper_type="early",
        stopper_params=dict(patience=25, min_delta=1e-5),  # patience 증가
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],
    ),

    # Normalizing flow - early stopping 없이 full epochs (변경 없음)
    "fastflow": SimpleNamespace(
        modeler_type="fastflow",
        model_type="fastflow",
        loss_type="fastflow",
        model_params=dict(
            backbone="wide_resnet50_2",
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
        ),
        num_epochs=200,  # 150 → 200 (early stopping 없으므로 적당히 증가)
        optimizer_type="rmsprop",
        optimizer_params=dict(lr=1e-3, alpha=0.99, weight_decay=1e-5),
        scheduler_type="exponential",
        scheduler_params=dict(gamma=0.99),
        stopper_type="none",  # No early stopping
        metric_list=[("likelihood", dict())],
    ),

    # Synthetic anomaly generation
    "draem": SimpleNamespace(
        modeler_type="draem",
        model_type="draem",
        loss_type="draem",
        model_params=dict(sspcab=False),
        num_epochs=150,  # 80 → 150
        optimizer_type="adam",
        optimizer_params=dict(lr=1e-4, betas=(0.5, 0.999), weight_decay=0),
        scheduler_type="multistep",
        scheduler_params=dict(milestones=[50, 100, 130], gamma=0.5),  # milestone 조정
        stopper_type="early",
        stopper_params=dict(patience=20, min_delta=1e-5),  # patience 증가
        metric_list=[("auroc", dict()), ("aupr", dict())],
    ),

    # Knowledge distillation - early stopping 없이 full epochs
    "efficientad": SimpleNamespace(
        modeler_type="efficientad",
        model_type="efficientad",
        loss_type=None,
        model_params=dict(
            model_size="small",
            teacher_out_channels=384,
            padding=False,
            pad_maps=True,
            use_imagenet_penalty=True,
        ),
        num_epochs=100,  # 70 → 100
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-2),
        scheduler_type="cosine",
        scheduler_params=dict(T_max=100, eta_min=1e-6),  # T_max 조정
        stopper_type="early",
        stopper_params=dict(patience=15, min_delta=1e-5),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],
    ),
}

# ===================================================================
# Config Utilities
# ===================================================================


def build_config(data_type, model_type):
    config = SimpleNamespace()
    config = merge_configs(config, BASE_CONFIGS)
    config = merge_configs(config, DATA_CONFIGS["default"])
    config = merge_configs(config, DATA_CONFIGS[data_type])
    config = merge_configs(config, MODEL_CONFIGS["default"])
    config = merge_configs(config, MODEL_CONFIGS[model_type])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = os.path.join(config.output_dir, f"{timestamp}_{data_type.lower()}_{model_type.lower()}")
    return config


def merge_configs(destination, source):
    import copy
    
    if isinstance(source, dict):
        source = SimpleNamespace(**source)
    else:
        source = copy.deepcopy(source)
    
    for key, value in source.__dict__.items():
        if key == "dataloader_params" and hasattr(destination, key):
            # dataloader_params만 병합(merge)
            existing_params = getattr(destination, key)
            if isinstance(existing_params, dict) and isinstance(value, dict):
                existing_params.update(value)
            else:
                setattr(destination, key, value)
        elif key.endswith("_params") and hasattr(destination, key):
            # 다른 _params는 치환(replace)
            setattr(destination, key, value)
        else:
            # 일반 속성은 치환
            setattr(destination, key, value)
    
    return destination


def save_config(config, output_dir, logger=None):
    """Save experiment configuration to JSON file."""
    import json
    from datetime import datetime

    folder_name = os.path.basename(os.path.normpath(output_dir))
    config_path = os.path.join(output_dir, f"config_{folder_name}.json")

    try:
        config_dict = {}    # Convert config to dictionary
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if isinstance(value, (int, float, str, bool, list, dict)):
                    config_dict[attr] = value
                elif hasattr(value, '__dict__'):
                    config_dict[attr] = str(value)
                else:
                    config_dict[attr] = str(value)

        config_data = {
            'experiment_name': folder_name,
            'configuration': config_dict
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        filename = os.path.basename(config_path)
        success_msg = f" > Config saved: {filename}"
        print(success_msg)
        if logger:
            logger.info(success_msg)

    except Exception as e:
        error_msg = f" > Config save failed: {str(e)}"
        print(error_msg)
        if logger:
            logger.error(error_msg)


def load_config(output_dir, logger=None, config_filename=None):
    """Load experiment configuration from JSON file."""
    import json
    from glob import glob
    from types import SimpleNamespace

    def dict_to_namespace_with_params(d):
        """Dict를 SimpleNamespace로 변환하되, _params 키는 dict로 유지"""
        if isinstance(d, dict):
            ns = SimpleNamespace()
            for key, value in d.items():
                if key.endswith('_params'):
                    # _params로 끝나는 키는 dict로 유지
                    setattr(ns, key, value)
                elif isinstance(value, dict):
                    # 중첩된 dict는 재귀적으로 SimpleNamespace로 변환
                    setattr(ns, key, dict_to_namespace_with_params(value))
                elif isinstance(value, list):
                    # 리스트 내 dict도 처리
                    setattr(ns, key, [dict_to_namespace_with_params(item) if isinstance(item, dict) else item for item in value])
                else:
                    # 기본 타입은 그대로
                    setattr(ns, key, value)
            return ns
        else:
            return d

    try:
        if config_filename:
            # 특정 파일명이 지정된 경우
            config_path = os.path.join(output_dir, config_filename)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_filename}")
        else:
            # config_*.json 패턴으로 가장 최신 파일 찾기
            config_pattern = os.path.join(output_dir, "config_*.json")
            config_files = glob(config_pattern)

            if not config_files:
                raise FileNotFoundError(f"No config files found in: {output_dir}")

            # 파일명의 timestamp 기준으로 가장 최신 파일 선택
            config_path = max(config_files, key=os.path.getctime)

        # JSON 파일 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # configuration 섹션에서 설정 추출
        if 'configuration' in config_data:
            config_dict = config_data['configuration']
        else:
            config_dict = config_data  # 호환성을 위해

        # SimpleNamespace로 변환 (_params는 dict로 유지)
        config = dict_to_namespace_with_params(config_dict)

        filename = os.path.basename(config_path)
        success_msg = f" > Config loaded: {filename}"
        print(success_msg)
        if logger:
            logger.info(success_msg)

        return config

    except Exception as e:
        error_msg = f" > Config load failed: {str(e)}"
        print(error_msg)
        if logger:
            logger.error(error_msg)
        return None