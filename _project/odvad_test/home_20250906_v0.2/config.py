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