"""
설정 관리 모듈
Config 클래스와 관련 유틸리티 함수들
"""

import torch
from dataclasses import dataclass, asdict, fields
import json
import os
import datetime
from typing import List, Optional, Dict, Any

@dataclass
class Config:
    # 필수 설정
    data_dir: str = '/mnt/d/datasets/mvtec'
    seed: int = 42
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size: int = 256
    normalize: bool = False
    category: str = 'bottle'
    batch_size: int = 32
    valid_ratio: float = 0.2
    
    # 모델 설정
    model_type: str = 'vanilla_ae'  # 새로 추가된 필드
    latent_dim: int = 512
    
    # 훈련 설정
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # 실험 설정
    save_models: bool = True
    save_results: bool = True
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_type}_{self.category}_{self.latent_dim}_{self.img_size}"


# ========================
# 설정 출력 함수들
# ========================

def print_config(config, style="simple"):
    """
    설정값을 출력합니다.
    
    Args:
        config: Config 인스턴스
        style: 출력 스타일 ("simple", "detailed", "grouped", "json", "summary")
    """
    if style == "simple":
        _print_simple(config)
    elif style == "detailed":
        _print_detailed(config)
    elif style == "grouped":
        _print_grouped(config)
    elif style == "json":
        _print_json(config)
    elif style == "summary":
        _print_summary(config)
    else:
        raise ValueError(f"Unknown style: {style}. Available: simple, detailed, grouped, json, summary")


def _print_simple(config):
    """간단한 형태로 출력"""
    print("Configuration:")
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {field.name}: {value}")


def _print_detailed(config):
    """상세한 형태로 출력 (박스 형태)"""
    print("=" * 60)
    print("🔧 EXPERIMENT CONFIGURATION")
    print("=" * 60)
    
    for field in fields(config):
        value = getattr(config, field.name)
        # 값의 타입에 따라 다른 포맷팅
        if isinstance(value, str):
            print(f"  {field.name:<20}: '{value}'")
        elif isinstance(value, float):
            print(f"  {field.name:<20}: {value:.6f}")
        else:
            print(f"  {field.name:<20}: {value}")
    
    print("=" * 60)


def _print_grouped(config):
    """카테고리별로 그룹화하여 출력"""
    # 설정을 논리적 그룹으로 분류
    data_fields = ['data_dir', 'img_size', 'normalize', 'category', 'batch_size', 'valid_ratio']
    model_fields = ['latent_dim']
    train_fields = ['num_epochs', 'learning_rate', 'weight_decay']
    experiment_fields = ['save_models', 'save_results', 'experiment_name']
    
    print("=" * 50)
    print("🔧 EXPERIMENT CONFIGURATION")
    print("=" * 50)
    
    # 데이터 설정
    print("\n📁 Data Configuration:")
    print("-" * 25)
    for field_name in data_fields:
        if hasattr(config, field_name):
            value = getattr(config, field_name)
            print(f"  {field_name:<15}: {value}")
    
    # 모델 설정
    print("\n🧠 Model Configuration:")
    print("-" * 26)
    for field_name in model_fields:
        if hasattr(config, field_name):
            value = getattr(config, field_name)
            print(f"  {field_name:<15}: {value}")
    
    # 훈련 설정
    print("\n🏃 Training Configuration:")
    print("-" * 29)
    for field_name in train_fields:
        if hasattr(config, field_name):
            value = getattr(config, field_name)
            if isinstance(value, float):
                print(f"  {field_name:<15}: {value:.6f}")
            else:
                print(f"  {field_name:<15}: {value}")
    
    # 실험 설정
    print("\n⚙️  Experiment Configuration:")
    print("-" * 32)
    for field_name in experiment_fields:
        if hasattr(config, field_name):
            value = getattr(config, field_name)
            print(f"  {field_name:<15}: {value}")
    
    print("\n" + "=" * 50)


def _print_json(config):
    """JSON 형태로 출력"""
    print("Configuration (JSON format):")
    print(json.dumps(asdict(config), indent=2))


def _print_summary(config):
    """핵심 설정만 간단히 출력"""
    print(f"🎯 Experiment: {config.experiment_name}")
    print(f"📂 Dataset: {config.category} ({config.img_size}x{config.img_size})")
    print(f"🔧 Model: Latent={config.latent_dim}, Batch={config.batch_size}")
    print(f"⏱️  Training: {config.num_epochs} epochs, LR={config.learning_rate}")


def print_config_comparison(configs, names=None):
    """
    여러 설정을 비교하여 출력
    
    Args:
        configs: Config 인스턴스들의 리스트
        names: 각 설정의 이름 (없으면 자동 생성)
    """
    if names is None:
        names = [f"Config {i+1}" for i in range(len(configs))]
    
    if len(configs) != len(names):
        raise ValueError("configs와 names의 길이가 일치하지 않습니다.")
    
    print("=" * 80)
    print("🔄 CONFIGURATION COMPARISON")
    print("=" * 80)
    
    # 모든 필드 수집
    all_fields = []
    for config in configs:
        for field in fields(config):
            if field.name not in all_fields:
                all_fields.append(field.name)
    
    # 헤더 출력
    header = f"{'Field':<20}"
    for name in names:
        header += f"{name:<20}"
    print(header)
    print("-" * len(header))
    
    # 각 필드별 값 비교
    for field_name in all_fields:
        row = f"{field_name:<20}"
        for config in configs:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if isinstance(value, float):
                    row += f"{value:<20.6f}"
                else:
                    row += f"{str(value):<20}"
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    print("=" * 80)


# ========================
# 설정 저장/로드 함수들
# ========================

def save_config(config, path, format="json"):
    """
    설정을 파일로 저장합니다.
    
    Args:
        config: Config 인스턴스
        path: 저장할 파일 경로
        format: 저장 형식 ("json", "yaml", "txt")
    """
    # 디렉토리가 없으면 생성
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    if format == "json":
        _save_config_json(config, path)
    elif format == "yaml":
        _save_config_yaml(config, path)
    elif format == "txt":
        _save_config_txt(config, path)
    else:
        raise ValueError(f"Unsupported format: {format}. Available: json, yaml, txt")


def _save_config_json(config, path):
    """JSON 형식으로 설정 저장"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    print(f"Configuration saved as JSON: {path}")


def _save_config_yaml(config, path):
    """YAML 형식으로 설정 저장"""
    try:
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)
        print(f"Configuration saved as YAML: {path}")
    except ImportError:
        print("PyYAML is not installed. Install with: pip install pyyaml")
        raise


def _save_config_txt(config, path):
    """텍스트 형식으로 설정 저장 (사람이 읽기 쉬운 형태)"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 50 + "\n\n")
        
        # 그룹별로 저장
        data_fields = ['data_dir', 'img_size', 'normalize', 'category', 'batch_size', 'valid_ratio']
        model_fields = ['latent_dim']
        train_fields = ['num_epochs', 'learning_rate', 'weight_decay']
        experiment_fields = ['save_models', 'save_results', 'experiment_name']
        
        f.write("Data Configuration:\n")
        f.write("-" * 20 + "\n")
        for field_name in data_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                f.write(f"  {field_name}: {value}\n")
        
        f.write("\nModel Configuration:\n")
        f.write("-" * 21 + "\n")
        for field_name in model_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                f.write(f"  {field_name}: {value}\n")
        
        f.write("\nTraining Configuration:\n")
        f.write("-" * 24 + "\n")
        for field_name in train_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                f.write(f"  {field_name}: {value}\n")
        
        f.write("\nExperiment Configuration:\n")
        f.write("-" * 27 + "\n")
        for field_name in experiment_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                f.write(f"  {field_name}: {value}\n")
    
    print(f"Configuration saved as TXT: {path}")


def load_config(path, format="auto"):
    """
    파일에서 설정을 로드합니다.
    
    Args:
        path: 로드할 파일 경로
        format: 파일 형식 ("auto", "json", "yaml")
    
    Returns:
        Config 인스턴스
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # 자동 형식 감지
    if format == "auto":
        if path.endswith('.json'):
            format = "json"
        elif path.endswith(('.yaml', '.yml')):
            format = "yaml"
        else:
            format = "json"  # 기본값
    
    if format == "json":
        return _load_config_json(path)
    elif format == "yaml":
        return _load_config_yaml(path)
    else:
        raise ValueError(f"Unsupported format: {format}. Available: json, yaml")


def _load_config_json(path):
    """JSON 파일에서 설정 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    print(f"Configuration loaded from JSON: {path}")
    return Config(**config_dict)


def _load_config_yaml(path):
    """YAML 파일에서 설정 로드"""
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"Configuration loaded from YAML: {path}")
        return Config(**config_dict)
    except ImportError:
        print("PyYAML is not installed. Install with: pip install pyyaml")
        raise


# ========================
# 설정 관리 유틸리티 함수들
# ========================

def update_config(config, **kwargs):
    """
    설정을 업데이트합니다.
    
    Args:
        config: Config 인스턴스
        **kwargs: 업데이트할 설정값들
    
    Returns:
        새로운 Config 인스턴스 (원본 유지)
    """
    # 현재 설정을 딕셔너리로 변환
    current_dict = asdict(config)
    
    # 새로운 값들로 업데이트
    for key, value in kwargs.items():
        if key in current_dict:
            current_dict[key] = value
        else:
            print(f"Warning: Unknown config key '{key}' ignored")
    
    # 새로운 인스턴스 생성
    return Config(**current_dict)


def merge_configs(base_config, override_config):
    """
    두 설정을 병합합니다.
    
    Args:
        base_config: 기본 설정
        override_config: 덮어쓸 설정
    
    Returns:
        병합된 새로운 Config 인스턴스
    """
    base_dict = asdict(base_config)
    override_dict = asdict(override_config)
    
    # override_config의 None이 아닌 값들로 업데이트
    for key, value in override_dict.items():
        if value is not None:
            base_dict[key] = value
    
    return Config(**base_dict)


def backup_config(config, backup_dir="config_backups"):
    """
    설정을 백업합니다.
    
    Args:
        config: Config 인스턴스
        backup_dir: 백업 디렉토리
    
    Returns:
        백업 파일 경로
    """
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"config_{config.experiment_name}_{timestamp}.json"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    save_config(config, backup_path, format="json")
    print(f"Configuration backed up: {backup_path}")
    
    return backup_path


def validate_config(config):
    """
    설정값의 유효성을 검증합니다.
    
    Args:
        config: Config 인스턴스
    
    Returns:
        bool: 유효하면 True, 그렇지 않으면 False
    """
    errors = []
    
    # 데이터 경로 확인
    if not os.path.exists(config.data_dir):
        errors.append(f"Data directory does not exist: {config.data_dir}")
    
    # 배치 크기 확인
    if config.batch_size <= 0 or config.batch_size > 512:
        errors.append(f"Invalid batch_size: {config.batch_size} (must be 1-512)")
    
    # 이미지 크기 확인
    if config.img_size not in [64, 128, 256, 512, 1024]:
        errors.append(f"Invalid img_size: {config.img_size} (recommended: 64, 128, 256, 512, 1024)")
    
    # 학습률 확인
    if config.learning_rate <= 0 or config.learning_rate > 1:
        errors.append(f"Invalid learning_rate: {config.learning_rate} (must be 0-1)")
    
    # 에포크 수 확인
    if config.num_epochs <= 0:
        errors.append(f"Invalid num_epochs: {config.num_epochs} (must be > 0)")
    
    # 유효 비율 확인
    if config.valid_ratio < 0 or config.valid_ratio >= 1:
        errors.append(f"Invalid valid_ratio: {config.valid_ratio} (must be 0-1)")
    
    # 잠재 차원 확인
    if config.latent_dim <= 0:
        errors.append(f"Invalid latent_dim: {config.latent_dim} (must be > 0)")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True


def create_default_configs():
    """
    기본 설정들을 생성합니다.
    
    Returns:
        dict: 다양한 기본 설정들
    """
    configs = {
        'baseline': Config(),
        
        'quick_test': Config(
            num_epochs=5,
            batch_size=16,
            save_models=False
        ),
        
        'high_res': Config(
            img_size=512,
            batch_size=16,
            learning_rate=5e-4
        ),
        
        'small_model': Config(
            latent_dim=256,
            batch_size=64,
            learning_rate=2e-3
        ),
        
        'large_model': Config(
            latent_dim=1024,
            batch_size=16,
            learning_rate=5e-4,
            num_epochs=100
        )
    }
    
    return configs


# ========================
# 사용 예시 및 테스트
# ========================

def test_config_module():
    """config 모듈 테스트 함수"""
    print("=" * 60)
    print("🧪 CONFIG MODULE TEST")
    print("=" * 60)
    
    # 1. 기본 설정 생성
    print("\n1. Creating default config...")
    config = Config(category='grid', batch_size=16)
    print_config(config, "summary")
    
    # 2. 설정 검증
    print("\n2. Validating config...")
    validate_config(config)
    
    # 3. 설정 저장
    print("\n3. Saving config...")
    save_config(config, "test_config.json")
    save_config(config, "test_config.txt", format="txt")
    
    # 4. 설정 로드
    print("\n4. Loading config...")
    loaded_config = load_config("test_config.json")
    
    # 5. 설정 업데이트
    print("\n5. Updating config...")
    updated_config = update_config(config, learning_rate=5e-4, num_epochs=30)
    print(f"Original LR: {config.learning_rate}")
    print(f"Updated LR: {updated_config.learning_rate}")
    
    # 6. 설정 비교
    print("\n6. Comparing configs...")
    print_config_comparison([config, updated_config], ["Original", "Updated"])
    
    # 7. 기본 설정들 생성
    print("\n7. Creating default configs...")
    default_configs = create_default_configs()
    for name, cfg in default_configs.items():
        print(f"  {name}: {cfg.experiment_name}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    # 모듈 테스트 실행
    test_config_module()