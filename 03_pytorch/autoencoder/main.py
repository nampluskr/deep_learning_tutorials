"""
범용 학습 스크립트
Config에 따라 다양한 모델을 학습할 수 있습니다.
"""

import os
import time
from datetime import datetime
import numpy as np
import torch
import json

# 모듈 imports
from config import Config, save_config, print_config, validate_config
from mvtec import get_transforms, get_dataloaders
from train import set_device, train_model, evaluate_model, create_metric_functions

# 모델 imports
from vanila_ae import VanillaAutoEncoder


def get_model(model_type, **model_params):
    """
    모델 타입에 따라 모델 인스턴스를 반환
    
    Args:
        model_type: 모델 타입 ('vanilla_ae', 'improved_ae', 'vae', etc.)
        **model_params: 모델 파라미터
    
    Returns:
        model: 초기화된 모델 인스턴스
    """
    if model_type == 'vanilla_ae':
        return VanillaAutoEncoder(latent_dim=model_params.get('latent_dim', 512))
    elif model_type == 'improved_ae':
        # from improved_ae import ImprovedAutoEncoder
        # return ImprovedAutoEncoder(latent_dim=model_params.get('latent_dim', 512))
        raise NotImplementedError("ImprovedAutoEncoder not implemented yet")
    elif model_type == 'vae':
        # from vae import VariationalAutoEncoder
        # return VariationalAutoEncoder(latent_dim=model_params.get('latent_dim', 512))
        raise NotImplementedError("VariationalAutoEncoder not implemented yet")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(config):
    """
    범용 모델 학습 메인 함수
    
    Args:
        config: Config 인스턴스 (model_type 필드 포함)
    """
    
    start_time = time.time()
    
    print("=" * 80)
    print(f"Model Training - {config.category}")
    print(f"Model Type: {getattr(config, 'model_type', 'vanilla_ae')}")
    print("=" * 80)
    
    # Config 검증 및 출력
    print_config(config, style="grouped")
    if not validate_config(config):
        print("Config validation failed!")
        return
    
    # 실험 디렉토리 설정
    model_type = getattr(config, 'model_type', 'vanilla_ae')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("./experiments", f"{model_type}_{config.category}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    
    # 디바이스 설정 및 시드 고정
    set_device(seed=config.seed)
    
    try:
        # 1. 데이터 로더 생성
        print("\nCreating data loaders...")
        train_transform, test_transform = get_transforms(
            img_size=config.img_size, 
            normalize=config.normalize
        )
        
        train_loader, valid_loader, test_loader = get_dataloaders(
            data_dir=config.data_dir,
            category=config.category,
            batch_size=config.batch_size,
            valid_ratio=config.valid_ratio,
            train_transform=train_transform,
            test_transform=test_transform
        )
        
        print(f"Data loaders created:")
        print(f"   - Training batches: {len(train_loader)}")
        print(f"   - Validation batches: {len(valid_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
        
        # 2. 모델 초기화
        print(f"\nInitializing {model_type} model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 파라미터 준비
        model_params = {
            'latent_dim': config.latent_dim,
        }
        
        # 모델 생성
        model = get_model(model_type, **model_params).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        print(f"   - Model: {model.__class__.__name__}")
        print(f"   - Parameters: {total_params:,}")
        print(f"   - Size: {model_size_mb:.1f} MB")
        print(f"   - Latent dim: {config.latent_dim}")
        print(f"   - Device: {device}")
        
        # 3. 메트릭 함수 준비
        metrics = create_metric_functions()
        print(f"   - Metrics: {list(metrics.keys())}")
        
        # 4. 모델 학습
        print(f"\nTraining model for {config.num_epochs} epochs...")
        print(f"   - Learning rate: {config.learning_rate}")
        print(f"   - Weight decay: {config.weight_decay}")
        
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=config.num_epochs,
            metrics=metrics
        )
        
        print("Training completed!")
        
        # 5. 모델 평가
        print("\nEvaluating model on test set...")
        results, detailed_results = evaluate_model(trained_model, test_loader)
        
        # 결과에 모델 정보 추가
        results['Category'] = config.category
        results['Model_Type'] = model_type
        results['Model_Name'] = model.__class__.__name__
        results['Latent_Dim'] = config.latent_dim
        results['Img_Size'] = config.img_size
        results['Epochs'] = config.num_epochs
        
        # 결과 출력
        print(f"\nEvaluation Results for {config.category}:")
        print("-" * 50)
        key_metrics = ['AUROC', 'AUPR', 'F1-Score', 'Accuracy', 'Threshold']
        for key in key_metrics:
            if key in results:
                value = results[key]
                if isinstance(value, float):
                    print(f"   {key:<12}: {value:.4f}")
                else:
                    print(f"   {key:<12}: {value}")
        
        # 6. 모델 저장
        if config.save_models:
            print("\nSaving models...")
            
            # 모델 상태 저장
            model_path = os.path.join(experiment_dir, "models", f"{model_type}_best.pth")
            torch.save(trained_model.state_dict(), model_path)
            print(f"   - Model state: {model_path}")
            
            # 전체 모델 저장
            full_model_path = os.path.join(experiment_dir, "models", f"{model_type}_full.pth")
            torch.save(trained_model, full_model_path)
            print(f"   - Full model: {full_model_path}")
        
        # 7. 결과 저장
        if config.save_results:
            # Config 저장
            config_path = os.path.join(experiment_dir, "config.json")
            save_config(config, config_path, format="json")
            
            # 결과 저장
            results_path = os.path.join(experiment_dir, "results.json")
            # NumPy types을 Python types로 변환
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_results[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # 훈련 히스토리 저장
            history_path = os.path.join(experiment_dir, "training_history.json")
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, list):
                    history_serializable[key] = value
                elif isinstance(value, np.ndarray):
                    history_serializable[key] = value.tolist()
                else:
                    history_serializable[key] = str(value)
            
            with open(history_path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
            
            print(f"Results saved: {results_path}")
        
        # 성공 메시지
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed successfully!")
        print("-" * 60)
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Category: {config.category}")
        print(f"   AUROC: {results.get('AUROC', 0):.4f}")
        print(f"   F1-Score: {results.get('F1-Score', 0):.4f}")
        print(f"   Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Results: {experiment_dir}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        
        # 에러 로그 저장
        try:
            error_log_path = os.path.join(experiment_dir, "error.log")
            with open(error_log_path, 'w') as f:
                import traceback
                f.write(f"Error occurred at: {datetime.now()}\n")
                f.write(f"Model: {model_type}\n")
                f.write(f"Category: {config.category}\n\n")
                f.write(traceback.format_exc())
            print(f"Error log saved: {error_log_path}")
        except:
            pass
        
        import traceback
        traceback.print_exc()
    
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Config 정의 - 다양한 모델 타입으로 실험
    config_list = [
        # Vanilla AutoEncoder 실험들
        Config(
            data_dir='/mnt/d/datasets/mvtec',
            category='bottle',
            num_epochs=50,
            batch_size=32,
            learning_rate=1e-3,
            latent_dim=512,
            img_size=256,
            normalize=False,
            save_models=True,
            save_results=True,
            experiment_name='vanilla_ae_baseline',
            model_type='vanilla_ae'  # 모델 타입 추가
        ),
        
        # 고해상도 실험
        Config(
            data_dir='/mnt/d/datasets/mvtec',
            category='bottle',
            num_epochs=30,
            batch_size=16,
            learning_rate=5e-4,
            latent_dim=512,
            img_size=512,
            normalize=False,
            save_models=True,
            save_results=True,
            experiment_name='vanilla_ae_high_res',
            model_type='vanilla_ae'
        ),
        
        # 큰 모델 실험
        Config(
            data_dir='/mnt/d/datasets/mvtec',
            category='bottle',
            num_epochs=80,
            batch_size=16,
            learning_rate=5e-4,
            latent_dim=1024,
            img_size=256,
            normalize=False,
            save_models=True,
            save_results=True,
            experiment_name='vanilla_ae_large',
            model_type='vanilla_ae'
        ),
        
        # 빠른 테스트
        Config(
            data_dir='/mnt/d/datasets/mvtec',
            category='bottle',
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-3,
            latent_dim=256,
            img_size=256,
            normalize=False,
            save_models=False,
            save_results=False,
            experiment_name='quick_test',
            model_type='vanilla_ae'
        ),
    ]
    
    print(f"Starting model training experiments with {len(config_list)} configurations")
    print("=" * 80)
    
    # 각 config에 대해 학습 실행
    for i, config in enumerate(config_list):
        print(f"\nRunning experiment {i+1}/{len(config_list)}: {config.experiment_name}")
        main(config)
        
        # 실험 간 간격
        if i < len(config_list) - 1:
            print(f"\nWaiting before next experiment...")
            time.sleep(2)
    
    print(f"\nAll experiments completed!")
    print("=" * 80)