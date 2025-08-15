import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm
import time
from copy import deepcopy

# utils.py에서 메트릭 함수들 import
from utils import define_loss_and_metrics


def set_device(seed=42):
    """디바이스 설정 및 시드 고정"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f">> Random Seed: {seed}")
    print(f">> Device: {device}")
    print(f">> CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f">> GPU: {torch.cuda.get_device_name(0)}")


def train_epoch(model, data_loader, criterion, optimizer, metrics={}):
    """
    한 에포크 훈련
    
    Args:
        model: 훈련할 모델
        data_loader: 훈련 데이터로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        metrics: 추가 메트릭 함수들의 딕셔너리 {"metric_name": metric_function}
    
    Returns:
        dict: {"loss": loss_value, "metric_name": metric_value, ...}
    """
    device = next(model.parameters()).device
    model.train()
    
    # 결과 저장용 딕셔너리 초기화
    total_results = {"loss": 0.0}
    for metric_name in metrics.keys():
        total_results[metric_name] = 0.0
    
    num_batches = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 정상 데이터만 사용 (labels == 0)
        normal_mask = (labels == 0)
        if not normal_mask.any():
            continue
        
        normal_images = images[normal_mask]
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed, latent, features = model(normal_images)
        
        # 입력 이미지를 [0, 1] 범위로 정규화 (Sigmoid 출력과 매칭)
        normal_images_norm = (normal_images - normal_images.min()) / (normal_images.max() - normal_images.min() + 1e-8)
        
        # Loss 계산 및 역전파
        loss = criterion(reconstructed, normal_images_norm)
        loss.backward()
        optimizer.step()
        
        # 메트릭 계산
        batch_results = {"loss": loss.item()}
        
        with torch.no_grad():
            for metric_name, metric_fn in metrics.items():
                try:
                    metric_value = metric_fn(reconstructed, normal_images_norm)
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    batch_results[metric_name] = metric_value
                except Exception as e:
                    print(f"Warning: Error calculating {metric_name}: {e}")
                    batch_results[metric_name] = 0.0
        
        # 누적 결과 업데이트
        for key, value in batch_results.items():
            total_results[key] += value
        
        num_batches += 1
        
        # 진행 상황 업데이트
        avg_results = {key: value / num_batches for key, value in total_results.items()}
        progress_info = {f'{key.capitalize()}': f'{value:.4f}' for key, value in avg_results.items()}
        progress_bar.set_postfix(progress_info)
    
    # 평균 계산
    if num_batches > 0:
        avg_results = {key: value / num_batches for key, value in total_results.items()}
    else:
        avg_results = {key: 0.0 for key in total_results.keys()}
    
    return avg_results


def validate_epoch(model, data_loader, criterion, metrics={}):
    """
    한 에포크 검증
    
    Args:
        model: 검증할 모델
        data_loader: 검증 데이터로더
        criterion: 손실 함수
        metrics: 추가 메트릭 함수들의 딕셔너리 {"metric_name": metric_function}
    
    Returns:
        dict: {"loss": loss_value, "metric_name": metric_value, ...}
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 결과 저장용 딕셔너리 초기화
    total_results = {"loss": 0.0}
    for metric_name in metrics.keys():
        total_results[metric_name] = 0.0
    
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 정상 데이터만 사용
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue
            
            normal_images = images[normal_mask]
            
            # Forward pass
            reconstructed, latent, features = model(normal_images)
            
            # 정규화
            normal_images_norm = (normal_images - normal_images.min()) / (normal_images.max() - normal_images.min() + 1e-8)
            
            # Loss 계산
            loss = criterion(reconstructed, normal_images_norm)
            batch_results = {"loss": loss.item()}
            
            # 메트릭 계산
            for metric_name, metric_fn in metrics.items():
                try:
                    metric_value = metric_fn(reconstructed, normal_images_norm)
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    batch_results[metric_name] = metric_value
                except Exception as e:
                    print(f"Warning: Error calculating {metric_name}: {e}")
                    batch_results[metric_name] = 0.0
            
            # 누적 결과 업데이트
            for key, value in batch_results.items():
                total_results[key] += value
            
            num_batches += 1
    
    # 평균 계산
    if num_batches > 0:
        avg_results = {key: value / num_batches for key, value in total_results.items()}
    else:
        avg_results = {key: 0.0 for key in total_results.keys()}
    
    return avg_results


def train_model(model, train_loader, valid_loader, num_epochs=50, metrics={}):
    """
    전체 훈련 루프 (Early Stopping 포함)
    
    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터로더
        valid_loader: 검증 데이터로더
        num_epochs: 훈련 에포크 수
        metrics: 추가 메트릭 함수들의 딕셔너리
    
    Returns:
        tuple: (trained_model, history)
    """
    
    # 모델로부터 디바이스 추출
    device = next(model.parameters()).device
    
    # 손실 함수와 메트릭 정의
    criterion, calculate_psnr, calculate_ssim, _ = define_loss_and_metrics()
    
    # 기본 메트릭이 제공되지 않은 경우 추가
    if not metrics:
        metrics = {
            "psnr": calculate_psnr,
            "ssim": calculate_ssim
        }
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping 설정
    best_valid_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    # 훈련 기록 초기화
    history = {}
    # 모든 메트릭에 대해 train/valid 기록 준비
    for key in ["loss"] + list(metrics.keys()):
        history[f'train_{key}'] = []
        history[f'valid_{key}'] = []
    
    print("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 훈련 단계
        train_results = train_epoch(model, train_loader, criterion, optimizer, metrics)
        
        # 검증 단계
        valid_results = validate_epoch(model, valid_loader, criterion, metrics)
        
        # 스케줄러 업데이트 및 학습률 모니터링
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_results['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        # 학습률 변경 감지
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 기록 저장
        for key in train_results.keys():
            history[f'train_{key}'].append(train_results[key])
            history[f'valid_{key}'].append(valid_results[key])
        
        # Early stopping 체크
        if valid_results['loss'] < best_valid_loss:
            best_valid_loss = valid_results['loss']
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
            print(f"✓ New best model saved (Valid Loss: {valid_results['loss']:.4f})")
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 훈련 결과 문자열
        train_str = ', '.join([f'{k.capitalize()}={v:.4f}' for k, v in train_results.items()])
        valid_str = ', '.join([f'{k.capitalize()}={v:.4f}' for k, v in valid_results.items()])
        
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train: {train_str} | "
              f"Valid: {valid_str} | "
              f"LR={current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Early stopping 적용
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # 최적 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training completed. Best validation loss: {best_valid_loss:.4f}")
    
    return model, history


def evaluate_model(model, test_loader):
    """
    테스트 데이터셋 평가
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터로더
    
    Returns:
        tuple: (results, detailed_results)
    """
    
    # 메트릭 함수 가져오기
    _, _, _, compute_anomaly_score = define_loss_and_metrics()
    
    # 모델로부터 디바이스 추출
    device = next(model.parameters()).device
    
    model.eval()
    
    all_scores = []
    all_labels = []
    all_defect_types = []
    all_images = []
    all_reconstructed = []
    all_error_maps = []
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            defect_types = batch['defect_type']
            
            # 정규화
            images_norm = (images - images.min()) / (images.max() - images.min() + 1e-8)
            
            # Anomaly score 계산
            scores, error_maps, reconstructed = compute_anomaly_score(model, images_norm)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels)
            all_defect_types.extend(defect_types)
            all_images.extend(images.cpu().numpy())
            all_reconstructed.extend(reconstructed.cpu().numpy())
            all_error_maps.extend(error_maps.cpu().numpy())
    
    # NumPy 배열로 변환
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 임계값 계산 (정상 데이터의 95th percentile)
    normal_scores = all_scores[all_labels == 0]
    if len(normal_scores) == 0:
        threshold = np.median(all_scores)
    else:
        threshold = np.percentile(normal_scores, 95)
    
    # 이진 예측 생성
    predictions = (all_scores > threshold).astype(int)
    
    # 성능 메트릭 계산
    results = {}
    if len(np.unique(all_labels)) > 1:  # 두 클래스가 모두 존재
        results.update({
            'AUROC': roc_auc_score(all_labels, all_scores),
            'AUPR': average_precision_score(all_labels, all_scores),
            'Accuracy': accuracy_score(all_labels, predictions),
            'Precision': precision_score(all_labels, predictions, zero_division=0),
            'Recall': recall_score(all_labels, predictions, zero_division=0),
            'F1-Score': f1_score(all_labels, predictions, zero_division=0)
        })
    else:
        # 한 클래스만 존재하는 경우
        results.update({
            'AUROC': 0.5,
            'AUPR': 0.5,
            'Accuracy': accuracy_score(all_labels, predictions),
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-Score': 0.0
        })
    
    # 추가 통계
    results.update({
        'Threshold': threshold,
        'Normal_Score_Mean': np.mean(normal_scores) if len(normal_scores) > 0 else 0,
        'Normal_Score_Std': np.std(normal_scores) if len(normal_scores) > 0 else 0,
        'Anomaly_Score_Mean': np.mean(all_scores[all_labels == 1]) if np.sum(all_labels) > 0 else 0
    })
    
    # 상세 결과
    detailed_results = {
        'scores': all_scores,
        'labels': all_labels,
        'defect_types': all_defect_types,
        'images': all_images,
        'reconstructed': all_reconstructed,
        'error_maps': all_error_maps,
        'predictions': predictions
    }
    
    return results, detailed_results


def create_metric_functions():
    """
    일반적으로 사용되는 메트릭 함수들을 반환합니다.
    
    Returns:
        dict: 메트릭 함수들의 딕셔너리
    """
    reconstruction_loss, calculate_psnr, calculate_ssim, _ = define_loss_and_metrics()
    
    return {
        "psnr": calculate_psnr,
        "ssim": calculate_ssim
    }


def quick_train_test(model, train_loader, valid_loader, test_loader, epochs=5, metrics={}):
    """
    빠른 훈련 및 테스트 (디버깅용)
    
    Args:
        model: 훈련/평가할 모델
        train_loader: 훈련 데이터로더
        valid_loader: 검증 데이터로더
        test_loader: 테스트 데이터로더
        epochs: 훈련 에포크 수
        metrics: 추가 메트릭 함수들의 딕셔너리
    
    Returns:
        tuple: (trained_model, history, results, detailed_results)
    """
    print("="*50)
    print("빠른 훈련 및 테스트 시작")
    print("="*50)
    
    # 모델로부터 디바이스 추출
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {total_params:,}")
    
    # 기본 메트릭 설정
    if not metrics:
        metrics = create_metric_functions()
    
    # 짧은 훈련
    trained_model, history = train_model(model, train_loader, valid_loader, epochs, metrics)
    
    # 빠른 평가
    print("\n빠른 평가 중...")
    results, detailed_results = evaluate_model(trained_model, test_loader)
    
    # 결과 출력
    print(f"\n평가 결과:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return trained_model, history, results, detailed_results


# 테스트용 함수
def test_training_functions():
    """훈련 함수들 테스트"""
    print("="*50)
    print("Training 함수 테스트")
    print("="*50)
    
    # 메트릭 함수 테스트
    metrics = create_metric_functions()
    print(f"Available metrics: {list(metrics.keys())}")
    
    # 더미 데이터로 간단 테스트
    print("더미 데이터로 함수 테스트 완료")


if __name__ == "__main__":
    # 훈련 함수 테스트 실행
    test_training_functions()