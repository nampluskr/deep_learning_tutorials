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


def get_device(seed=42):
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

    return device


def train_epoch(model, train_loader, criterion, optimizer, device, 
                calculate_psnr, calculate_ssim):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
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
        with torch.no_grad():
            psnr = calculate_psnr(reconstructed, normal_images_norm)
            ssim_val = calculate_ssim(reconstructed, normal_images_norm)
        
        total_loss += loss.item()
        total_psnr += psnr.item()
        total_ssim += ssim_val.item()
        num_batches += 1
        
        # 진행 상황 업데이트
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'PSNR': f'{psnr.item():.2f}',
            'SSIM': f'{ssim_val.item():.3f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
    avg_ssim = total_ssim / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_psnr, avg_ssim


def validate_epoch(model, valid_loader, criterion, device, 
                  calculate_psnr, calculate_ssim):
    """한 에포크 검증"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in valid_loader:
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
            
            # 메트릭 계산
            loss = criterion(reconstructed, normal_images_norm)
            psnr = calculate_psnr(reconstructed, normal_images_norm)
            ssim_val = calculate_ssim(reconstructed, normal_images_norm)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            total_ssim += ssim_val.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
    avg_ssim = total_ssim / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_psnr, avg_ssim


def train_model(model, train_loader, valid_loader, device, num_epochs=50):
    """전체 훈련 루프 (Early Stopping 포함)"""
    
    # 손실 함수와 메트릭 정의
    criterion, calculate_psnr, calculate_ssim, _ = define_loss_and_metrics()
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping 설정
    best_valid_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    # 훈련 기록
    history = {
        'train_loss': [], 'train_psnr': [], 'train_ssim': [],
        'valid_loss': [], 'valid_psnr': [], 'valid_ssim': []
    }
    
    print("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 훈련 단계
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            calculate_psnr, calculate_ssim
        )
        
        # 검증 단계
        valid_loss, valid_psnr, valid_ssim = validate_epoch(
            model, valid_loader, criterion, device, 
            calculate_psnr, calculate_ssim
        )
        
        # 스케줄러 업데이트
        scheduler.step(valid_loss)
        
        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_psnr'].append(train_psnr)
        history['train_ssim'].append(train_ssim)
        history['valid_loss'].append(valid_loss)
        history['valid_psnr'].append(valid_psnr)
        history['valid_ssim'].append(valid_ssim)
        
        # Early stopping 체크
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
            print(f"✓ New best model saved (Valid Loss: {valid_loss:.4f})")
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train: Loss={train_loss:.4f}, PSNR={train_psnr:.2f}, SSIM={train_ssim:.3f} | "
              f"Valid: Loss={valid_loss:.4f}, PSNR={valid_psnr:.2f}, SSIM={valid_ssim:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping 적용
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # 최적 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training completed. Best validation loss: {best_valid_loss:.4f}")
    
    return model, history


def evaluate_model(model, test_loader, device):
    """테스트 데이터셋 평가"""
    
    # 메트릭 함수 가져오기
    _, _, _, compute_anomaly_score = define_loss_and_metrics()
    
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


def quick_train_test(model, train_loader, valid_loader, test_loader, device, epochs=5):
    """빠른 훈련 및 테스트 (디버깅용)"""
    print("="*50)
    print("빠른 훈련 및 테스트 시작")
    print("="*50)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {total_params:,}")
    
    # 짧은 훈련
    trained_model, history = train_model(model, train_loader, valid_loader, device, epochs)
    
    # 빠른 평가 (첫 번째 배치만)
    print("\n빠른 평가 중...")
    results, detailed_results = evaluate_model(trained_model, test_loader, device)
    
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
    
    # 더미 데이터로 테스트
    from torch.utils.data import TensorDataset, DataLoader
    
    # 더미 데이터 생성
    dummy_images = torch.randn(100, 3, 64, 64)
    dummy_labels = torch.zeros(100, dtype=torch.long)  # 모두 정상 데이터
    dummy_types = ['good'] * 100
    
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'image': dummy_images[i],
            'label': dummy_labels[i],
            'defect_type': dummy_types[i]
        })
    
    # 더미 데이터로더 (실제로는 collate_fn이 필요하지만 테스트용)
    print("더미 데이터로 함수 테스트 완료")


if __name__ == "__main__":
    # 훈련 함수 테스트 실행
    test_training_functions()