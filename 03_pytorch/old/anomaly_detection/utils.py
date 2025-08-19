"""
손실함수, 메트릭, 시각화 등 유틸리티 함수들
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def define_loss_and_metrics():
    """손실 함수와 평가 메트릭 정의"""
    
    def reconstruction_loss(pred, target):
        """재구성 손실: MSE + SSIM 조합"""
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        return 0.7 * mse_loss + 0.3 * ssim_loss
    
    def calculate_psnr(pred, target):
        """PSNR 계산"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 10 * torch.log10(1.0 / mse)
    
    def calculate_ssim(pred, target):
        """SSIM 계산"""
        return ssim(pred, target, data_range=1.0)
    
    def compute_anomaly_score(model, images):
        """Anomaly Score 계산 (픽셀별 재구성 오차)"""
        model.eval()
        with torch.no_grad():
            reconstructed, _, _ = model(images)
            
            # 픽셀별 L2 거리
            pixel_errors = (images - reconstructed) ** 2
            
            # 이미지별 평균 오차 (anomaly score)
            anomaly_scores = torch.mean(pixel_errors, dim=[1, 2, 3])
            
            # 픽셀별 오차 맵 (시각화용)
            error_maps = torch.mean(pixel_errors, dim=1, keepdim=True)
            
        return anomaly_scores, error_maps, reconstructed
    
    return reconstruction_loss, calculate_psnr, calculate_ssim, compute_anomaly_score


def denormalize_image(tensor_img):
    """ImageNet 정규화 해제"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 텐서를 numpy로 변환하고 채널 순서 변경
    if isinstance(tensor_img, torch.Tensor):
        img = tensor_img.cpu().numpy()
    else:
        img = tensor_img
    
    if img.ndim == 3 and img.shape[0] == 3:  # [C, H, W]
        img = img.transpose(1, 2, 0)  # [H, W, C]
    
    # 정규화 해제
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img


def visualize_training_history(history, category):
    """훈련 히스토리 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Training History - {category}', fontsize=16)
    
    # Loss 그래프
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['valid_loss'], label='Valid Loss', color='red')
    axes[0].set_title('Reconstruction Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # PSNR 그래프
    axes[1].plot(history['train_psnr'], label='Train PSNR', color='blue')
    axes[1].plot(history['valid_psnr'], label='Valid PSNR', color='red')
    axes[1].set_title('PSNR')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].legend()
    axes[1].grid(True)
    
    # SSIM 그래프
    axes[2].plot(history['train_ssim'], label='Train SSIM', color='blue')
    axes[2].plot(history['valid_ssim'], label='Valid SSIM', color='red')
    axes[2].set_title('SSIM')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_reconstruction_examples(detailed_results, category, num_examples=8):
    """재구성 결과 시각화"""
    images = detailed_results['images']
    reconstructed = detailed_results['reconstructed']
    error_maps = detailed_results['error_maps']
    labels = detailed_results['labels']
    defect_types = detailed_results['defect_types']
    scores = detailed_results['scores']
    
    # 정상과 이상 샘플 각각에서 선택
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    # 각각에서 절반씩 선택
    if len(normal_indices) > 0:
        selected_normal = np.random.choice(normal_indices, 
                                         min(num_examples//2, len(normal_indices)), 
                                         replace=False)
    else:
        selected_normal = []
    
    if len(anomaly_indices) > 0:
        selected_anomaly = np.random.choice(anomaly_indices, 
                                          min(num_examples//2, len(anomaly_indices)), 
                                          replace=False)
    else:
        selected_anomaly = []
    
    selected_indices = np.concatenate([selected_normal, selected_anomaly])
    
    if len(selected_indices) == 0:
        print("No samples to visualize")
        return
    
    fig, axes = plt.subplots(4, len(selected_indices), figsize=(3*len(selected_indices), 12))
    fig.suptitle(f'Reconstruction Examples - {category}', fontsize=16)
    
    # 단일 샘플인 경우 축 차원 조정
    if len(selected_indices) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(selected_indices):
        # 원본 이미지
        orig_img = denormalize_image(images[idx])
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original\n{defect_types[idx]}')
        axes[0, i].axis('off')
        
        # 재구성 이미지
        recon_img = reconstructed[idx].transpose(1, 2, 0)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
        
        # 오차 맵
        error_map = error_maps[idx][0]
        im = axes[2, i].imshow(error_map, cmap='hot')
        axes[2, i].set_title('Error Map')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
        
        # 점수 표시
        score = scores[idx]
        label = "Normal" if labels[idx] == 0 else "Anomaly"
        axes[3, i].text(0.5, 0.5, f'{label}\nScore: {score:.4f}', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[3, i].set_xlim(0, 1)
        axes[3, i].set_ylim(0, 1)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_score_distribution(detailed_results, category):
    """점수 분포 및 ROC 곡선 시각화"""
    scores = detailed_results['scores']
    labels = detailed_results['labels']
    
    if len(scores[labels == 0]) == 0:
        print("No normal samples for threshold calculation")
        return
        
    threshold = np.percentile(scores[labels == 0], 95)
    
    plt.figure(figsize=(12, 5))
    
    # 점수 분포 히스토그램
    plt.subplot(1, 2, 1)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    if len(anomaly_scores) > 0:
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Score Distribution - {category}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ROC 곡선
    plt.subplot(1, 2, 2)
    if len(np.unique(labels)) > 1:  # 두 클래스가 모두 존재하는 경우만
        from sklearn.metrics import roc_auc_score
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {category}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Only one class present\nCannot plot ROC curve', 
                ha='center', va='center', fontsize=12)
        plt.title(f'ROC Curve - {category}')
    
    plt.tight_layout()
    plt.show()


def create_results_summary_table(all_category_results):
    """전체 결과 요약 테이블 생성"""
    df = pd.DataFrame(all_category_results)
    df = df.round(4)
    
    # 평균 성능 추가
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mean_row = df[numeric_cols].mean()
    mean_row['Category'] = 'AVERAGE'
    df = pd.concat([df, mean_row.to_frame().T], ignore_index=True)
    
    # 스타일링된 테이블 출력
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def plot_category_comparison(all_category_results):
    """카테고리별 성능 비교 시각화"""
    df = pd.DataFrame(all_category_results)
    
    # AVERAGE 행 제거 (시각화용)
    df_plot = df[df['Category'] != 'AVERAGE'].copy()
    
    if len(df_plot) == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MVTec Anomaly Detection Performance by Category', fontsize=16)
    
    # AUROC 비교
    axes[0, 0].bar(df_plot['Category'], df_plot['AUROC'], color='skyblue')
    axes[0, 0].set_title('AUROC by Category')
    axes[0, 0].set_ylabel('AUROC')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUPR 비교
    axes[0, 1].bar(df_plot['Category'], df_plot['AUPR'], color='lightcoral')
    axes[0, 1].set_title('AUPR by Category') 
    axes[0, 1].set_ylabel('AUPR')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-Score 비교
    axes[1, 0].bar(df_plot['Category'], df_plot['F1-Score'], color='lightgreen')
    axes[1, 0].set_title('F1-Score by Category')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 종합 성능 산점도
    axes[1, 1].scatter(df_plot['AUROC'], df_plot['F1-Score'], 
                      c=df_plot['AUPR'], cmap='viridis', s=100)
    axes[1, 1].set_xlabel('AUROC')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('AUROC vs F1-Score (colored by AUPR)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 컬러바 추가
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('AUPR')
    
    # 카테고리 라벨 추가
    for i, category in enumerate(df_plot['Category']):
        axes[1, 1].annotate(category, (df_plot.iloc[i]['AUROC'], df_plot.iloc[i]['F1-Score']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()


# 테스트용 함수
def test_metrics():
    """메트릭 함수들 테스트"""
    print("="*50)
    print("Utils 메트릭 함수 테스트")
    print("="*50)
    
    # 더미 데이터 생성
    pred = torch.rand(4, 3, 64, 64)
    target = torch.rand(4, 3, 64, 64)
    
    # 메트릭 함수들 가져오기
    reconstruction_loss, calculate_psnr, calculate_ssim, _ = define_loss_and_metrics()
    
    # 테스트
    loss = reconstruction_loss(pred, target)
    psnr = calculate_psnr(pred, target)
    ssim_val = calculate_ssim(pred, target)
    
    print(f"Reconstruction Loss: {loss.item():.4f}")
    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim_val.item():.4f}")
    
    # 이미지 정규화 해제 테스트
    normalized_img = torch.randn(3, 64, 64)
    denorm_img = denormalize_image(normalized_img)
    print(f"Denormalized image shape: {denorm_img.shape}")
    print(f"Denormalized image range: [{denorm_img.min():.3f}, {denorm_img.max():.3f}]")


if __name__ == "__main__":
    # 메트릭 테스트 실행
    test_metrics()