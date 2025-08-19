"""
Vision Anomaly Detection Metrics and Loss Functions
OLED 화질 이상 검출을 위한 다양한 손실 함수와 평가 메트릭
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
)
from pytorch_msssim import ssim, ms_ssim
import lpips


# =============================================================================
# Loss Functions for Anomaly Detection
# =============================================================================

class ReconstructionLoss(nn.Module):
    """기본 재구성 손실 함수들"""

    def __init__(self, loss_type='mse', reduction='mean'):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction

    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'mae' or self.loss_type == 'l1':
            return F.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(pred, target, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class SSIMLoss(nn.Module):
    """SSIM 기반 손실 함수"""

    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel

    def forward(self, pred, target):
        # SSIM은 높을수록 좋으므로 1에서 빼서 loss로 변환
        ssim_value = ssim(pred, target,
                         data_range=self.data_range,
                         size_average=self.size_average)
        return 1 - ssim_value


class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM 손실 함수"""

    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel

    def forward(self, pred, target):
        ms_ssim_value = ms_ssim(pred, target,
                               data_range=self.data_range,
                               size_average=self.size_average)
        return 1 - ms_ssim_value


class PerceptualLoss(nn.Module):
    """VGG 기반 Perceptual Loss"""

    def __init__(self, feature_layers=[2, 7, 12, 21], use_gpu=True):
        super().__init__()

        # VGG16 feature extractor
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.ModuleList()

        layers = []
        for i, layer in enumerate(vgg):
            layers.append(layer)
            if i in feature_layers:
                self.feature_extractor.append(nn.Sequential(*layers))
                layers = []

        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # VGG expects input in [0, 1] range
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = 0
        for extractor in self.feature_extractor:
            pred_feat = extractor(pred_norm)
            target_feat = extractor(target_norm)
            loss += self.mse_loss(pred_feat, target_feat)

        return loss


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity Loss"""

    def __init__(self, net='alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)

        # Freeze LPIPS parameters
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # LPIPS expects input in [-1, 1] range
        pred_norm = pred * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0

        return self.lpips(pred_norm, target_norm).mean()


class CombinedLoss(nn.Module):
    """여러 손실 함수의 조합"""

    def __init__(self, loss_weights=None):
        super().__init__()

        if loss_weights is None:
            loss_weights = {
                'mse': 1.0,
                'ssim': 0.5,
                'perceptual': 0.1
            }

        self.loss_weights = loss_weights
        self.losses = nn.ModuleDict()

        if 'mse' in loss_weights:
            self.losses['mse'] = ReconstructionLoss('mse')
        if 'mae' in loss_weights:
            self.losses['mae'] = ReconstructionLoss('mae')
        if 'ssim' in loss_weights:
            self.losses['ssim'] = SSIMLoss()
        if 'ms_ssim' in loss_weights:
            self.losses['ms_ssim'] = MS_SSIMLoss()
        if 'perceptual' in loss_weights:
            self.losses['perceptual'] = PerceptualLoss()
        if 'lpips' in loss_weights:
            self.losses['lpips'] = LPIPSLoss()

    def forward(self, pred, target):
        total_loss = 0
        loss_dict = {}

        for loss_name, weight in self.loss_weights.items():
            if loss_name in self.losses:
                loss_value = self.losses[loss_name](pred, target)
                weighted_loss = weight * loss_value
                total_loss += weighted_loss
                loss_dict[f'{loss_name}_loss'] = loss_value.item()

        return total_loss, loss_dict


class VAELoss(nn.Module):
    """VAE를 위한 손실 함수 (Reconstruction + KL Divergence)"""

    def __init__(self, beta=1.0, recon_loss_type='mse'):
        super().__init__()
        self.beta = beta
        self.recon_loss = ReconstructionLoss(recon_loss_type)

    def forward(self, pred, target, mu, logvar):
        # Reconstruction loss
        recon_loss = self.recon_loss(pred, target)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= target.size(0)  # Average over batch

        # Total VAE loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced anomaly detection"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Anomaly Score Computation Functions
# =============================================================================

def compute_reconstruction_error(pred, target, method='mse', pixel_wise=False):
    """재구성 오차 계산"""
    if method == 'mse':
        error = (pred - target) ** 2
    elif method == 'mae':
        error = torch.abs(pred - target)
    elif method == 'huber':
        error = F.smooth_l1_loss(pred, target, reduction='none')
    else:
        raise ValueError(f"Unknown method: {method}")

    if pixel_wise:
        return error  # (B, C, H, W)
    else:
        return error.view(error.size(0), -1).mean(dim=1)  # (B,)


def compute_ssim_error(pred, target, data_range=1.0):
    """SSIM 기반 오차 계산"""
    batch_size = pred.size(0)
    ssim_errors = []

    for i in range(batch_size):
        ssim_val = ssim(pred[i:i+1], target[i:i+1], data_range=data_range, size_average=True)
        ssim_errors.append(1 - ssim_val)

    return torch.stack(ssim_errors)


def compute_perceptual_error(pred, target, feature_extractor=None):
    """Perceptual 오차 계산"""
    if feature_extractor is None:
        # Simple VGG-based perceptual error
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:12]
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        feature_extractor = vgg

    # Normalize for VGG
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

    pred_norm = (pred - mean) / std
    target_norm = (target - mean) / std

    pred_feat = feature_extractor(pred_norm)
    target_feat = feature_extractor(target_norm)

    error = F.mse_loss(pred_feat, target_feat, reduction='none')
    return error.view(error.size(0), -1).mean(dim=1)


def compute_combined_anomaly_score(pred, target, weights=None):
    """여러 방법을 조합한 anomaly score 계산"""
    if weights is None:
        weights = {'mse': 0.5, 'ssim': 0.3, 'perceptual': 0.2}

    scores = {}

    if 'mse' in weights:
        scores['mse'] = compute_reconstruction_error(pred, target, 'mse')
    if 'mae' in weights:
        scores['mae'] = compute_reconstruction_error(pred, target, 'mae')
    if 'ssim' in weights:
        scores['ssim'] = compute_ssim_error(pred, target)
    if 'perceptual' in weights:
        scores['perceptual'] = compute_perceptual_error(pred, target)

    # Normalize scores to [0, 1] range
    normalized_scores = {}
    for key, score in scores.items():
        score_min = score.min()
        score_max = score.max()
        if score_max > score_min:
            normalized_scores[key] = (score - score_min) / (score_max - score_min)
        else:
            normalized_scores[key] = torch.zeros_like(score)

    # Weighted combination
    final_score = torch.zeros_like(list(normalized_scores.values())[0])
    for key, weight in weights.items():
        if key in normalized_scores:
            final_score += weight * normalized_scores[key]

    return final_score, normalized_scores


# =============================================================================
# Evaluation Metrics for Anomaly Detection
# =============================================================================

def compute_auroc(y_true, y_scores):
    """AUROC (Area Under ROC Curve) 계산"""
    if len(np.unique(y_true)) < 2:
        return 0.5  # Only one class present
    return roc_auc_score(y_true, y_scores)


def compute_aupr(y_true, y_scores):
    """AUPR (Area Under Precision-Recall Curve) 계산"""
    if len(np.unique(y_true)) < 2:
        return np.mean(y_true)  # Baseline performance
    return average_precision_score(y_true, y_scores)


def compute_classification_metrics(y_true, y_pred):
    """분류 메트릭들 계산"""
    metrics = {}

    if len(np.unique(y_true)) < 2:
        # Only one class present
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0
    else:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def find_optimal_threshold(y_true, y_scores, method='f1'):
    """최적 임계값 찾기"""
    if len(np.unique(y_true)) < 2:
        return np.median(y_scores)

    if method == 'f1':
        # F1-score를 최대화하는 임계값
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
        best_f1 = 0
        best_threshold = np.median(y_scores)

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    elif method == 'youden':
        # Youden's J statistic을 최대화
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        return thresholds[optimal_idx]

    elif method == 'percentile':
        # 정상 데이터의 95th percentile
        normal_scores = y_scores[y_true == 0]
        if len(normal_scores) > 0:
            return np.percentile(normal_scores, 95)
        else:
            return np.median(y_scores)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_psnr(pred, target, data_range=1.0):
    """PSNR (Peak Signal-to-Noise Ratio) 계산"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(data_range ** 2 / mse)


def compute_ssim_metric(pred, target, data_range=1.0):
    """SSIM 메트릭 계산"""
    return ssim(pred, target, data_range=data_range, size_average=True)


# =============================================================================
# OLED-specific Metrics
# =============================================================================

def compute_uniformity_score(image, mask=None):
    """OLED 화면 균일도 점수 계산"""
    if mask is not None:
        image = image * mask

    # 각 채널별 표준편차 계산
    std_per_channel = torch.std(image.view(image.size(0), image.size(1), -1), dim=2)

    # 전체 균일도 점수 (낮을수록 균일함)
    uniformity_score = torch.mean(std_per_channel, dim=1)

    return uniformity_score


def compute_contrast_score(image):
    """대비 점수 계산"""
    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    sobel_x = sobel_x.view(1, 1, 3, 3).to(image.device)
    sobel_y = sobel_y.view(1, 1, 3, 3).to(image.device)

    # Convert to grayscale
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    # Apply Sobel filters
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)

    # Compute edge magnitude
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    # Average edge magnitude as contrast score
    contrast_score = torch.mean(edge_magnitude.view(edge_magnitude.size(0), -1), dim=1)

    return contrast_score


def compute_mura_detection_score(pred, target, sensitivity=1.0):
    """무라(Mura) 검출 점수 계산 (OLED 특화)"""
    # 재구성 오차 계산
    recon_error = compute_reconstruction_error(pred, target, 'mse', pixel_wise=True)

    # 공간적 평활화를 통한 노이즈 제거
    kernel_size = 5
    blur_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
    blur_kernel = blur_kernel.to(recon_error.device)

    # 각 채널별로 블러 적용
    smoothed_error = torch.zeros_like(recon_error)
    for c in range(recon_error.size(1)):
        smoothed_error[:, c:c+1] = F.conv2d(
            recon_error[:, c:c+1], blur_kernel, padding=kernel_size//2
        )

    # 임계값 기반 무라 영역 검출
    threshold = sensitivity * torch.mean(smoothed_error)
    mura_mask = (smoothed_error > threshold).float()

    # 무라 검출 점수 (비율)
    mura_score = torch.mean(mura_mask.view(mura_mask.size(0), -1), dim=1)

    return mura_score, mura_mask


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_scores(scores, method='min_max'):
    """점수 정규화"""
    if method == 'min_max':
        min_val = scores.min()
        max_val = scores.max()
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        else:
            return torch.zeros_like(scores)

    elif method == 'z_score':
        mean_val = scores.mean()
        std_val = scores.std()
        if std_val > 0:
            return (scores - mean_val) / std_val
        else:
            return torch.zeros_like(scores)

    elif method == 'sigmoid':
        return torch.sigmoid(scores)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_comprehensive_metrics(y_true, y_scores, threshold=None):
    """종합적인 메트릭 계산"""
    results = {}

    # Convert to numpy if torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.cpu().numpy()

    # AUROC and AUPR
    results['auroc'] = compute_auroc(y_true, y_scores)
    results['aupr'] = compute_aupr(y_true, y_scores)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_scores, method='f1')

    results['threshold'] = threshold

    # Classification metrics
    y_pred = (y_scores >= threshold).astype(int)
    classification_metrics = compute_classification_metrics(y_true, y_pred)
    results.update(classification_metrics)

    # Additional statistics
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]

    if len(normal_scores) > 0:
        results['normal_score_mean'] = np.mean(normal_scores)
        results['normal_score_std'] = np.std(normal_scores)

    if len(anomaly_scores) > 0:
        results['anomaly_score_mean'] = np.mean(anomaly_scores)
        results['anomaly_score_std'] = np.std(anomaly_scores)

    return results


# =============================================================================
# Loss and Metric Factory Functions
# =============================================================================

def get_loss_function(loss_name, **kwargs):
    """손실 함수 팩토리"""
    loss_functions = {
        'mse': lambda: ReconstructionLoss('mse'),
        'mae': lambda: ReconstructionLoss('mae'),
        'huber': lambda: ReconstructionLoss('huber'),
        'ssim': lambda: SSIMLoss(**kwargs),
        'ms_ssim': lambda: MS_SSIMLoss(**kwargs),
        'perceptual': lambda: PerceptualLoss(**kwargs),
        'lpips': lambda: LPIPSLoss(**kwargs),
        'combined': lambda: CombinedLoss(**kwargs),
        'vae': lambda: VAELoss(**kwargs),
        'focal': lambda: FocalLoss(**kwargs),
    }

    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")

    return loss_functions[loss_name]()


def get_anomaly_score_function(method_name):
    """Anomaly score 계산 함수 팩토리"""
    score_functions = {
        'mse': lambda p, t: compute_reconstruction_error(p, t, 'mse'),
        'mae': lambda p, t: compute_reconstruction_error(p, t, 'mae'),
        'ssim': lambda p, t: compute_ssim_error(p, t),
        'perceptual': lambda p, t: compute_perceptual_error(p, t),
        'combined': lambda p, t: compute_combined_anomaly_score(p, t)[0],
    }

    if method_name not in score_functions:
        raise ValueError(f"Unknown score function: {method_name}")

    return score_functions[method_name]


# =============================================================================
# Test Functions
# =============================================================================

def test_metrics():
    """메트릭 함수들 테스트"""
    print("=" * 60)
    print("Testing Metrics and Loss Functions")
    print("=" * 60)

    # 테스트 데이터 생성
    batch_size = 4
    channels = 3
    height, width = 256, 256

    # 가상 데이터
    pred = torch.randn(batch_size, channels, height, width).clamp(0, 1)
    target = torch.randn(batch_size, channels, height, width).clamp(0, 1)

    # 가상 분류 데이터
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])

    print(f"Test data shape: {pred.shape}")

    # Loss functions 테스트
    print("\nTesting Loss Functions:")

    loss_tests = [
        ('MSE', ReconstructionLoss('mse')),
        ('MAE', ReconstructionLoss('mae')),
        ('SSIM', SSIMLoss()),
        ('Perceptual', PerceptualLoss()),
    ]

    for name, loss_fn in loss_tests:
        try:
            if name == 'Perceptual':
                # Perceptual loss는 시간이 오래 걸리므로 작은 이미지로 테스트
                small_pred = pred[:, :, :64, :64]
                small_target = target[:, :, :64, :64]
                loss_value = loss_fn(small_pred, small_target)
            else:
                loss_value = loss_fn(pred, target)
            print(f"   {name} Loss: {loss_value.item():.4f}")
        except Exception as e:
            print(f"   {name} Loss: Error - {e}")

    # Anomaly score 계산 테스트
    print("\nTesting Anomaly Score Functions:")

    try:
        mse_scores = compute_reconstruction_error(pred, target, 'mse')
        ssim_scores = compute_ssim_error(pred, target)
        print(f"   MSE Scores: {mse_scores}")
        print(f"   SSIM Scores: {ssim_scores}")
    except Exception as e:
        print(f"   Anomaly Scores: Error - {e}")

    # 평가 메트릭 테스트
    print("\nTesting Evaluation Metrics:")

    try:
        metrics = compute_comprehensive_metrics(y_true, y_scores)
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
    except Exception as e:
        print(f"   Evaluation Metrics: Error - {e}")

    # OLED 특화 메트릭 테스트
    print("\nTesting OLED-specific Metrics:")

    try:
        uniformity = compute_uniformity_score(pred)
        contrast = compute_contrast_score(pred)
        mura_score, mura_mask = compute_mura_detection_score(pred, target)

        print(f"   Uniformity Scores: {uniformity}")
        print(f"   Contrast Scores: {contrast}")
        print(f"   Mura Detection Scores: {mura_score}")
        print(f"   Mura Mask Shape: {mura_mask.shape}")
    except Exception as e:
        print(f"   OLED Metrics: Error - {e}")

    print("\nAll tests completed!")


if __name__ == "__main__":

    test_metrics()