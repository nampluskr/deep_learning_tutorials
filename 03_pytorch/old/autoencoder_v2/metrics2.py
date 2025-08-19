"""
Vision Anomaly Detection Metrics and Loss Functions
이상 탐지를 위한 다양한 손실 함수와 평가 지표들을 포함합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
from pytorch_msssim import ssim, ms_ssim
import torchvision.models as models
from typing import Tuple, Dict, Optional, Union


# =============================================================================
# Reconstruction Loss Functions
# =============================================================================

class MSELoss(nn.Module):
    """Mean Squared Error Loss"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.mse(pred, target)


class L1Loss(nn.Module):
    """L1 (MAE) Loss"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.l1(pred, target)


class SSIMLoss(nn.Module):
    """SSIM Loss"""
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel
    
    def forward(self, pred, target):
        # SSIM은 클수록 좋으므로 1에서 빼서 loss로 사용
        ssim_value = ssim(pred, target, data_range=self.data_range, 
                         size_average=self.size_average)
        return 1 - ssim_value


class CombinedLoss(nn.Module):
    """MSE + SSIM Combined Loss"""
    def __init__(self, mse_weight=0.5, ssim_weight=0.5, data_range=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = MSELoss()
        self.ssim_loss = SSIMLoss(data_range=data_range)
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        return self.mse_weight * mse + self.ssim_weight * ssim_loss


class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG features"""
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], use_gpu=True):
        super().__init__()
        
        # VGG19 features 로드
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.feature_layers = feature_layers
        self.feature_extractors = nn.ModuleList()
        
        # 지정된 레이어들에서 feature를 추출하는 모듈 생성
        for i, layer_idx in enumerate(feature_layers):
            if i == 0:
                extractor = nn.Sequential(*list(vgg.children())[:layer_idx+1])
            else:
                prev_idx = feature_layers[i-1]
                extractor = nn.Sequential(*list(vgg.children())[prev_idx+1:layer_idx+1])
            
            # Freeze parameters
            for param in extractor.parameters():
                param.requires_grad = False
            
            self.feature_extractors.append(extractor)
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        loss = 0.0
        
        # 첫 번째 feature 추출
        pred_feat = self.feature_extractors[0](pred)
        target_feat = self.feature_extractors[0](target)
        loss += self.mse_loss(pred_feat, target_feat)
        
        # 나머지 feature들 순차적으로 추출
        for extractor in self.feature_extractors[1:]:
            pred_feat = extractor(pred_feat)
            target_feat = extractor(target_feat)
            loss += self.mse_loss(pred_feat, target_feat)
        
        return loss


class GradientLoss(nn.Module):
    """Gradient Loss for edge preservation"""
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
        
        # Sobel filters for gradient computation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def forward(self, pred, target):
        device = pred.device
        channels = pred.size(1)
        
        # Sobel filters to device and expand for all channels
        sobel_x = self.sobel_x.to(device).expand(channels, 1, 3, 3)
        sobel_y = self.sobel_y.to(device).expand(channels, 1, 3, 3)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        else:  # l2
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


# =============================================================================
# VAE Loss Functions
# =============================================================================

class VAELoss(nn.Module):
    """VAE Loss: Reconstruction + KL Divergence"""
    def __init__(self, beta=1.0, recon_loss_type='mse'):
        super().__init__()
        self.beta = beta
        self.recon_loss_type = recon_loss_type
        
        if recon_loss_type == 'mse':
            self.recon_loss = nn.MSELoss()
        elif recon_loss_type == 'l1':
            self.recon_loss = nn.L1Loss()
        elif recon_loss_type == 'combined':
            self.recon_loss = CombinedLoss()
        else:
            raise ValueError(f"Unknown reconstruction loss type: {recon_loss_type}")
    
    def forward(self, pred, target, mu, logvar):
        # Reconstruction loss
        recon_loss = self.recon_loss(pred, target)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= target.size(0)  # Average over batch
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class BetaVAELoss(nn.Module):
    """β-VAE Loss with adjustable β"""
    def __init__(self, beta_schedule='constant', beta_max=4.0, warmup_epochs=10):
        super().__init__()
        self.beta_schedule = beta_schedule
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.recon_loss = nn.MSELoss()
    
    def update_beta(self, epoch):
        """Update beta value based on schedule"""
        self.current_epoch = epoch
        
        if self.beta_schedule == 'constant':
            return self.beta_max
        elif self.beta_schedule == 'linear':
            return min(self.beta_max, self.beta_max * epoch / self.warmup_epochs)
        elif self.beta_schedule == 'cyclical':
            cycle_length = self.warmup_epochs * 2
            cycle_position = epoch % cycle_length
            if cycle_position < self.warmup_epochs:
                return self.beta_max * cycle_position / self.warmup_epochs
            else:
                return self.beta_max * (cycle_length - cycle_position) / self.warmup_epochs
        else:
            return 1.0
    
    def forward(self, pred, target, mu, logvar, epoch=None):
        if epoch is not None:
            beta = self.update_beta(epoch)
        else:
            beta = self.beta_max
        
        # Reconstruction loss
        recon_loss = self.recon_loss(pred, target)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= target.size(0)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


# =============================================================================
# Image Quality Metrics
# =============================================================================

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred, target, data_range=1.0):
    """Calculate SSIM (Structural Similarity Index)"""
    ssim_value = ssim(pred, target, data_range=data_range, size_average=True)
    return ssim_value.item()


def calculate_ms_ssim(pred, target, data_range=1.0):
    """Calculate MS-SSIM (Multi-Scale SSIM)"""
    ms_ssim_value = ms_ssim(pred, target, data_range=data_range, size_average=True)
    return ms_ssim_value.item()


def calculate_lpips(pred, target, net='alex'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net)
        
        # Convert to [-1, 1] range if needed
        if pred.max() <= 1.0 and pred.min() >= 0.0:
            pred = pred * 2.0 - 1.0
            target = target * 2.0 - 1.0
        
        lpips_value = loss_fn(pred, target)
        return lpips_value.mean().item()
    except ImportError:
        print("LPIPS requires 'lpips' package. Install with: pip install lpips")
        return None


# =============================================================================
# Anomaly Detection Metrics
# =============================================================================

def compute_anomaly_score(model, images, score_type='reconstruction'):
    """
    Compute anomaly scores for given images
    
    Args:
        model: Trained model
        images: Input images tensor
        score_type: Type of score ('reconstruction', 'feature', 'gradient')
    
    Returns:
        scores: Anomaly scores tensor
        reconstructed: Reconstructed images
        error_maps: Per-pixel error maps
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # AutoEncoder
            if 'vae' in model.__class__.__name__.lower():
                # VAE
                reconstructed, latent, features, mu, logvar = model(images)
            else:
                # Regular AE
                reconstructed, latent, features = model(images)
        else:
            # Direct reconstruction
            reconstructed = model(images)
        
        # Compute error maps
        if score_type == 'reconstruction':
            error_maps = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
        elif score_type == 'feature':
            # Feature-based error (if available)
            if 'features' in locals():
                feature_error = torch.mean(features ** 2, dim=1, keepdim=True)
                error_maps = F.interpolate(feature_error, size=images.shape[-2:], mode='bilinear')
            else:
                error_maps = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
        elif score_type == 'gradient':
            # Gradient-based error
            grad_loss = GradientLoss()
            error_maps = grad_loss(reconstructed, images)
        else:
            error_maps = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
        
        # Compute overall scores (mean of error maps)
        scores = torch.mean(error_maps.view(error_maps.size(0), -1), dim=1)
    
    return scores, reconstructed, error_maps


def find_optimal_threshold(scores, labels, method='youden'):
    """
    Find optimal threshold for anomaly detection
    
    Args:
        scores: Anomaly scores
        labels: True labels (0: normal, 1: anomaly)
        method: Threshold selection method ('youden', 'f1', 'precision_recall')
    
    Returns:
        optimal_threshold: Best threshold value
        metrics: Dictionary of metrics at optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    if method == 'youden':
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif method == 'f1':
        # Maximum F1 score
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else pr_thresholds[-1]
        
    elif method == 'precision_recall':
        # Balance precision and recall
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        diff = np.abs(precision - recall)
        optimal_idx = np.argmin(diff)
        optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    else:
        # Default: 95th percentile of normal scores
        normal_scores = scores[labels == 0]
        optimal_threshold = np.percentile(normal_scores, 95) if len(normal_scores) > 0 else np.median(scores)
    
    # Compute metrics at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1_score': f1_score(labels, predictions, zero_division=0),
        'auroc': roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5,
        'aupr': average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    }
    
    return optimal_threshold, metrics


def evaluate_anomaly_detection(scores, labels, threshold=None):
    """
    Comprehensive evaluation of anomaly detection performance
    
    Args:
        scores: Anomaly scores
        labels: True labels (0: normal, 1: anomaly)
        threshold: Detection threshold (if None, use optimal threshold)
    
    Returns:
        results: Dictionary containing all metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(scores, labels, method='youden')
    
    # Binary predictions
    predictions = (scores >= threshold).astype(int)
    
    # Basic classification metrics
    results = {
        'threshold': threshold,
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1_score': f1_score(labels, predictions, zero_division=0),
    }
    
    # ROC and PR metrics (only if both classes present)
    if len(np.unique(labels)) > 1:
        results['auroc'] = roc_auc_score(labels, scores)
        results['aupr'] = average_precision_score(labels, scores)
    else:
        results['auroc'] = 0.5
        results['aupr'] = 0.5
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    results.update({
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
    })
    
    # Score statistics
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    results.update({
        'normal_score_mean': np.mean(normal_scores) if len(normal_scores) > 0 else 0,
        'normal_score_std': np.std(normal_scores) if len(normal_scores) > 0 else 0,
        'anomaly_score_mean': np.mean(anomaly_scores) if len(anomaly_scores) > 0 else 0,
        'anomaly_score_std': np.std(anomaly_scores) if len(anomaly_scores) > 0 else 0,
        'score_separation': (np.mean(anomaly_scores) - np.mean(normal_scores)) if len(anomaly_scores) > 0 and len(normal_scores) > 0 else 0
    })
    
    return results


# =============================================================================
# Pixel-level Anomaly Detection Metrics
# =============================================================================

def evaluate_pixel_anomaly_detection(error_maps, masks, threshold=None):
    """
    Evaluate pixel-level anomaly detection performance
    
    Args:
        error_maps: Pixel-level error maps [B, H, W] or [B, 1, H, W]
        masks: Ground truth masks [B, H, W] or [B, 1, H, W] (0: normal, 1: anomaly)
        threshold: Detection threshold
    
    Returns:
        results: Dictionary containing pixel-level metrics
    """
    # Flatten arrays
    if error_maps.dim() == 4:
        error_maps = error_maps.squeeze(1)
    if masks.dim() == 4:
        masks = masks.squeeze(1)
    
    error_flat = error_maps.flatten().cpu().numpy()
    mask_flat = masks.flatten().cpu().numpy()
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(error_flat, mask_flat, method='youden')
    
    # Binary predictions
    pred_flat = (error_flat >= threshold).astype(int)
    
    # Compute metrics
    results = evaluate_anomaly_detection(error_flat, mask_flat, threshold)
    results['pixel_auroc'] = results['auroc']
    results['pixel_aupr'] = results['aupr']
    
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def get_loss_function(loss_type, **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
    
    Returns:
        loss_function: Loss function instance
    """
    if loss_type == 'mse':
        return MSELoss(**kwargs)
    elif loss_type == 'l1':
        return L1Loss(**kwargs)
    elif loss_type == 'ssim':
        return SSIMLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'perceptual':
        return PerceptualLoss(**kwargs)
    elif loss_type == 'gradient':
        return GradientLoss(**kwargs)
    elif loss_type == 'vae':
        return VAELoss(**kwargs)
    elif loss_type == 'beta_vae':
        return BetaVAELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_metric_functions():
    """Create commonly used metric functions for training"""
    return {
        'psnr': calculate_psnr,
        'ssim': calculate_ssim,
        'ms_ssim': calculate_ms_ssim
    }


# =============================================================================
# Test Functions
# =============================================================================

def test_loss_functions():
    """Test all loss functions"""
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    channels = 3
    height, width = 64, 64
    
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    mu = torch.randn(batch_size, 512)
    logvar = torch.randn(batch_size, 512)
    
    # Test reconstruction losses
    loss_functions = [
        ('MSE', MSELoss()),
        ('L1', L1Loss()),
        ('SSIM', SSIMLoss()),
        ('Combined', CombinedLoss()),
        ('Gradient', GradientLoss()),
    ]
    
    for name, loss_fn in loss_functions:
        try:
            loss = loss_fn(pred, target)
            print(f"{name} Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"{name} Loss: Error - {e}")
    
    # Test VAE losses
    vae_loss = VAELoss()
    total_loss, recon_loss, kl_loss = vae_loss(pred, target, mu, logvar)
    print(f"VAE Loss - Total: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
    
    print("\nLoss function tests completed!")


def test_metrics():
    """Test metric functions"""
    print("=" * 60)
    print("Testing Metric Functions")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    channels = 3
    height, width = 64, 64
    
    pred = torch.rand(batch_size, channels, height, width)
    target = torch.rand(batch_size, channels, height, width)
    
    # Test image quality metrics
    psnr = calculate_psnr(pred, target)
    ssim_val = calculate_ssim(pred, target)
    ms_ssim_val = calculate_ms_ssim(pred, target)
    
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"MS-SSIM: {ms_ssim_val:.4f}")
    
    # Test anomaly detection metrics
    scores = np.random.rand(100)
    labels = np.random.choice([0, 1], 100, p=[0.8, 0.2])
    
    threshold, metrics = find_optimal_threshold(scores, labels)
    print(f"\nOptimal threshold: {threshold:.4f}")
    print(f"Metrics: {metrics}")
    
    results = evaluate_anomaly_detection(scores, labels)
    print(f"\nEvaluation results: {results}")
    
    print("\nMetric function tests completed!")


if __name__ == "__main__":
    print("Vision Anomaly Detection Metrics and Loss Functions")
    print("=" * 60)
    
    # Test loss functions
    test_loss_functions()
    
    print()
    
    # Test metrics
    test_metrics()