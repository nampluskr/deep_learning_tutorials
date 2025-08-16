import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from pytorch_msssim import ssim, ms_ssim


# =============================================================================
# Metric Factory Functions
# =============================================================================

def get_metrics(include_perceptual=False):
    metrics = {
        # 'acc': binary_accuracy,
        'ssim': ssim_loss,
        'psnr': psnr,
    }
    if include_perceptual:
        metrics['perceptual'] = perceptual_loss

    return metrics

# =============================================================================
# Basic Reconstruction Metrics
# =============================================================================

def mse_loss(pred, target):
    """Mean Squared Error loss"""
    return F.mse_loss(pred, target)


def l1_loss(pred, target):
    """L1 (Mean Absolute Error) loss"""
    return F.l1_loss(pred, target)


def l2_loss(pred, target):
    """L2 loss (same as MSE)"""
    return mse_loss(pred, target)


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss (Huber loss)"""
    return F.smooth_l1_loss(pred, target, beta=beta)


# =============================================================================
# Classification Metrics (for evaluation)
# =============================================================================

def binary_accuracy(pred, target, threshold=0.5):
    """Binary classification accuracy"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    return (pred_binary == target_binary).float().mean()


def pixel_accuracy(pred, target, threshold=0.5):
    """Pixel-wise accuracy for binary segmentation"""
    return binary_accuracy(pred, target, threshold)


# =============================================================================
# Perceptual Quality Metrics
# =============================================================================

def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim_loss(pred, target, data_range=1.0, size_average=True):
    """Structural Similarity Index loss"""
    return ssim(pred, target, data_range=data_range, size_average=size_average)


def ms_ssim_loss(pred, target, data_range=1.0, size_average=True):
    """Multi-Scale Structural Similarity Index loss"""
    return ms_ssim(pred, target, data_range=data_range, size_average=size_average)


def ssim_l1_loss(pred, target, ssim_weight=0.5, l1_weight=0.5):
    """Combined SSIM + L1 loss for better perceptual quality"""
    ssim_val = ssim_loss(pred, target)
    l1_val = l1_loss(pred, target)
    return ssim_weight * (1 - ssim_val) + l1_weight * l1_val


def perceptual_loss(pred, target, feature_layers=[0, 1, 2, 3]):
    """Perceptual loss using VGG features (simplified version)"""
    # For now, return L1 loss as fallback
    return l1_loss(pred, target)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_reconstruction_error(pred, target, method='mse'):
    """Compute reconstruction error using specified method"""
    if method == 'mse':
        error = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    elif method == 'l1':
        error = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])
    elif method == 'l2':
        error = torch.sqrt(torch.mean((pred - target) ** 2, dim=[1, 2, 3]))
    else:
        raise ValueError(f"Unknown method: {method}")

    return error


def normalize_scores(scores, method='minmax'):
    """Normalize anomaly scores"""
    scores = np.array(scores)

    if method == 'minmax':
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return scores
    elif method == 'zscore':
        mean_score, std_score = scores.mean(), scores.std()
        if std_score > 0:
            return (scores - mean_score) / std_score
        else:
            return scores
    else:
        raise ValueError(f"Unknown normalization method: {method}")


if __name__ == "__main__":
    # Example usage
    print("Available metrics:")

    # Test basic metrics
    pred = torch.rand(4, 3, 64, 64)
    target = torch.rand(4, 3, 64, 64)

    reconstruction_metrics = get_reconstruction_metrics()
    print(f"Reconstruction metrics: {list(reconstruction_metrics.keys())}")

    for name, metric_fn in reconstruction_metrics.items():
        try:
            value = metric_fn(pred, target)
            print(f"{name}: {value:.4f}")
        except Exception as e:
            print(f"{name}: Error - {e}")