import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_msssim import ssim as pytorch_ssim
from pytorch_msssim import ms_ssim as pytorch_ms_ssim


# =============================================================================
# Metric Factory Functions
# =============================================================================

def load_criterion():
    return combined_loss

def load_metrics(include_perceptual=False):
    metrics = {
        'ssim': ssim,
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


def ssim(pred, target, data_range=1.0, size_average=True):
    """Structural Similarity Index metric (higher is better)"""
    return pytorch_ssim(pred, target, 
                        data_range=data_range, 
                        size_average=size_average)


def ssim_loss(pred, target, data_range=1.0, size_average=True):
    """Structural Similarity Index loss (lower is better)"""
    return 1 - pytorch_ssim(pred, target, 
                            data_range=data_range, 
                            size_average=size_average)


def ms_ssim(pred, target, data_range=1.0, size_average=True):
    """Multi-Scale Structural Similarity Index metric (higher is better)"""
    return pytorch_ms_ssim(pred, target, 
                           data_range=data_range,
                           size_average=size_average)


def ms_ssim_loss(pred, target, data_range=1.0, size_average=True):
    """Multi-Scale Structural Similarity Index loss (lower is better)"""
    return 1 - pytorch_ms_ssim(pred, target, 
                               data_range=data_range, 
                               size_average=size_average)


def ssim_l1_loss(pred, target, ssim_weight=0.5, l1_weight=0.5):
    """Combined SSIM + L1 loss for better perceptual quality"""
    ssim_loss_val = ssim_loss(pred, target)
    l1_val = l1_loss(pred, target)
    return ssim_weight * ssim_loss_val + l1_weight * l1_val


def perceptual_loss(pred, target, feature_layers=[0, 1, 2, 3]):
    """Perceptual loss using VGG features (simplified version)"""
    # For now, return L1 loss as fallback
    return l1_loss(pred, target)


# =============================================================================
# Advanced Loss Functions for Autoencoder Training
# =============================================================================

def combined_loss(pred, target, l1_weight=0.7, ssim_weight=0.3):
    """Combined L1 + SSIM loss for better reconstruction quality"""
    l1_val = l1_loss(pred, target)
    ssim_val = ssim_loss(pred, target)
    return l1_weight * l1_val + ssim_weight * ssim_val


def perceptual_l1_loss(pred, target, l1_weight=0.8, perceptual_weight=0.2):
    """Combined L1 + Perceptual loss (using L1 as simplified perceptual)"""
    l1_val = l1_loss(pred, target)
    # Simplified perceptual loss using gradient similarity
    pred_grad = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]) + \
                torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_grad = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]) + \
                  torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    grad_loss = F.l1_loss(pred_grad, target_grad)
    
    return l1_weight * l1_val + perceptual_weight * grad_loss


def robust_loss(pred, target, alpha=0.2):
    """Robust loss combining L1 and L2 (less sensitive to outliers than MSE)"""
    l1_val = l1_loss(pred, target)
    l2_val = mse_loss(pred, target)
    return alpha * l2_val + (1 - alpha) * l1_val


def focal_mse_loss(pred, target, alpha=2.0, gamma=2.0):
    """Focal MSE loss that focuses on harder examples"""
    mse_val = (pred - target) ** 2
    pt = torch.exp(-mse_val)
    focal_weight = alpha * (1 - pt) ** gamma
    return torch.mean(focal_weight * mse_val)


def edge_preserving_loss(pred, target, edge_weight=0.1):
    """L1 loss with edge preservation penalty"""
    l1_val = l1_loss(pred, target)
    
    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    device = pred.device
    sobel_x = sobel_x.to(device).repeat(pred.size(1), 1, 1, 1)
    sobel_y = sobel_y.to(device).repeat(pred.size(1), 1, 1, 1)
    
    # Calculate edges
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
    target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
    target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))
    
    edge_loss = F.l1_loss(pred_edge_x, target_edge_x) + F.l1_loss(pred_edge_y, target_edge_y)
    
    return l1_val + edge_weight * edge_loss


def charbonnier_loss(pred, target, epsilon=1e-3):
    """Charbonnier loss (differentiable variant of L1 loss)"""
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return torch.mean(loss)


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
    pass