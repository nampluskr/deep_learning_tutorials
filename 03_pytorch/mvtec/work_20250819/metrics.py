"""
Loss functions and evaluation metrics for anomaly detection
Contains various loss functions and metrics for training and evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_msssim import ssim as pytorch_ssim
from pytorch_msssim import ms_ssim as pytorch_ms_ssim


# =============================================================================
# Factory Functions
# =============================================================================

def get_loss_fn(loss_type='combined', **loss_params):
    """Get loss function with configurable parameters"""
    loss_functions = {
        'mse': lambda: mse_loss,
        'l1': lambda: l1_loss,
        'l2': lambda: l2_loss,
        'smooth_l1': lambda: lambda pred, target: smooth_l1_loss(
            pred, target, beta=loss_params.get('beta', 1.0)
        ),
        'charbonnier': lambda: lambda pred, target: charbonnier_loss(
            pred, target, epsilon=loss_params.get('epsilon', 1e-3)
        ),
        'combined': lambda: lambda pred, target: combined_loss(
            pred, target, 
            l1_weight=loss_params.get('l1_weight', 0.7),
            ssim_weight=loss_params.get('ssim_weight', 0.3)
        ),
        'ssim': lambda: ssim_loss,
        'ms_ssim': lambda: ms_ssim_loss,
        'perceptual_l1': lambda: lambda pred, target: perceptual_l1_loss(
            pred, target,
            l1_weight=loss_params.get('l1_weight', 0.8),
            perceptual_weight=loss_params.get('perceptual_weight', 0.2)
        ),
        'edge_preserving': lambda: lambda pred, target: edge_preserving_loss(
            pred, target, edge_weight=loss_params.get('edge_weight', 0.1)
        ),
        'focal_mse': lambda: lambda pred, target: focal_mse_loss(
            pred, target,
            alpha=loss_params.get('alpha', 2.0),
            gamma=loss_params.get('gamma', 2.0)
        ),
        'robust': lambda: lambda pred, target: robust_loss(
            pred, target, alpha=loss_params.get('alpha', 0.2)
        )
    }
    
    if loss_type not in loss_functions:
        available = ', '.join(loss_functions.keys())
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
    
    loss_fn = loss_functions[loss_type]()
    print(f" > Creating loss function: {loss_type} with params: {loss_params}")
    
    return loss_fn


def get_metrics(metric_names=None, **metric_params):
    """Get metric functions with unified return format"""
    metric_functions = {
        'psnr': lambda: lambda pred, target: psnr(
            pred, target, max_val=metric_params.get('max_val', 1.0)
        ),
        'ssim': lambda: lambda pred, target: ssim(
            pred, target, 
            data_range=metric_params.get('data_range', 1.0),
            size_average=metric_params.get('size_average', True)
        ),
        'ms_ssim': lambda: lambda pred, target: ms_ssim(
            pred, target,
            data_range=metric_params.get('data_range', 1.0),
            size_average=metric_params.get('size_average', True)
        ),
        'pixel_accuracy': lambda: lambda pred, target: pixel_accuracy(
            pred, target, threshold=metric_params.get('threshold', 0.5)
        ),
        'binary_accuracy': lambda: lambda pred, target: binary_accuracy(
            pred, target, threshold=metric_params.get('threshold', 0.5)
        )
    }
    
    if metric_names is None:
        selected_metrics = {name: func() for name, func in metric_functions.items()}
        print(f"Creating all metrics: {list(selected_metrics.keys())}")
        return selected_metrics
    
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    invalid_names = [name for name in metric_names if name not in metric_functions]
    if invalid_names:
        available = ', '.join(metric_functions.keys())
        raise ValueError(f"Unknown metrics: {invalid_names}. Available: {available}")
    
    selected_metrics = {name: metric_functions[name]() for name in metric_names}
    print(f" > Creating metrics: {list(selected_metrics.keys())} with params: {metric_params}")
    
    return selected_metrics


# =============================================================================
# Basic Reconstruction Loss Functions
# =============================================================================

def mse_loss(pred, target, reduction='mean'):
    """Mean Squared Error loss"""
    return F.mse_loss(pred, target, reduction=reduction)


def l1_loss(pred, target, reduction='mean'):
    """L1 (Mean Absolute Error) loss"""
    return F.l1_loss(pred, target, reduction=reduction)


def l2_loss(pred, target, reduction='mean'):
    """L2 loss (same as MSE)"""
    return mse_loss(pred, target, reduction)


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    """Smooth L1 loss (Huber loss)"""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)


def charbonnier_loss(pred, target, epsilon=1e-3, reduction='mean'):
    """Charbonnier loss (differentiable variant of L1 loss)"""
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# =============================================================================
# Perceptual Quality Metrics
# =============================================================================

def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
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


# =============================================================================
# Classification Metrics
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
# Advanced Loss Functions
# =============================================================================

def combined_loss(pred, target, l1_weight=0.7, ssim_weight=0.3):
    """Combined L1 + SSIM loss for better reconstruction quality"""
    l1_val = l1_loss(pred, target)
    ssim_val = ssim_loss(pred, target)
    return l1_weight * l1_val + ssim_weight * ssim_val


def perceptual_l1_loss(pred, target, l1_weight=0.8, perceptual_weight=0.2):
    """Combined L1 + Gradient-based perceptual loss"""
    l1_val = l1_loss(pred, target)
    
    # Gradient-based perceptual loss
    pred_grad_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    pred_grad_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_grad_x = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    target_grad_y = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    
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


# Global variables for edge filters (cached for efficiency)
_sobel_x = None
_sobel_y = None

def edge_preserving_loss(pred, target, edge_weight=0.1):
    """L1 loss with edge preservation penalty"""
    global _sobel_x, _sobel_y
    
    l1_val = l1_loss(pred, target)
    
    device = pred.device
    # Initialize Sobel filters if not cached or device changed
    if _sobel_x is None or _sobel_x.device != device:
        _sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        _sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Expand filters for all channels
    sobel_x = _sobel_x.repeat(pred.size(1), 1, 1, 1)
    sobel_y = _sobel_y.repeat(pred.size(1), 1, 1, 1)
    
    # Calculate edges
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
    target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
    target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))
    
    edge_loss = F.l1_loss(pred_edge_x, target_edge_x) + F.l1_loss(pred_edge_y, target_edge_y)
    
    return l1_val + edge_weight * edge_loss


if __name__ == "__main__":
    # Example usage
    pred = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    
    # Test loss functions
    print("Testing loss functions:")
    loss_fn = get_loss_fn('combined')
    loss = loss_fn(pred, target)
    print(f"Combined loss: {loss.item():.4f}")
    
    # Test metrics
    print("\nTesting metrics:")
    metrics = get_metrics(['ssim', 'psnr'])
    for name, metric_fn in metrics.items():
        value = metric_fn(pred, target)
        print(f"{name}: {value.item():.4f}")
