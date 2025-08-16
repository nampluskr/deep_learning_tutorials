import torch
import torch.nn.functional as F
import numpy as np

from pytorch_msssim import ssim as pytorch_ssim
from pytorch_msssim import ms_ssim as pytorch_ms_ssim


# =============================================================================
# Factory Functions
# =============================================================================

def get_loss_fn(loss_type='combined'):
    """Get loss function by name"""
    loss_functions = {
        'mse': mse_loss,
        'l1': l1_loss,
        'l2': l2_loss,
        'smooth_l1': smooth_l1_loss,
        'charbonnier': charbonnier_loss,
        'combined': combined_loss,
        'ssim': ssim_loss,
        'ms_ssim': ms_ssim_loss,
        'perceptual_l1': perceptual_l1_loss,
        'edge_preserving': edge_preserving_loss,
        'focal_mse': focal_mse_loss,
        'robust': robust_loss
    }
    
    if loss_type not in loss_functions:
        available = ', '.join(loss_functions.keys())
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
    
    return loss_functions[loss_type]


def get_metrics(metric_names=None):
    """Get metric functions by names"""
    all_metrics = {
        'ssim': ssim,
        'psnr': psnr,
        'ms_ssim': ms_ssim,
        'pixel_accuracy': pixel_accuracy,
        'binary_accuracy': binary_accuracy
    }
    
    if metric_names is None:
        return all_metrics
    
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    selected_metrics = {}
    for name in metric_names:
        if name not in all_metrics:
            available = ', '.join(all_metrics.keys())
            raise ValueError(f"Unknown metric: {name}. Available: {available}")
        selected_metrics[name] = all_metrics[name]
    
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


# =============================================================================
# Utility Functions for Anomaly Detection
# =============================================================================

def compute_reconstruction_error(pred, target, method='mse', reduction='none'):
    """Compute reconstruction error using specified method"""
    if method == 'mse':
        if reduction == 'none':
            error = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        else:
            error = F.mse_loss(pred, target, reduction=reduction)
    elif method == 'l1':
        if reduction == 'none':
            error = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])
        else:
            error = F.l1_loss(pred, target, reduction=reduction)
    elif method == 'l2':
        if reduction == 'none':
            error = torch.sqrt(torch.mean((pred - target) ** 2, dim=[1, 2, 3]))
        else:
            error = torch.sqrt(F.mse_loss(pred, target, reduction=reduction))
    elif method == 'ssim':
        # SSIM-based error (1 - SSIM for each sample)
        if reduction == 'none':
            # Compute SSIM for each sample in batch
            batch_errors = []
            for i in range(pred.size(0)):
                ssim_val = ssim(pred[i:i+1], target[i:i+1])
                batch_errors.append(1 - ssim_val)
            error = torch.stack(batch_errors)
        else:
            error = 1 - ssim(pred, target)
    else:
        raise ValueError(f"Unknown method: {method}. Available: mse, l1, l2, ssim")

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
    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        if mad > 0:
            return (scores - median) / (1.4826 * mad)  # 1.4826 for normal distribution
        else:
            return scores
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# Batch-wise Metrics for Efficiency
# =============================================================================

def batch_psnr(pred, target, max_val=1.0):
    """Compute PSNR for each sample in batch"""
    mse_per_sample = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    # Handle zero MSE case
    psnr_values = torch.where(
        mse_per_sample > 0,
        20 * torch.log10(max_val / torch.sqrt(mse_per_sample)),
        torch.tensor(float('inf'), device=pred.device)
    )
    return psnr_values


def batch_ssim(pred, target, data_range=1.0):
    """Compute SSIM for each sample in batch"""
    batch_size = pred.size(0)
    ssim_values = []
    
    for i in range(batch_size):
        ssim_val = ssim(pred[i:i+1], target[i:i+1], data_range=data_range)
        ssim_values.append(ssim_val)
    
    return torch.stack(ssim_values)


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
    
    # Test reconstruction error
    print("\nTesting reconstruction error:")
    errors = compute_reconstruction_error(pred, target, method='mse')
    print(f"MSE errors shape: {errors.shape}")
    print(f"MSE errors: {errors}")