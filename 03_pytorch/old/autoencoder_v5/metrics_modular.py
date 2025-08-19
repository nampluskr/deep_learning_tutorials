import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import SSIM, MS_SSIM


# =============================================================================
# Factory Functions
# =============================================================================

def get_loss_module(loss_type='combined', **kwargs):
    """Get loss module by name"""
    loss_modules = {
        'mse': lambda: MSELoss(),
        'l1': lambda: L1Loss(),
        'l2': lambda: L2Loss(),
        'smooth_l1': lambda: SmoothL1Loss(**kwargs),
        'charbonnier': lambda: CharbonnierLoss(**kwargs),
        'combined': lambda: CombinedLoss(**kwargs),
        'ssim': lambda: SSIMLoss(**kwargs),
        'ms_ssim': lambda: MSSSIMLoss(**kwargs),
        'perceptual': lambda: PerceptualLoss(**kwargs),
        'perceptual_l1': lambda: PerceptualL1Loss(**kwargs),
        'edge_preserving': lambda: EdgePreservingLoss(**kwargs),
        'focal_mse': lambda: FocalMSELoss(**kwargs),
        'robust': lambda: RobustLoss(**kwargs)
    }
    
    if loss_type not in loss_modules:
        available = ', '.join(loss_modules.keys())
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
    
    return loss_modules[loss_type]()


def get_metric_modules(metric_names=None, **kwargs):
    """Get metric modules by names"""
    all_metrics = {
        'ssim': lambda: SSIMMetric(**kwargs),
        'psnr': lambda: PSNRMetric(**kwargs),
        'ms_ssim': lambda: MSSSIMMetric(**kwargs),
        'pixel_accuracy': lambda: PixelAccuracy(**kwargs),
        'binary_accuracy': lambda: BinaryAccuracy(**kwargs)
    }
    
    if metric_names is None:
        return {name: metric_fn() for name, metric_fn in all_metrics.items()}
    
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    selected_metrics = {}
    for name in metric_names:
        if name not in all_metrics:
            available = ', '.join(all_metrics.keys())
            raise ValueError(f"Unknown metric: {name}. Available: {available}")
        selected_metrics[name] = all_metrics[name]()
    
    return selected_metrics


# =============================================================================
# Basic Loss Modules
# =============================================================================

class MSELoss(nn.Module):
    """Mean Squared Error loss module"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.mse(pred, target)


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss module"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.l1(pred, target)


class L2Loss(nn.Module):
    """L2 loss module (same as MSE)"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.mse(pred, target)


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss (Huber loss) module"""
    
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction=reduction)
    
    def forward(self, pred, target):
        return self.smooth_l1(pred, target)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss module (differentiable variant of L1 loss)"""
    
    def __init__(self, epsilon=1e-3, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# =============================================================================
# Perceptual Quality Loss Modules
# =============================================================================

class SSIMLoss(nn.Module):
    """Structural Similarity Index loss module"""
    
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.ssim = SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)


class MSSSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity Index loss module"""
    
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, pred, target):
        return 1 - self.ms_ssim(pred, target)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, feature_layers=[1, 6, 11, 20], weights=None, requires_grad=False):
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.ModuleList()
        
        # Extract specified layers
        current_layer = 0
        for layer_idx in feature_layers:
            layers = []
            for i in range(current_layer, layer_idx + 1):
                layers.append(vgg[i])
            self.feature_extractor.append(nn.Sequential(*layers))
            current_layer = layer_idx + 1
        
        # Freeze VGG parameters if not trainable
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Feature weights
        if weights is None:
            self.weights = [1.0] * len(feature_layers)
        else:
            self.weights = weights
        
        # Normalization for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_input(self, x):
        """Normalize input for VGG (ImageNet normalization)"""
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        """Extract VGG features from input"""
        x = self.normalize_input(x)
        features = []
        for extractor in self.feature_extractor:
            x = extractor(x)
            features.append(x)
        return features
    
    def forward(self, pred, target):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for i, (pred_feat, target_feat, weight) in enumerate(zip(pred_features, target_features, self.weights)):
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class PerceptualL1Loss(nn.Module):
    """Combined L1 + Perceptual loss module"""
    
    def __init__(self, l1_weight=0.8, perceptual_weight=0.2, **perceptual_kwargs):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss(**perceptual_kwargs)
    
    def forward(self, pred, target):
        l1_val = self.l1_loss(pred, target)
        perceptual_val = self.perceptual_loss(pred, target)
        
        return self.l1_weight * l1_val + self.perceptual_weight * perceptual_val


# =============================================================================
# Advanced Loss Modules
# =============================================================================

class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss module"""
    
    def __init__(self, l1_weight=0.7, ssim_weight=0.3, **ssim_kwargs):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss(**ssim_kwargs)
    
    def forward(self, pred, target):
        l1_val = self.l1_loss(pred, target)
        ssim_val = self.ssim_loss(pred, target)
        
        return self.l1_weight * l1_val + self.ssim_weight * ssim_val


class RobustLoss(nn.Module):
    """Robust loss combining L1 and L2"""
    
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = L1Loss()
        self.l2_loss = MSELoss()
    
    def forward(self, pred, target):
        l1_val = self.l1_loss(pred, target)
        l2_val = self.l2_loss(pred, target)
        
        return self.alpha * l2_val + (1 - self.alpha) * l1_val


class FocalMSELoss(nn.Module):
    """Focal MSE loss that focuses on harder examples"""
    
    def __init__(self, alpha=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        mse_val = (pred - target) ** 2
        pt = torch.exp(-mse_val)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        return torch.mean(focal_weight * mse_val)


class EdgePreservingLoss(nn.Module):
    """L1 loss with edge preservation penalty"""
    
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.edge_weight = edge_weight
        self.l1_loss = L1Loss()
        
        # Register Sobel filters as buffers (automatically moved to device)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_edges(self, x):
        """Compute edge maps using Sobel filters"""
        # Expand filters for all channels
        sobel_x = self.sobel_x.repeat(x.size(1), 1, 1, 1)
        sobel_y = self.sobel_y.repeat(x.size(1), 1, 1, 1)
        
        edge_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
        
        return edge_x, edge_y
    
    def forward(self, pred, target):
        l1_val = self.l1_loss(pred, target)
        
        pred_edge_x, pred_edge_y = self.compute_edges(pred)
        target_edge_x, target_edge_y = self.compute_edges(target)
        
        edge_loss = F.l1_loss(pred_edge_x, target_edge_x) + F.l1_loss(pred_edge_y, target_edge_y)
        
        return l1_val + self.edge_weight * edge_loss


# =============================================================================
# Metric Modules
# =============================================================================

class PSNRMetric(nn.Module):
    """Peak Signal-to-Noise Ratio metric module"""
    
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'), device=pred.device)
        return 20 * torch.log10(self.max_val / torch.sqrt(mse))


class SSIMMetric(nn.Module):
    """Structural Similarity Index metric module"""
    
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.ssim = SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, pred, target):
        return self.ssim(pred, target)


class MSSSIMMetric(nn.Module):
    """Multi-Scale Structural Similarity Index metric module"""
    
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, pred, target):
        return self.ms_ssim(pred, target)


class BinaryAccuracy(nn.Module):
    """Binary classification accuracy module"""
    
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred, target):
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        return (pred_binary == target_binary).float().mean()


class PixelAccuracy(nn.Module):
    """Pixel-wise accuracy module"""
    
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.binary_accuracy = BinaryAccuracy(threshold=threshold)
    
    def forward(self, pred, target):
        return self.binary_accuracy(pred, target)


# =============================================================================
# Anomaly Detection Utility Modules
# =============================================================================

class ReconstructionErrorComputer(nn.Module):
    """Module for computing reconstruction errors for anomaly detection"""
    
    def __init__(self, method='mse', reduction='none'):
        super().__init__()
        self.method = method
        self.reduction = reduction
        
        if method == 'mse':
            self.criterion = MSELoss(reduction='none')
        elif method == 'l1':
            self.criterion = L1Loss(reduction='none')
        elif method == 'ssim':
            self.criterion = SSIMLoss(size_average=False)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, pred, target):
        if self.method in ['mse', 'l1']:
            # Compute per-pixel error then average over spatial dimensions
            error_map = self.criterion(pred, target)
            if self.reduction == 'none':
                # Return per-sample error
                return torch.mean(error_map, dim=[1, 2, 3])
            else:
                return error_map
        elif self.method == 'ssim':
            if self.reduction == 'none':
                # Compute SSIM for each sample in batch
                batch_errors = []
                for i in range(pred.size(0)):
                    ssim_val = 1 - self.criterion(pred[i:i+1], target[i:i+1])
                    batch_errors.append(ssim_val)
                return torch.stack(batch_errors)
            else:
                return 1 - self.criterion(pred, target)


class BatchMetrics(nn.Module):
    """Module for computing metrics across batch samples"""
    
    def __init__(self):
        super().__init__()
        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric(size_average=False)
    
    def batch_psnr(self, pred, target, max_val=1.0):
        """Compute PSNR for each sample in batch"""
        mse_per_sample = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        # Handle zero MSE case
        psnr_values = torch.where(
            mse_per_sample > 0,
            20 * torch.log10(max_val / torch.sqrt(mse_per_sample)),
            torch.tensor(float('inf'), device=pred.device)
        )
        return psnr_values
    
    def batch_ssim(self, pred, target):
        """Compute SSIM for each sample in batch"""
        batch_size = pred.size(0)
        ssim_values = []
        
        for i in range(batch_size):
            ssim_val = self.ssim_metric(pred[i:i+1], target[i:i+1])
            ssim_values.append(ssim_val)
        
        return torch.stack(ssim_values)
    
    def forward(self, pred, target, metric='psnr'):
        if metric == 'psnr':
            return self.batch_psnr(pred, target)
        elif metric == 'ssim':
            return self.batch_ssim(pred, target)
        else:
            raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred = torch.randn(4, 3, 64, 64).to(device)
    target = torch.randn(4, 3, 64, 64).to(device)
    
    # Test loss modules
    print("Testing loss modules:")
    loss_module = get_loss_module('perceptual').to(device)
    loss = loss_module(pred, target)
    print(f"Perceptual loss: {loss.item():.4f}")
    
    # Test metric modules
    print("\nTesting metric modules:")
    metrics = get_metric_modules(['ssim', 'psnr'])
    for name, metric_module in metrics.items():
        metric_module = metric_module.to(device)
        value = metric_module(pred, target)
        print(f"{name}: {value.item():.4f}")
    
    # Test reconstruction error computer
    print("\nTesting reconstruction error computer:")
    error_computer = ReconstructionErrorComputer(method='mse').to(device)
    errors = error_computer(pred, target)
    print(f"MSE errors shape: {errors.shape}")
    print(f"MSE errors: {errors}")
    
    # Test batch metrics
    print("\nTesting batch metrics:")
    batch_metrics = BatchMetrics().to(device)
    psnr_batch = batch_metrics(pred, target, metric='psnr')
    print(f"Batch PSNR: {psnr_batch}")