"""Metrics specific to gradient-based anomaly detection models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssim import ssim
from .lpips import LPIPS


class PSNRMetric(nn.Module):
    """Peak Signal-to-Noise Ratio metric."""
    
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, preds, targets):
        mse = F.mse_loss(preds, targets, reduction='mean')
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(torch.tensor(self.max_val) / torch.sqrt(mse))
        return psnr.item()


class SSIMMetric(nn.Module):
    """Structural Similarity Index Measure metric."""
    
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        ssim_value = ssim(preds, targets, data_range=self.data_range, size_average=True)
        return ssim_value.item()


class LPIPSMetric(nn.Module):
    """Learned Perceptual Image Patch Similarity metric."""
    
    def __init__(self, net='alex', data_range=1.0):
        super().__init__()
        self.net = net
        self.data_range = data_range
        
        try:
            self.lpips_model = LPIPS(
                pretrained=True, 
                net=net, 
                eval_mode=True,
                verbose=True,
                model_path=None  # Will use backbones folder automatically
            )
        except Exception as e:
            print(f"Warning: LPIPS model initialization failed: {e}")
            self.lpips_model = None

    def forward(self, preds, targets):
        if self.lpips_model is None:
            # Fallback to MSE if LPIPS unavailable
            mse_loss = F.mse_loss(preds, targets, reduction='mean')
            return mse_loss.item()
        
        # LPIPS expects values in [-1, 1] range if normalize=True
        # or [0, 1] range if normalize=False
        with torch.no_grad():
            if self.data_range == 1.0:
                # Input is [0, 1], use normalize=True for [-1, 1] conversion
                lpips_value = self.lpips_model(preds, targets, normalize=True)
            else:
                # Input is already in correct range
                lpips_value = self.lpips_model(preds, targets, normalize=False)
            
            return lpips_value.mean().item()


class FeatureSimilarityMetric(nn.Module):
    """Feature similarity metric for distillation-based models."""
    
    def __init__(self, similarity_fn='cosine'):
        super().__init__()
        self.similarity_fn = similarity_fn

    def forward(self, teacher_features, student_features):
        """Compute similarity between teacher and student features."""
        if self.similarity_fn == 'cosine':
            teacher_norm = F.normalize(teacher_features, dim=1)
            student_norm = F.normalize(student_features, dim=1)
            similarity = F.cosine_similarity(teacher_norm, student_norm, dim=1)
            return torch.mean(similarity).item()
        
        elif self.similarity_fn == 'mse':
            mse = F.mse_loss(teacher_features, student_features, reduction='mean')
            return mse.item()
        
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_fn}")
