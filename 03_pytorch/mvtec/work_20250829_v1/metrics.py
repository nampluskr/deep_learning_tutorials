# metrics.py
# Standard anomaly detection metrics + OLED-specific metrics + LPIPS (nn.Module 기반)

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as sk_ssim, peak_signal_noise_ratio as sk_psnr
from skimage.color import rgb2lab
from skimage.measure import label, regionprops
from scipy.fftpack import fft2
import pywt


# -------------------------
# Standard Metrics
# -------------------------

class AUROC(nn.Module):
    """AUROC metric (image-level)"""
    def __init__(self):
        super().__init__()

    def forward(self, scores, labels):
        scores = np.array(scores).flatten()
        labels = np.array(labels).flatten()
        return roc_auc_score(labels, scores)


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_np = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
        y_np = y.detach().cpu().numpy().transpose(0, 2, 3, 1)
        psnr_vals = [sk_psnr(xi, yi, data_range=1.0) for xi, yi in zip(x_np, y_np)]
        return float(np.mean(psnr_vals))


class SSIM(nn.Module):
    """Structural Similarity Index"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_np = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
        y_np = y.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ssim_vals = [sk_ssim(xi, yi, channel_axis=-1, data_range=1.0) for xi, yi in zip(x_np, y_np)]
        return float(np.mean(ssim_vals))


# -------------------------
# OLED-specific Metrics
# -------------------------

class DeltaE2000(nn.Module):
    """ΔE2000 color difference"""
    def __init__(self):
        super().__init__()

    def forward(self, img1: np.ndarray, img2: np.ndarray):
        lab1, lab2 = rgb2lab(img1), rgb2lab(img2)
        diff = lab1 - lab2
        return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=-1))))


class FFTPhaseDifference(nn.Module):
    """FFT phase difference"""
    def __init__(self):
        super().__init__()

    def forward(self, img1: np.ndarray, img2: np.ndarray):
        f1, f2 = np.angle(fft2(img1.mean(axis=-1))), np.angle(fft2(img2.mean(axis=-1)))
        return float(np.mean(np.abs(f1 - f2)))


class WaveletDifference(nn.Module):
    """Wavelet coefficient difference"""
    def __init__(self, wavelet="haar"):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, img1: np.ndarray, img2: np.ndarray):
        coeffs1, coeffs2 = pywt.dwt2(img1.mean(axis=-1), self.wavelet), pywt.dwt2(img2.mean(axis=-1), self.wavelet)
        diff = [np.mean(np.abs(c1 - c2)) for (c1, c2) in zip(coeffs1, coeffs2)]
        return float(np.mean(diff))


class ConnectedComponentScore(nn.Module):
    """Connected component analysis on anomaly map"""
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, anomaly_map: np.ndarray):
        binary = (anomaly_map > self.threshold).astype(np.uint8)
        labeled = label(binary)
        regions = regionprops(labeled)
        return float(np.mean([r.area for r in regions])) if len(regions) else 0.0


# -------------------------
# LPIPS (local copy)
# -------------------------

class NetLinLayer(nn.Module):
    """A single linear layer used inside LPIPS metric."""
    def __init__(self, chn_in, chn_out=1, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS(nn.Module):
    """LPIPS perceptual similarity metric (copied from original)."""

    def __init__(self, backbone="alex", pretrained_weights=None):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "alex":
            from torchvision.models import alexnet
            net = alexnet(weights=None).features
            channels = [64, 192, 384, 256, 256]
        elif backbone == "vgg":
            from torchvision.models import vgg16
            net = vgg16(weights=None).features
            channels = [64, 128, 256, 512, 512]
        elif backbone == "squeeze":
            from torchvision.models import squeezenet1_1
            net = squeezenet1_1(weights=None).features
            channels = [64, 128, 256, 384, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.lin_layers = nn.ModuleList([NetLinLayer(c) for c in channels])

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x, y):
        # Normalize to [-1,1]
        x, y = (x*2-1), (y*2-1)
        feats_x, feats_y = [], []
        cur_x, cur_y = x, y
        for layer in self.net:
            cur_x = layer(cur_x)
            cur_y = layer(cur_y)
            feats_x.append(cur_x)
            feats_y.append(cur_y)

        diffs = []
        for f1, f2, lin in zip(feats_x, feats_y, self.lin_layers):
            diffs.append(lin((f1-f2)**2))
        val = torch.mean(torch.cat(diffs, dim=0))
        return val.item()


class LPIPSMetric(nn.Module):
    """Wrapper for LPIPS evaluation with .to(device) support"""
    def __init__(self, net="alex", weights_path=None, device="cpu"):
        super().__init__()
        self.model = LPIPS(backbone=net, pretrained_weights=weights_path).to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            val = self.model(x.to(next(self.model.parameters()).device),
                             y.to(next(self.model.parameters()).device))
        return float(val)
