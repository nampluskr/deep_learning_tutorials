"""Metrics specific to normalizing flow-based anomaly detection models."""

import torch
import torch.nn as nn
import numpy as np


class LogLikelihoodMetric(nn.Module):
    """Log-likelihood metric for flow-based models."""
    
    def __init__(self):
        super().__init__()

    def forward(self, log_prob):
        """Compute average log-likelihood."""
        return torch.mean(log_prob).item()


class BPDMetric(nn.Module):
    """Bits Per Dimension metric for flow-based models."""
    
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims  # (C, H, W)
        self.total_dims = np.prod(input_dims)

    def forward(self, log_prob):
        """Convert log-likelihood to bits per dimension."""
        # BPD = -log_prob / (dims * log(2))
        bpd = -log_prob / (self.total_dims * np.log(2.0))
        return torch.mean(bpd).item()


class FlowJacobianMetric(nn.Module):
    """Jacobian determinant metric for flow stability."""
    
    def __init__(self):
        super().__init__()

    def forward(self, log_jacobian_det):
        """Compute statistics of Jacobian determinant."""
        jac_det = torch.exp(log_jacobian_det)
        return {
            'mean_jac_det': torch.mean(jac_det).item(),
            'std_jac_det': torch.std(jac_det).item(),
            'min_jac_det': torch.min(jac_det).item(),
            'max_jac_det': torch.max(jac_det).item(),
        }