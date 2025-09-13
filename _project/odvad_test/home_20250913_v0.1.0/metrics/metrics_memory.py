"""Metrics specific to memory-based anomaly detection models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisDistanceMetric(nn.Module):
    """Mahalanobis distance-based anomaly score metric."""
    
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, mean, inv_covariance):
        """Compute average Mahalanobis distance for embeddings."""
        batch_size, channels, height, width = embeddings.shape
        embeddings_flat = embeddings.reshape(batch_size, channels, -1)
        
        # Compute distances for each spatial location
        distances = []
        for i in range(height * width):
            delta = embeddings_flat[:, :, i] - mean[:, i]
            dist = torch.sqrt(torch.sum(delta * torch.matmul(inv_covariance[i], delta.T).T, dim=1))
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)
        return torch.mean(distances)


class MemoryEfficiencyMetric(nn.Module):
    """Memory usage efficiency metric for memory-based models."""
    
    def __init__(self):
        super().__init__()

    def forward(self, memory_bank_size, total_samples, feature_dim):
        """Compute memory efficiency ratio."""
        memory_usage = memory_bank_size * feature_dim
        theoretical_max = total_samples * feature_dim
        efficiency = memory_usage / theoretical_max if theoretical_max > 0 else 0.0
        return efficiency
