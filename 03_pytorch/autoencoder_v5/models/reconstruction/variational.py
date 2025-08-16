"""
Variational autoencoder models for anomaly detection.

This module implements variational autoencoders (VAE) and its variants
that use probabilistic latent representations for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Tuple, Dict, Any

from ..base import ConvBlock, DeconvBlock
from ..base.utils import (
    get_final_conv_size,
    get_output_padding_for_target_size,
    validate_input_size
)


class VAEEncoder(nn.Module):
    """Variational encoder with reparameterization trick"""
    
    def __init__(self, in_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Validate input size
        validate_input_size(input_size, min_size=32)

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),      
            ConvBlock(32, 64),               
            ConvBlock(64, 128),              
            ConvBlock(128, 256),             
            ConvBlock(256, 512),             
        )

        # Adaptive pooling for consistent feature size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Separate heads for mean and log variance
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backpropagation through sampling"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.conv_blocks(x)
        pooled = self.adaptive_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar, features


class VAEDecoder(nn.Module):
    """Variational decoder for reconstructing from latent samples"""
    
    def __init__(self, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Validate input size
        validate_input_size(input_size, min_size=32)

        # Start from fixed 4x4 feature map
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (512, 4, 4))

        # Calculate target sizes for reconstruction
        target_sizes = []
        size = input_size
        for _ in range(5):
            size = size // 2
            target_sizes.append(size)
        target_sizes.reverse()

        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(512, 256, output_padding=self._get_output_padding(4, target_sizes[0])),     
            DeconvBlock(256, 128, output_padding=self._get_output_padding(target_sizes[0], target_sizes[1])),  
            DeconvBlock(128, 64, output_padding=self._get_output_padding(target_sizes[1], target_sizes[2])),   
            DeconvBlock(64, 32, output_padding=self._get_output_padding(target_sizes[2], target_sizes[3])),    
        ])
        
        # Final layer
        final_output_padding = self._get_output_padding(target_sizes[3], input_size)
        self.final_conv = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, 
                                           padding=1, output_padding=final_output_padding)
        self.final_activation = nn.Sigmoid()

    def _get_output_padding(self, input_size, target_size):
        """Calculate output padding to reach target size"""
        return get_output_padding_for_target_size(input_size, target_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = self.unflatten(x)
        
        for deconv_block in self.deconv_blocks:
            x = deconv_block(x)
            
        x = self.final_conv(x)
        reconstructed = self.final_activation(x)
        return reconstructed


class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(in_channels, latent_dim, input_size)
        self.decoder = VAEDecoder(out_channels, latent_dim, input_size)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Validate input size
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        z, mu, logvar, features = self.encoder(x)
        reconstructed = self.decoder(z)
        
        return reconstructed, z, mu, logvar

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples from prior distribution"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor, 
                    beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute VAE loss components"""
        
        # Reconstruction loss (BCE or MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with beta weighting
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class BetaVAE(VAE):
    """β-VAE with adjustable beta parameter for disentanglement"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, 
                 input_size=256, beta=4.0):
        super().__init__(in_channels, out_channels, latent_dim, input_size)
        self.beta = beta

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward(x)

    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute β-VAE loss with fixed beta"""
        return super().compute_loss(x, reconstructed, mu, logvar, beta=self.beta)


class WAE(nn.Module):
    """Wasserstein Autoencoder for anomaly detection"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, 
                 input_size=256, lambda_reg=10.0):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        
        self.encoder = VAEEncoder(in_channels, latent_dim, input_size)
        self.decoder = VAEDecoder(out_channels, latent_dim, input_size)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Validate input size
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        z, mu, logvar, features = self.encoder(x)
        reconstructed = self.decoder(z)
        
        # For WAE, we use the mean (deterministic encoding)
        return reconstructed, mu

    def compute_mmd_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy loss"""
        # Sample from prior (standard normal)
        z_prior = torch.randn_like(z)
        
        # Compute MMD between z and z_prior using RBF kernel
        def compute_kernel(x, y, sigma=1.0):
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
            
            x = x.unsqueeze(1)  # (x_size, 1, dim)
            y = y.unsqueeze(0)  # (1, y_size, dim)
            
            tiled_x = x.expand(x_size, y_size, dim)
            tiled_y = y.expand(x_size, y_size, dim)
            
            kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
            return torch.exp(-kernel_input / sigma)
        
        x_kernel = compute_kernel(z, z)
        y_kernel = compute_kernel(z_prior, z_prior)
        xy_kernel = compute_kernel(z, z_prior)
        
        mmd_loss = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd_loss

    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor, 
                    z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute WAE loss components"""
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # MMD regularization loss
        mmd_loss = self.compute_mmd_loss(z)
        
        # Total loss
        total_loss = recon_loss + self.lambda_reg * mmd_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'mmd_loss': mmd_loss
        }


class ConditionalVAE(VAE):
    """Conditional VAE with class conditioning"""
    
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512, 
                 input_size=256, num_classes=10):
        super().__init__(in_channels, out_channels, latent_dim, input_size)
        self.num_classes = num_classes
        
        # Modify encoder and decoder for conditioning
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Modify encoder FC layers
        self.encoder.fc_mu = nn.Linear(512 * 4 * 4 + latent_dim, latent_dim)
        self.encoder.fc_logvar = nn.Linear(512 * 4 * 4 + latent_dim, latent_dim)
        
        # Modify decoder FC layer
        self.decoder.fc = nn.Linear(latent_dim + latent_dim, 512 * 4 * 4)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode with conditioning
        features = self.encoder.conv_blocks(x)
        pooled = self.encoder.adaptive_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Add label embedding
        label_emb = self.label_embedding(labels)
        pooled_cond = torch.cat([pooled, label_emb], dim=1)
        
        mu = self.encoder.fc_mu(pooled_cond)
        logvar = self.encoder.fc_logvar(pooled_cond)
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode with conditioning
        z_cond = torch.cat([z, label_emb], dim=1)
        reconstructed = self.decoder(z_cond)
        
        return reconstructed, z, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between encoded distribution and standard normal"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor, 
                       loss_type: str = 'mse') -> torch.Tensor:
    """Compute reconstruction loss"""
    if loss_type == 'mse':
        return F.mse_loss(x_recon, x, reduction='sum')
    elif loss_type == 'bce':
        return F.binary_cross_entropy(x_recon, x, reduction='sum')
    elif loss_type == 'l1':
        return F.l1_loss(x_recon, x, reduction='sum')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test variational autoencoder models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    input_size = 256
    x = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    print("Testing VAE:")
    vae = VAE(latent_dim=256, input_size=input_size).to(device)
    reconstructed, z, mu, logvar = vae(x)
    loss_dict = vae.compute_loss(x, reconstructed, mu, logvar)
    print(f"Input: {x.shape}")
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Latent: {z.shape}")
    print(f"Loss components: {list(loss_dict.keys())}")
    
    print("\nTesting β-VAE:")
    beta_vae = BetaVAE(latent_dim=256, input_size=input_size, beta=4.0).to(device)
    reconstructed, z, mu, logvar = beta_vae(x)
    loss_dict = beta_vae.compute_loss(x, reconstructed, mu, logvar)
    print(f"β-VAE loss components: {list(loss_dict.keys())}")
    
    print("\nTesting WAE:")
    wae = WAE(latent_dim=256, input_size=input_size).to(device)
    reconstructed, z = wae(x)
    loss_dict = wae.compute_loss(x, reconstructed, z)
    print(f"WAE loss components: {list(loss_dict.keys())}")
    
    print("\nTesting ConditionalVAE:")
    cvae = ConditionalVAE(latent_dim=256, input_size=input_size, num_classes=10).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    reconstructed, z, mu, logvar = cvae(x, labels)
    print(f"Conditional VAE output: {reconstructed.shape}")
    
    print("\nAll variational models working correctly!")