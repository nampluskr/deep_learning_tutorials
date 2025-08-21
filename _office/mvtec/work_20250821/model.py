import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    """Basic deconvolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv_block(x)


class VanillaEncoder(nn.Module):
    """Vanilla CNN encoder for autoencoder"""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        latent = self.fc(pooled)
        return latent, features


class VanillaDecoder(nn.Module):
    """Vanilla CNN decoder for autoencoder"""
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*8*8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        self.deconv_blocks = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent):
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VanillaAutoencoder(nn.Module):
    """Vanilla autoencoder combining encoder and decoder"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = "vanilla_ae"

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'features': features,
            'input': x
        }

    def compute_loss(self, outputs, loss_fn_dict):
        """Compute vanilla autoencoder loss"""
        losses = {}
        total_loss = 0
        
        for loss_name, loss_config in loss_fn_dict.items():
            loss_fn = loss_config['fn']
            weight = loss_config.get('weight', 1.0)
            
            if loss_name == 'reconstruction':
                loss_value = loss_fn(outputs['reconstructed'], outputs['input'])
            else:
                continue  # Skip unsupported loss types
                
            losses[loss_name] = loss_value
            total_loss += weight * loss_value
        
        losses['total'] = total_loss
        return losses


class VAEEncoder(nn.Module):
    """VAE encoder with reparameterization trick"""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        return mu, logvar, features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """VAE decoder identical to vanilla decoder"""
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*8*8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        self.deconv_blocks = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        reconstructed = self.deconv_blocks(x)
        return reconstructed


class VAE(nn.Module):
    """Variational autoencoder with KL divergence loss"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = "vae"

    def forward(self, x):
        mu, logvar, features = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'features': features,
            'input': x
        }

    def compute_loss(self, outputs, loss_fn_dict):
        """Compute VAE loss including KL divergence"""
        losses = {}
        total_loss = 0
        
        for loss_name, loss_config in loss_fn_dict.items():
            loss_fn = loss_config['fn']
            weight = loss_config.get('weight', 1.0)
            
            if loss_name == 'reconstruction':
                loss_value = loss_fn(outputs['reconstructed'], outputs['input'])
            elif loss_name == 'kl_divergence':
                loss_value = self._kl_loss(outputs['mu'], outputs['logvar'])
            else:
                continue  # Skip unsupported loss types
                
            losses[loss_name] = loss_value
            total_loss += weight * loss_value
        
        losses['total'] = total_loss
        return losses

    def _kl_loss(self, mu, logvar):
        """Compute KL divergence loss"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class MemoryBankModel(nn.Module):
    """Example memory bank model for future expansion"""
    def __init__(self, feature_extractor, memory_bank_size=1000):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.memory_bank_size = memory_bank_size
        self.model_type = "memory_bank"
        
        # Initialize memory bank (will be filled during training)
        self.register_buffer('memory_bank', torch.empty(0))
        self.memory_filled = False

    def forward(self, x):
        features = self.feature_extractor(x)
        
        return {
            'features': features,
            'input': x,
            'memory_bank': self.memory_bank if self.memory_filled else None
        }

    def update_memory_bank(self, features):
        """Update memory bank with new features"""
        if not self.memory_filled:
            if self.memory_bank.size(0) < self.memory_bank_size:
                self.memory_bank = torch.cat([self.memory_bank, features.detach()], dim=0)
            else:
                self.memory_filled = True

    def compute_loss(self, outputs, loss_fn_dict):
        """Compute memory bank based loss"""
        losses = {}
        total_loss = 0
        
        # Update memory bank during training
        if self.training and 'features' in outputs:
            self.update_memory_bank(outputs['features'])
        
        for loss_name, loss_config in loss_fn_dict.items():
            loss_fn = loss_config['fn']
            weight = loss_config.get('weight', 1.0)
            
            if loss_name == 'memory_distance' and self.memory_filled:
                # Compute distance to memory bank
                loss_value = loss_fn(outputs['features'], outputs['memory_bank'])
            else:
                continue  # Skip unsupported loss types
                
            losses[loss_name] = loss_value
            total_loss += weight * loss_value
        
        losses['total'] = total_loss
        return losses


# Base class for future anomaly detection models
class BaseAnomalyModel(nn.Module):
    """Base class for all anomaly detection models"""
    def __init__(self):
        super().__init__()
        self.model_type = "base"

    def forward(self, x):
        """Should return a dictionary with model outputs"""
        raise NotImplementedError

    def compute_loss(self, outputs, loss_fn_dict):
        """Should compute loss from outputs and loss function dictionary"""
        raise NotImplementedError

    def compute_anomaly_score(self, outputs, score_type='default'):
        """Compute anomaly score from model outputs"""
        if score_type == 'reconstruction' and 'reconstructed' in outputs:
            return torch.mean((outputs['input'] - outputs['reconstructed']) ** 2, dim=[1, 2, 3])
        else:
            raise NotImplementedError(f"Score type {score_type} not supported for {self.model_type}")


# Loss function classes for different paradigms
class ReconstructionLoss:
    """Reconstruction loss for autoencoder-based models"""
    def __init__(self, loss_type='mse'):
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def __call__(self, pred, target):
        return self.loss_fn(pred, target)


class KLDivergenceLoss:
    """KL divergence loss for VAE"""
    def __init__(self):
        pass

    def __call__(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class MemoryDistanceLoss:
    """Distance loss for memory bank based models"""
    def __init__(self, distance_type='l2'):
        self.distance_type = distance_type

    def __call__(self, features, memory_bank):
        if memory_bank is None or memory_bank.size(0) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Compute minimum distance to memory bank
        distances = torch.cdist(features, memory_bank, p=2)
        min_distances = torch.min(distances, dim=1)[0]
        return torch.mean(min_distances)
