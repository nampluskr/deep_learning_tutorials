import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from time import time
from abc import ABC, abstractmethod

from trainer import BaseTrainer


#############################################################
# Trainer for STFPMV2 Model
#############################################################

class STFPMV2Trainer(BaseTrainer):
    """Trainer for STFPMV2 model with Manual model's techniques."""
    
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None,
                 use_amp=True, pretrain_penalty=False):
        from model_stfpmv2 import STFPMV2Loss, STFPMV2Metric
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.pretrain_penalty = pretrain_penalty
        
        # Only optimize student extractor and student adapter parameters
        student_params = []
        student_params.extend(self.model.student_extractor.parameters())
        student_params.extend(self.model.student_adapter.parameters())
        
        self.optimizer = optimizer or optim.AdamW(student_params, lr=1e-4, weight_decay=1e-5)
        
        # Add cosine annealing scheduler (Manual model's technique)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6  # Assuming 100 epochs
        )
        
        self.loss_fn = loss_fn or STFPMV2Loss(pretrain_penalty=pretrain_penalty)
        self.metrics = metrics or {'similarity': STFPMV2Metric()}
        
        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
    def run_epoch(self, loader, mode='train', desc=""):
        """Run one epoch with Manual model's training techniques."""
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0
        
        # Set model mode
        if mode == 'train':
            self.model.student_extractor.train()
            self.model.student_adapter.train()
            self.model.teacher_extractor.eval()
            self.model.teacher_adapter.eval()
        else:
            self.model.eval()
        
        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                else:
                    images = batch[0].to(self.device)
                
                batch_size = images.size(0)
                num_images += batch_size
                
                # Forward pass with mixed precision
                if mode == 'train' and self.use_amp:
                    with torch.cuda.amp.autocast():
                        teacher_features, student_features = self.model(images)
                        loss = self.loss_fn(teacher_features, student_features,
                                          model=self.model if self.pretrain_penalty else None)
                else:
                    teacher_features, student_features = self.model(images)
                    loss = self.loss_fn(teacher_features, student_features,
                                      model=self.model if self.pretrain_penalty else None)
                
                if mode == 'train':
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        # Gradient clipping (Manual model's technique)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.student_extractor.parameters()) + 
                            list(self.model.student_adapter.parameters()), 
                            max_norm=5.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.student_extractor.parameters()) + 
                            list(self.model.student_adapter.parameters()), 
                            max_norm=5.0
                        )
                        self.optimizer.step()
                
                # Update loss
                results["loss"] += loss.item() * batch_size
                
                # Compute metrics
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        metric_value = metric_fn(teacher_features, student_features)
                        results[name] += metric_value.item() * batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    **{n: f"{v/num_images:.3f}" for n, v in results.items()}
                })
        
        # Step scheduler after each epoch
        if mode == 'train':
            self.scheduler.step()
            
        return {name: value / num_images for name, value in results.items()}
    
    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):
        return self.test_feature_based(test_loader, output_dir, show_image, img_prefix,
            skip_normal, skip_anomaly, num_max)


#############################################################
# Trainer for EfficientADV2 Model
#############################################################

class EfficientADV2Trainer(BaseTrainer):
    """Trainer for EfficientADV2 model with Manual model's techniques."""
    
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None,
                 use_amp=True, pretrain_penalty=False):
        from model_efficientadv2 import EfficientADV2Loss, EfficientADV2Metric
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.pretrain_penalty = pretrain_penalty
        
        # Only optimize student encoder, student adapter, and decoder parameters
        trainable_params = []
        trainable_params.extend(self.model.student_net.parameters())
        trainable_params.extend(self.model.student_adapter.parameters())
        trainable_params.extend(self.model.decoder.parameters())
        
        self.optimizer = optimizer or optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        
        # Add cosine annealing scheduler (Manual model's technique)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6  # Assuming 100 epochs
        )
        
        self.loss_fn = loss_fn or EfficientADV2Loss(pretrain_penalty=pretrain_penalty)
        self.metrics = metrics or {'similarity': EfficientADV2Metric()}
        
        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
    def run_epoch(self, loader, mode='train', desc=""):
        """Run one epoch with Manual model's training techniques."""
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0
        
        # Set model mode
        if mode == 'train':
            self.model.student_net.train()
            self.model.student_adapter.train()
            self.model.decoder.train()
            self.model.teacher_net.eval()
            self.model.teacher_adapter.eval()
        else:
            self.model.eval()
        
        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                else:
                    images = batch[0].to(self.device)
                
                batch_size = images.size(0)
                num_images += batch_size
                
                # Forward pass with mixed precision
                if mode == 'train' and self.use_amp:
                    with torch.cuda.amp.autocast():
                        reconstructed, teacher_features, student_features = self.model(images)
                        loss = self.loss_fn(reconstructed, images, teacher_features, student_features,
                                          model=self.model if self.pretrain_penalty else None)
                else:
                    reconstructed, teacher_features, student_features = self.model(images)
                    loss = self.loss_fn(reconstructed, images, teacher_features, student_features,
                                      model=self.model if self.pretrain_penalty else None)
                
                if mode == 'train':
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        # Gradient clipping (Manual model's technique)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.student_net.parameters()) + 
                            list(self.model.student_adapter.parameters()) +
                            list(self.model.decoder.parameters()), 
                            max_norm=5.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.student_net.parameters()) + 
                            list(self.model.student_adapter.parameters()) +
                            list(self.model.decoder.parameters()), 
                            max_norm=5.0
                        )
                        self.optimizer.step()
                
                # Update loss
                results["loss"] += loss.item() * batch_size
                
                # Compute metrics
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        metric_value = metric_fn(reconstructed, images, teacher_features, student_features)
                        results[name] += metric_value.item() * batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    **{n: f"{v/num_images:.3f}" for n, v in results.items()}
                })
        
        # Step scheduler after each epoch
        if mode == 'train':
            self.scheduler.step()
            
        return {name: value / num_images for name, value in results.items()}
    
    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):
        return self.test_feature_based(test_loader, output_dir, show_image, img_prefix,
            skip_normal, skip_anomaly, num_max)
