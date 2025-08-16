import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import sys
from tqdm import tqdm
import time
from copy import deepcopy

from mvtec import get_transforms, get_dataloaders
from autoencoder import get_model


def set_seed(seed=42):
    """디바이스 설정 및 시드 고정"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f" > Seed:   {seed}")
    print(f" > Device: {device}")
    if torch.cuda.is_available():
        print(f" > GPU:    {torch.cuda.get_device_name(0)}")


def train_model(config):
    ## Step-1. Seed
    print("\n*** Setting Seed...")
    set_seed(seed=config.seed)

    ## Step-2. Data
    print("\n*** Loading data...")
    train_transform, test_transform = get_transforms(
        img_size=config.img_size,
        normalize=config.normalize
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        category=config.category,
        batch_size=config.batch_size,
        valid_ratio=config.valid_ratio,
        train_transform=train_transform,
        test_transform=test_transform
    )

    ## Step-3. Modeling
    print("\n*** Creating model...")
    model = get_model(config).to(config.device)
    print(f" > Model: {config.model_type}")


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    metrics = {
        "l1": nn.L1Loss(),
        "bce": nn.BCELoss()
    }

    ## Step-4. Training
    print("\n*** Starting training...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode='min', factor=0.5, patience=5)

    for epoch in range(1, config.num_epochs + 1):
        train_results = train_epoch(model, train_loader, criterion, optimizer, metrics=metrics)
        train_info = ", ".join([f'{key}={value:.4f}' for key, value in train_results.items()])

        valid_results = evaluate_epoch(model, valid_loader, criterion, metrics=metrics)
        valid_info = ", ".join([f'{key}={value:.4f}' for key, value in valid_results.items()])

        print(f" > Epoch [{epoch:2d}/{config.num_epochs}] "
            f"{train_info} | (val) {valid_info}")

        scheduler.step(valid_results["loss"])

    print("\n*** Training completed.")


def train_epoch(model, data_loader, criterion, optimizer, metrics={}):
    device = next(model.parameters()).device
    model.train()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    with tqdm(data_loader, desc="Train", file=sys.stdout, ascii=True,
              leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 정상 데이터만 사용 (labels == 0)
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]

            optimizer.zero_grad()
            reconstructed, latent, features = model(normal_images)

            # 입력 이미지를 [0, 1] 범위로 정규화 (Sigmoid 출력과 매칭)
            normal_images_norm = (normal_images - normal_images.min()) / (normal_images.max() - normal_images.min() + 1e-8)
            loss = criterion(reconstructed, normal_images_norm)
            loss.backward()
            optimizer.step()

            # 메트릭 업데이트
            results["loss"] += loss.item()
            with torch.no_grad():
                for metric_name, metric_fn in metrics.items():
                    metric_value = metric_fn(reconstructed, normal_images_norm)
                    results[metric_name] += metric_value.item()

            num_batches += 1
            progress_info = {f'{key}': f'{value/num_batches:.4f}'
                             for key, value in results.items()}
            progress_bar.set_postfix(progress_info)

    return {key: value / len(data_loader) for key, value in results.items()}


@torch.no_grad()
def evaluate_epoch(model, data_loader, criterion, metrics={}):
    device = next(model.parameters()).device
    model.train()

    results = {"loss": 0.0}
    for metric_name in metrics.keys():
        results[metric_name] = 0.0

    num_batches = 0
    with tqdm(data_loader, desc="Evaluate", file=sys.stdout, ascii=True,
              leave=False) as progress_bar:
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 정상 데이터만 사용 (labels == 0)
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]
            reconstructed, latent, features = model(normal_images)

            # 입력 이미지를 [0, 1] 범위로 정규화 (Sigmoid 출력과 매칭)
            normal_images_norm = (normal_images - normal_images.min()) / (normal_images.max() - normal_images.min() + 1e-8)
            loss = criterion(reconstructed, normal_images_norm)

            results["loss"] += loss.item()
            for metric_name, metric_fn in metrics.items():
                metric_value = metric_fn(reconstructed, normal_images_norm)
                results[metric_name] += metric_value.item()

            num_batches += 1
            progress_info = {f'{key}': f'{value/num_batches:.4f}'
                             for key, value in results.items()}
            progress_bar.set_postfix(progress_info)

    return {key: value / len(data_loader) for key, value in results.items()}


if __name__ == "__main__":

    # Example usage
    print("This module is intended to be imported, not run directly.")
    # You can add code here to test the functions if needed.
    # For example, you can create a dummy model, data loader, criterion, and optimizer
    # and call train_epoch or evaluate_epoch with them.
