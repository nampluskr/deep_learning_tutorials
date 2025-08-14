```python
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T

import cv2
import sys
import random
from time import time
from tqdm import tqdm


from pytorch_msssim import ssim


## Dataset
class MVTec(Dataset):
    def __init__(self, data_dir, categories, split, transform=None):
        super().__init__()

        self.transform = transform
        self.image_paths = []
        self.labels = []

        for category in categories:
            category_path = os.path.join(data_dir, category, split)
            if split == "train":
                label = 0
                for path in glob(os.path.join(category_path, "good", "*.png")):
                    self.image_paths.append(path)
                    self.labels.append(label)
            else:
                for subfolder in os.listdir(category_path):
                    label = 0 if subfolder == "good" else 1
                    for path in glob(os.path.join(category_path, subfolder, "*.png")):
                        self.image_paths.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()
        # return {"image": image, "label": label, "path": path}
        return {"image": image, "label": label}


## Modeling
class ConvBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv_block(x)


class VanilaEncoder(nn.Module):
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


class VanilaDecoder(nn.Module):
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
        x = self.fc(latent)                     # (B, 512 * 8 * 8)
        x = self.unflatten(x)                   # (B, 512, 8, 8)
        reconstructed = self.deconv_blocks(x)
        return reconstructed

class VanilaAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features


## Trainer
def train(model, data_loader, loss_fn, optimizer, metrics={}):
    device = next(model.parameters()).device
    model.train()

    functions = {"loss": loss_fn}
    functions.update(metrics)
    results = {name: 0.0 for name in functions.keys()}

    with tqdm(data_loader, desc="Training", leave=False, file=sys.stdout,
              dynamic_ncols=True,
              ncols=100, ascii=True) as pbar:
        for cnt, data in enumerate(pbar):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            normal_mask = labels == 0
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]

            optimizer.zero_grad()
            pred, *_ = model(normal_images)
            loss = loss_fn(pred, normal_images)
            loss.backward()
            optimizer.step()

            results["loss"] += loss.item()
            for name, func in functions.items():
                if name != "loss":
                    results[name] += func(pred, normal_images).item()

            pbar.set_postfix({k: f"{v/(cnt + 1):.3f}" for k, v in results.items()})

    return {k: v/len(data_loader) for k, v in results.items()}


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, metrics=None):
    device = next(model.parameters()).device
    model.eval()

    functions = {"loss": loss_fn}
    functions.update(metrics)
    results = {name: 0.0 for name in functions.keys()}

    with tqdm(data_loader, desc="Evaluation", leave=False, file=sys.stdout,
            dynamic_ncols=True,
            ncols=100, ascii=True) as pbar:
        for cnt, data in enumerate(pbar):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            normal_mask = labels == 0
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]
            pred, *_ = model(normal_images)
            loss = loss_fn(pred, normal_images)

            results["loss"] += loss.item()
            for name, func in functions.items():
                if name != "loss":
                    results[name] += func(pred, normal_images).item()

            pbar.set_postfix({k: f"{v/(cnt + 1):.3f}" for k, v in results.items()})

    return {k: v/len(data_loader) for k, v in results.items()}


def split_train_valid(dataset, valid_ratio, seed=42):
    data_size = len(dataset)
    valid_size = int(data_size * valid_ratio)
    train_size = data_size - valid_size

    torch.manual_seed(seed)
    train_subset, valid_subset = random_split(dataset, [train_size, valid_size])
    return train_subset.indices, valid_subset.indices

def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.5 * (1 - ssim(pred, target)) + 0.5 * bce(pred, target)

def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()

def psnr(pred, target):
    mse = nn.MSELoss()(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 ** 2 / mse)


if __name__ == "__main__":

    ## Hyperparameters
    img_size = 256
    learning_rate = 1e-4
    num_epochs = 10


    ## Augmentations
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
    ])

    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])


    ## Data Loaders
    data_dir = '/home/namu/myspace/NAMU/datasets/mvtec'
    categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                  'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                  'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # categories = ['bottle', 'grid', 'tile']

    train_dataset = MVTec(data_dir, categories, split="train", transform=train_transform)
    valid_dataset = MVTec(data_dir, categories, split="train", transform=test_transform)
    train_indices, valid_indices = split_train_valid(train_dataset, valid_ratio=0.2)

    train_dataset = Subset(train_dataset, train_indices)
    valid_dataset = Subset(valid_dataset, valid_indices)
    test_dataset  = MVTec(data_dir, categories, split="test", transform=test_transform)

    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True, "persistent_workers": True}
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)
    valid_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, **kwargs)

    ## Modeling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = VanilaEncoder(in_channels=3, latent_dim=512)
    decoder = VanilaDecoder(out_channels=3, latent_dim=512)
    model = VanilaAutoEncoder(encoder, decoder).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    metrics = {"mse": nn.MSELoss(),
               "bce": nn.BCELoss(),
               "acc": binary_accuracy,
               "ssim": ssim,
               "psnr": psnr}


    ## Training Loop
    history = {"loss": []}
    history.update({name: [] for name in metrics.keys()})
    history.update({f"val_{name}": [] for name in history.keys()})

    for epoch in range(1, num_epochs + 1):
        start_time = time()

        ## Training
        train_results = train(model, train_loader, loss_fn, optimizer, metrics=metrics)
        train_desc = ', '.join([f"{k}={v:.3f}" for k, v in train_results.items()])

        for name, value in train_results.items():
            history[name].append(value)

        ## Validation
        valid_results = evaluate(model, valid_loader, loss_fn, metrics=metrics)
        valid_desc = ', '.join([f"val_{k}={v:.3f}" for k, v in valid_results.items()])

        for name, value in valid_results.items():
            history[f"val_{name}"].append(value)

        epoch_time = time() - start_time
        print(f"[Epoch {epoch:2d}/{num_epochs}] {train_desc} | {valid_desc} ({epoch_time:.0f}s)")


    ## Evaluation
    test_results = evaluate(model, test_loader, loss_fn, metrics=metrics)
    test_desc = ', '.join([f"test_{k}={v:.3f}" for k, v in test_results.items()])
    print(f">> Test: {test_desc}")
```
