## MVTec - Anomaly Detection Using Vanila CNN Autoencoder

### Libraries

```python
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
from time import time
import sys
from tqdm import tqdm
```


### Dataset

```python
class MVTec(Dataset):
    def __init__(self, data_dir, categories, split, img_size):
        super().__init__()

        self.image_paths = []
        self.labels = []

        for category in categories:
            category_path = os.path.join(data_dir, category, split)
            if split == "train":
                label = 0
                for path in glob(os.path.join(category_path, "good", "*.png")):
                    self.image_paths.append(path)
                    self.labels.append(label)
            elif split == "test":
                for subfolder in os.listdir(category_path):
                    label = 0 if subfolder == "good" else 1
                    for path in glob(os.path.join(category_path, subfolder, "*.png")):
                        self.image_paths.append(path)
                        self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return {"image": image, "label": label}
```


### Modeling - Encoder

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)

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
```


### Modeling - Decoder

```python
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
```


### Modeling - Autoencoder

```python
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
```


### Training

```python
def train(model, train_loader, loss_fn, optimizer):
    device = next(model.parameters()).device
    model.train()

    total_loss = 0
    with tqdm(train_loader, leave=False, file=sys.stdout,
              dynamic_ncols=True, ascii=True) as pbar:
        pbar.set_description("Training")
        for n_batches, data in enumerate(pbar):
            images = data['image'].to(device)
            labels = data['label']

            normal_mask = labels == 0
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]

            optimizer.zero_grad()
            reconstructed, _ = model(normal_images)
            loss = loss_fn(reconstructed, normal_images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss = f"{total_loss / (n_batches + 1):.4f}")

    return total_loss / len(train_loader)

def evaluate(model, test_loader, loss_fn):
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    with torch.no_grad():
        with tqdm(test_loader, leave=False, file=sys.stdout,
              dynamic_ncols=True, ascii=True) as pbar:
            pbar.set_description("Evaluation")
            for n_batches, data in enumerate(pbar):
                images = data['image'].to(device)
                labels = data['label']

                normal_mask = labels == 0
                if not normal_mask.any():
                    continue

                normal_images = images[normal_mask]
                reconstructed, _ = model(normal_images)
                loss = loss_fn(reconstructed, normal_images)

                total_loss += loss.item()
                pbar.set_postfix(loss = f"{total_loss / (n_batches + 1):.4f}")
    return total_loss / len(test_loader)
```

### main

```python
## Data Loaders
data_dir = '/home/namu/myspace/NAMU/datasets/mvtec'
categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
              'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
              'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# categories = ['grid', 'tile']

kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True, "persistent_workers": True}
train_dataset = MVTec(data_dir, categories, split="train", img_size=256)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)

test_dataset = MVTec(data_dir, categories, split="test", img_size=256)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, **kwargs)

## Modeling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VanilaEncoder(in_channels=3, latent_dim=512)
decoder = VanilaDecoder(out_channels=3, latent_dim=512)
model = AutoEncoder(encoder, decoder).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

## Training
num_epochs = 10
train_losses = []
valid_losses = []
best_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    start_time = time()

    train_loss = train(model, train_loader, loss_fn, optimizer)
    valid_loss = evaluate(model, test_loader, loss_fn)

    scheduler.step(valid_loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    epoch_time = time() - start_time

    print(f"[Epoch {epoch:2d}/{num_epochs}] "
          f"loss={train_loss:.4f}, "
          f"val_loss={valid_loss:.4f} "
          f"({epoch_time:.0f}s)")

    if valid_loss < best_loss:
        best_loss = valid_loss
        # if save_path:
        #     save_model(model, optimizer, train_losses, val_losses, save_path)
        print(f">> Best model saved! Val Loss: {valid_loss:.4f}")
```
