## Dataloaders

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_paths = []
        self.labels    = []   # 0 = normal, 1 = anomaly

        if split == 'train':
            good_dir = os.path.join(self.root_dir, 'train', 'good')
            for fname in os.listdir(good_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(good_dir, fname)
                    self.img_paths.append(full_path)
                    self.labels.append(0)
        else:
            test_dir = os.path.join(self.root_dir, 'test')
            for sub_name in os.listdir(test_dir):
                sub_path = os.path.join(test_dir, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                for fname in os.listdir(sub_path):
                    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    full_path = os.path.join(sub_path, fname)
                    self.img_paths.append(full_path)
                    self.labels.append(0 if sub_name == 'good' else 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx]).long()
        return img, label

def denormalize(tensor):
    device = tensor.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(-1, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)

def get_transforms(img_size):
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomRotation(degrees=15),
        # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, test_transform

def get_dataloaders(root, category, batch_size, img_size=256):
    data_root = os.path.join(root, category)
    train_transform, test_transform = get_transforms(img_size=img_size)

    train_set = MVTecDataset(root_dir=data_root, split='train', transform=train_transform)
    test_set = MVTecDataset(root_dir=data_root, split='test', transform=test_transform)

    dataloader_params = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dataloader_params)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **dataloader_params)
    return train_loader, test_loader
```

## Modeling

```python
import torch.nn as nn
import torch.nn.functional as F

class VanillaAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)            # (64, H/2, W/2)
        self.enc2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)          # (128, H/4, W/4)
        self.enc3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)         # (256, H/8, W/8)
        self.enc4 = nn.Conv2d(256, latent_dim, 4, stride=2, padding=1)  # (latent, H/16, W/16)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        latent = self.relu(self.enc4(x))

        # Decoder
        x = self.relu(self.dec1(latent))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec3(x))
        recon = self.dec4(x)
        # recon = torch.sigmoid(self.dec4(x))   # [0,1]
        return recon
```

### Training

```python
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

def train(model, optimizer, loss_fn, train_loader, device, num_epochs=50):
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            recon = model(imgs)
            loss = loss_fn(recon, imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {epoch_loss:.6f}")

    return model

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []

    for imgs, lbl in loader:
        imgs = imgs.to(device)
        recon = model(imgs)

        mse = F.mse_loss(recon, imgs, reduction="none")
        mse = mse.view(mse.size(0), -1).mean(dim=1)   # (B,)

        all_scores.extend(mse.cpu().numpy().tolist())
        all_labels.extend(lbl.cpu().numpy().tolist())

    return np.array(all_scores), np.array(all_labels)

def test(scores, labels):
    auc = roc_auc_score(labels, scores)
    fpr, tpr, threshold = roc_curve(labels, scores)

    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = threshold[best_idx]

    print(f"ROC-AUC : {auc:.4f}")
    print(f"Best threshold : {best_threshold:.4f}")
    return best_threshold, auc

def show_results(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    print("\nConfusion Matrix")
    print(confusion_matrix(labels, preds))
    print("\nClassification Report")
    print(classification_report(labels, preds, target_names=["normal", "anomaly"]))
```

### Evaluation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_histogram(scores, labels, bins=51):
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    binary_labels = (labels != 0).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    sns.histplot(scores, bins=bins, kde=True,
        color="steelblue", edgecolor="black", ax=ax1,)
    ax1.set_title("Score Distribution (All)")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")

    sns.histplot(scores[binary_labels == 0], bins=bins, kde=True,
        color="green", label="Normal (0)", stat="count",
        alpha=0.6, edgecolor="black", ax=ax2,)
    sns.histplot(scores[binary_labels == 1], bins=bins, kde=True,
        color="red", label="Anomaly (1)", stat="count",
        alpha=0.6, edgecolor="black", ax=ax2,)
    ax2.set_title("Score Distribution by Label")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

### main

```python
data_dir="/home/namu/myspace/NAMU/datasets/mvtec"
category = "bottle"

train_loader, test_loader = get_dataloaders(data_dir, category=category, batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VanillaAE().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()

model_trained = train(model, optimizer, loss_fn, train_loader, device, num_epochs=50)

scores, labels = predict(model_trained, test_loader, device)
threshold, _ = test(scores, labels)
show_results(scores, labels, threshold)
show_histogram(scores, labels)
```
