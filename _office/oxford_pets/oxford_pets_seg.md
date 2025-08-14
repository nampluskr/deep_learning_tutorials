```python
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
```

```python
class OxfordPets(Dataset):
    """ Dataset for Segmentation """
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if split == 'train':
            split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
        elif split == 'test':
            split_file = os.path.join(data_dir, 'annotations', 'test.txt')

        # https://github.com/tensorflow/models/issues/3134
        images_png = [
            "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
            "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
            "Abyssinian_5", "Abyssinian_34",
        ]
        images_corrupt = ["chihuahua_121", "beagle_116"]

        self.samples = []
        with open(split_file) as file:
            for line in file:
                filename, *_ = line.strip().split()
                if filename not in images_corrupt + images_png:
                    image_path = os.path.join(self.data_dir, 'images', filename + ".jpg")
                    mask_path = os.path.join(self.data_dir, 'annotations', 'trimaps', filename + ".png")
                    self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # (H, W, 3)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # (H, W)
        mask -= 1    # 0: background 1: pet, 2: border
        # mask[mask ==  2] = 255

        if self.transform:
            transforemed = self.transform(image=image, mask=mask)
            image = transforemed["image"]
            mask = transforemed["mask"].long()

        return image, mask
```

```python
train_transform = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
valid_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
train_dataset = OxfordPets(data_dir, split="train", transform=train_transform)
valid_dataset = OxfordPets(data_dir, split="test", transform=valid_transform)

kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True, "persistent_workers": True}
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, **kwargs)
```

```python
from torchvision.models.segmentation import fcn_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 3

model = fcn_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=1)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

```python
n_epochs = 10
for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        print(preds)
        break

        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch:2d}/{n_epochs}] laoss={avg_loss:.3f}")
```
