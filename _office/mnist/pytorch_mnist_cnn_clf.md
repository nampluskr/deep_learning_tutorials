## MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Data

```python
import numpy as np
import os
import gzip

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

class MNIST(Dataset):
    def __init__(self, data_dir, train=True):
        if train:
            self.images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
            self.labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
            self.transform = transforms.Compose([
                Preprocess(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ])
        else:
            self.images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
            self.labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
            self.transform = Preprocess()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()
        label = torch.tensor(self.labels[idx]).long()

        if self.transform:
            image = self.transform(image)
        return image, label

class Preprocess:
    def __call__(self, image):
        image = torch.unsqueeze(image, dim=0) / 255.0
        return image    ## (N, 1, 28, 28)
```

### Modeling

```python
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
    
class Accuracy:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=1)
        return torch.eq(y_pred, y_true).float().mean()
```

### Training - 1

```python
data_dir = r"D:\Non_Documents\2025\datasets\fashion_mnist"
train_data = MNIST(data_dir, train=True)
valid_data = MNIST(data_dir, train=False)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, x.dtype, y.shape, y.dtype)

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
metric = Accuracy()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
for epoch in range(1, n_epochs + 1):

    ## Training
    model.train()
    train_loss, train_acc = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += metric(y_pred, y).item()
        
    ## Validattion        
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            valid_loss += criterion(y_pred, y).item()
            valid_acc += metric(y_pred, y).item()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:2d}/{n_epochs}] "
              f"loss: {train_loss/len(train_loader):.3f} "
              f"acc: {train_acc/len(train_loader):.3f} | "
              f"val_loss: {valid_loss/len(valid_loader):.3f} "
              f"val_acc: {valid_acc/len(valid_loader):.3f}")
```

### Training - 2

```python
import sys
from tqdm import tqdm

def train(model, loader, optimizer, criterion, metric):
    device = next(model.parameters()).device
    model.train()
    batch_loss, batch_acc = 0, 0
    with tqdm(loader, leave=False, file=sys.stdout,
              dynamic_ncols=True, ascii=True) as pbar:
        for x, y in pbar:
            pbar.set_description("Training")
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_acc += metric(y_pred, y).item()

    return {"loss": batch_loss/len(loader), "acc": batch_acc/len(loader)}

def evaluate(model, loader, criterion, metric):
    device = next(model.parameters()).device
    model.eval()
    batch_loss, batch_acc = 0, 0
    with torch.no_grad():
        with tqdm(loader, leave=False, file=sys.stdout,
                  dynamic_ncols=True, ascii=True) as pbar:
            for x, y in loader:
                pbar.set_description("Evaluation")
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                batch_loss += criterion(y_pred, y).item()
                batch_acc += metric(y_pred, y).item()

    return {"loss": batch_loss/len(loader), "acc": batch_acc/len(loader)}
```

```python
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
metric = Accuracy()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
for epoch in range(1, n_epochs + 1):

    train_res = train(model, train_loader, optimizer, criterion, metric)
    valid_res = evaluate(model, valid_loader, criterion, metric)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:2d}/{n_epochs}] "
              f"loss: {train_res['loss']:.3f} acc: {train_res['acc']:.3f} | "
              f"val_loss: {valid_res['loss']:.3f} val_acc: {valid_res['acc']:.3f}")
```

```
[  1/10] loss: 0.523 acc: 0.809 | val_loss: 0.410 val_acc: 0.856
[  2/10] loss: 0.345 acc: 0.875 | val_loss: 0.355 val_acc: 0.872
[  3/10] loss: 0.300 acc: 0.890 | val_loss: 0.299 val_acc: 0.894
[  4/10] loss: 0.270 acc: 0.901 | val_loss: 0.289 val_acc: 0.892
[  5/10] loss: 0.251 acc: 0.908 | val_loss: 0.302 val_acc: 0.891
[  6/10] loss: 0.235 acc: 0.913 | val_loss: 0.275 val_acc: 0.903
[  7/10] loss: 0.222 acc: 0.917 | val_loss: 0.260 val_acc: 0.906
[  8/10] loss: 0.210 acc: 0.923 | val_loss: 0.266 val_acc: 0.905
[  9/10] loss: 0.200 acc: 0.927 | val_loss: 0.283 val_acc: 0.898
[ 10/10] loss: 0.192 acc: 0.929 | val_loss: 0.253 val_acc: 0.914
```
