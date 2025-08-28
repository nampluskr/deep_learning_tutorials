import os
from glob import glob
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


def get_transforms(img_size=256):
    train_transform = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.ToDtype(torch.float32, scale=True),
    ])
    test_transform = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, test_transform


def get_dataloader(name, **params):
    available_list = {
        'mvtec': MVTecDataloader,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {'num_workers': 8, 'pin_memory': True}
    default_params.update(params)
    return selected(model.parameters(), **default_params)


# =============================================================================
# MVTec Dataset
# =============================================================================

class MVTecDataset(Dataset):
    def __init__(self, data_dir, categories, split, transform=None, normal_only=False):
        super().__init__()
        self.transform = transform
        self.normal_only = normal_only
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
                    if self.normal_only and label == 1:
                        continue
                    for path in glob(os.path.join(category_path, subfolder, "*.png")):
                        self.image_paths.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = read_image(path, mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return {"image": image, "label": label}


class MVTecDataloader:
    def __init__(self, data_dir, categories, train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16, valid_ratio=0.2, seed=42, **params):
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.params = params

        # datasets
        train_dataset = MVTecDataset(self.data_dir, categories, 'train', train_transform)
        valid_dataset = MVTecDataset(self.data_dir, categories, 'train', test_transform)

        total_size = len(train_dataset)
        valid_size = int(valid_ratio * total_size)
        train_size = total_size - valid_size

        torch.manual_seed(seed)
        train_subset, valid_subset = random_split(
            range(total_size), [train_size, valid_size],
            generator=torch.Generator().manual_seed(seed)
        )
        self.train_dataset = Subset(train_dataset, train_subset.indices)
        self.valid_dataset = Subset(valid_dataset, valid_subset.indices)
        self.test_dataset = MVTecDataset(self.data_dir, categories, 'test', test_transform)

    def train_loader(self):
        return DataLoader(self.train_dataset, self.train_batch_size, 
            shuffle=True, drop_last=True, **self.params)

    def valid_loader(self):
        return DataLoader(self.valid_dataset, self.test_batch_size, 
            shuffle=False, drop_last=False, **self.params)

    def test_loader(self):
        return DataLoader(self.test_dataset, self.test_batch_size, 
            shuffle=False, drop_last=False, **self.params)
