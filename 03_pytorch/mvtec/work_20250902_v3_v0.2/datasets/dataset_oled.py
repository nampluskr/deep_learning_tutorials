import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.io import read_image, ImageReadMode


class OLEDDataset(Dataset):
    """OLED custom dataset for anomaly detection"""

    def __init__(self, data_dir, categories, split, transform=None, normal_only=False, **kwargs):
        super().__init__()
        self.transform = transform
        self.normal_only = normal_only
        self.image_paths = []
        self.labels = []

        # OLED specific settings (customizable)
        self.extensions = kwargs.get('extensions', ['*.png', '*.jpg', '*.jpeg', '*.bmp'])
        self.normal_folder = kwargs.get('normal_folder', 'normal')

        for category in categories:
            category_path = os.path.join(data_dir, category, split)
            if not os.path.exists(category_path):
                continue

            if split == "train":
                # Train: only normal samples
                normal_path = os.path.join(category_path, self.normal_folder)
                self._add_files(normal_path, label=0)
            else:
                # Test: all subfolders
                if os.path.exists(category_path):
                    for subfolder in os.listdir(category_path):
                        subfolder_path = os.path.join(category_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            label = 0 if subfolder == self.normal_folder else 1
                            if self.normal_only and label == 1:
                                continue
                            self._add_files(subfolder_path, label=label)

    def _add_files(self, dir_path, label):
        """Add files from directory with given label"""
        if not os.path.exists(dir_path):
            return

        for ext in self.extensions:
            paths = sorted(glob(os.path.join(dir_path, ext)))
            for path in paths:
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


class OLEDDataloader:
    """Common dataloader factory for all anomaly detection datasets"""

    def __init__(self, data_dir, categories,
                 train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16,
                 valid_ratio=0.2, seed=42, **params):

        self.categories = categories
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.params = params

        # Create datasets
        train_dataset = OLEDDataset(data_dir, categories, 'train', train_transform, **params)

        # Handle validation split
        if valid_ratio > 0.0:
            total_size = len(train_dataset)
            valid_size = int(valid_ratio * total_size)
            train_size = total_size - valid_size

            torch.manual_seed(seed)
            train_subset, valid_subset = random_split(
                range(total_size), [train_size, valid_size],
                generator=torch.Generator().manual_seed(seed)
            )

            self.train_dataset = Subset(train_dataset, train_subset.indices)

            valid_dataset = OLEDDataset(data_dir, categories, 'train', test_transform, **params)
            self.valid_dataset = Subset(valid_dataset, valid_subset.indices)
        else:
            self.train_dataset = train_dataset
            self.valid_dataset = None

        self.test_dataset = OLEDDataset(data_dir, categories, 'test', test_transform, **params)

    def train_loader(self):
        return DataLoader(self.train_dataset, self.train_batch_size,
            shuffle=True, drop_last=True, **self.params)

    def valid_loader(self):
        if self.valid_dataset is None:
            return None
        return DataLoader(self.valid_dataset, self.test_batch_size,
            shuffle=False, drop_last=False, **self.params)

    def test_loader(self):
        return DataLoader(self.test_dataset, self.test_batch_size,
            shuffle=False, drop_last=False, **self.params)