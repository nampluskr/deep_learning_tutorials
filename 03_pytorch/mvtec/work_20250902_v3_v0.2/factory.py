import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from .dataset_base import BaseDataloader, load_image


class VisADataset(Dataset):

    def __init__(self, data_dir, categories, transform=None, 
                 load_normal=False, load_anomaly=False, **kwargs):

        super().__init__()
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.extensions = kwargs.get('extensions', ['*.JPG', '*.jpg', '*.png'])

        for category in categories:
            category_path = os.path.join(data_dir, category)
            if not os.path.exists(category_path):
                return

            if load_normal:
                normal_path = os.path.join(category_path, "Data", "Images", "Normal")
                self._add_files(normal_path, label=0)

            if load_anomaly:
                anomaly_path = os.path.join(category_path, "Data", "Images", "Anomaly")
                self._add_files(anomaly_path, label=1)

    def _add_files(self, dir_path, label):
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
        image = load_image(path, mode="RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return {"image": image, "label": label}


class VisADataloader(BaseDataloader):
    def __init__(self, data_dir, categories,
                 train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16,
                 test_ratio=0.2, valid_ratio=0.0, seed=42, **params):

        super().__init__(data_dir, categories,
                         train_transform=train_transform, test_transform=test_transform,
                         train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                         test_ratio=test_ratio, valid_ratio=valid_ratio, seed=seed, **params)

    def get_dataset(self, data_dir, categories, transform, load_normal, load_anomaly, **params):
        return VisADataset(data_dir, categories, transform, load_normal, load_anomaly, **params)