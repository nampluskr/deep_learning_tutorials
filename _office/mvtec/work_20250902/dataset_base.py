import os
import numpy as np
import skimage.io

import torch
import torchvision.io
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision.transforms import v2


def load_image(path, mode='RGB'):
    """Load image as tensor [C,H,W], float32, normalized to [0,1] range."""

    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        if mode == 'RGB':
            img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)
        else:  # mode == 'L' (grayscale)
            img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.GRAY)
        return img.float() / 255.0

    # Fallback to skimage for other formats (e.g., BMP, TIFF)
    else:
        img = skimage.io.imread(path, as_gray=(mode == 'L'))

        # Handle different data types
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        elif img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

        # Convert to PyTorch tensor and rearrange dimensions
        if img.ndim == 2:  # Grayscale image
            img = torch.from_numpy(img).unsqueeze(0).float()  # [1, H, W]
        else:  # RGB image
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]

        # Normalize to [0, 1] if pixel values are in [0, 255]
        return img / 255.0 if img.max() > 1.0 else img


class TrainTransform:
    def __init__(self, img_size=256, **params):
        flip_prob = params.get('flip_prob', 0.5)
        rotation_degrees = params.get('rotation_degrees', 15)
        brightness = params.get('brightness', 0.1)
        contrast = params.get('contrast', 0.1)
        saturation = params.get('saturation', 0.1)
        hue = params.get('hue', 0.05)

        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.RandomHorizontalFlip(p=flip_prob),
            v2.RandomVerticalFlip(p=flip_prob),
            v2.RandomRotation(degrees=rotation_degrees),
            v2.ColorJitter(brightness=brightness, contrast=contrast,
                          saturation=saturation, hue=hue),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestTransform:
    def __init__(self, img_size=256, **params):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, image):
        return self.transform(image)


class BaseDataloader:
    def __init__(self, data_dir, categories,
                 train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16,
                 test_ratio=0.2, valid_ratio=0.0, seed=42, **params):

        self.categories = categories
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.params = params

        train_normal_dataset = self.get_dataset(data_dir, categories, train_transform, load_normal=True, load_anomaly=False, **params)
        test_normal_dataset = self.get_dataset(data_dir, categories, test_transform, load_normal=True, load_anomaly=False, **params)

        total_size = len(train_normal_dataset)
        test_size = int(total_size * test_ratio)
        valid_size = int(total_size * valid_ratio)
        train_size = total_size - test_size - valid_size

        train_subset, valid_subset, test_subset = random_split(
            range(total_size), [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        self.train_normal_dataset = Subset(train_normal_dataset, train_subset.indices)
        self.valid_normal_dataset = Subset(test_normal_dataset, valid_subset.indices)
        self.test_normal_dataset = Subset(test_normal_dataset, test_subset.indices)
        self.test_anomaly_dataset = self.get_dataset(data_dir, categories, test_transform, load_normal=False, load_anomaly=True, **params)

    def get_dataset(self, data_dir, categories, transform, load_normal, load_anomaly, **params):
        raise NotImplementedError("get_dataset method must be implemented in subclass")

    def train_loader(self):
        return DataLoader(self.train_normal_dataset, self.train_batch_size,
                          shuffle=True, drop_last=True, **self.params)

    def valid_loader(self):
        if len(self.valid_normal_dataset) == 0:
            return None
        else:
            return DataLoader(self.valid_normal_dataset, self.test_batch_size,
                              shuffle=False, drop_last=False, **self.params)

    def test_loader(self):
        if len(self.test_normal_dataset) == 0 and len(self.test_anomaly_dataset) == 0:
            return None
        else:
            test_dataset = ConcatDataset([self.test_normal_dataset, self.test_anomaly_dataset])
            return DataLoader(test_dataset, self.test_batch_size,
                              shuffle=False, drop_last=False, **self.params)
