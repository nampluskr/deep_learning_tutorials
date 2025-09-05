import os
import numpy as np
from glob import glob
import skimage.io
from functools import cached_property

import torch
import torchvision.io
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
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


# ===================================================================
# Base Dataloader
# ===================================================================

class BaseDataloader:
    def __init__(self, data_dir, categories,
                 train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16,
                 test_ratio=0.2, valid_ratio=0.0, seed=42,
                 train_shuffle=True, test_shuffle=False,
                 train_drop_last=True, test_drop_last=False,
                 **params):

        self.categories = categories
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        self.train_drop_last = train_drop_last
        self.test_drop_last = test_drop_last
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
                          shuffle=self.train_shuffle,
                          drop_last=self.train_drop_last,
                          **self.params)

    def valid_loader(self):
        if len(self.valid_normal_dataset) == 0:
            return None
        else:
            return DataLoader(self.valid_normal_dataset, self.test_batch_size,
                              shuffle=self.test_shuffle,
                              drop_last=self.test_drop_last,
                              **self.params)

    def test_loader(self):
        if len(self.test_normal_dataset) == 0 and len(self.test_anomaly_dataset) == 0:
            return None
        else:
            test_dataset = ConcatDataset([self.test_normal_dataset, self.test_anomaly_dataset])
            return DataLoader(test_dataset, self.test_batch_size,
                              shuffle=self.test_shuffle,
                              drop_last=self.test_drop_last,
                              **self.params)


# ===================================================================
# MVTec Dataloader
# ===================================================================

class MVTecDataset(Dataset):

    def __init__(self, data_dir, categories, transform=None, 
        load_normal=False, load_anomaly=False, **kwargs):

        super().__init__()
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.extensions = kwargs.get('extensions', ['*.png'])

        for category in categories:
            category_path = os.path.join(data_dir, category)
            if not os.path.exists(category_path):
                return

            if load_normal:
                train_normal_path = os.path.join(category_path, "train", "good")
                test_normal_path = os.path.join(category_path, "test", "good")
                self._add_files(train_normal_path, label=0)
                self._add_files(test_normal_path, label=0)

            if load_anomaly:
                for subfolder in os.listdir(os.path.join(category_path, "test")):
                    if subfolder != "good":
                        test_anomaly_path = os.path.join(category_path, "test", subfolder)
                        self._add_files(test_anomaly_path, label=1)      

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
    
    @cached_property  
    def anomaly_count(self):
        return sum(self.labels)
    
    @cached_property
    def normal_count(self):
        return len(self.labels) - self.anomaly_count


class MVTecDataloader(BaseDataloader):
    def __init__(self, data_dir, categories,
                 train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16,
                 test_ratio=0.2, valid_ratio=0.0, seed=42,
                 train_shuffle=True, test_shuffle=False,
                 train_drop_last=True, test_drop_last=False,
                 **params):

        super().__init__(data_dir, categories,
                         train_transform=train_transform, test_transform=test_transform,
                         train_batch_size=train_batch_size, test_batch_size=test_batch_size,
                         test_ratio=test_ratio, valid_ratio=valid_ratio, seed=seed,
                         train_shuffle=train_shuffle, test_shuffle=test_shuffle,
                         train_drop_last=train_drop_last, test_drop_last=test_drop_last,
                         **params)

    def get_dataset(self, data_dir, categories, transform, load_normal, load_anomaly, **params):
        return MVTecDataset(data_dir, categories, transform, load_normal, load_anomaly, **params)
