import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


class MVTecDataset(Dataset):
    """MVTec anomaly detection dataset using torchvision.io.read_image"""
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
        image = read_image(path, mode=ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()
        return {"input": image, "target": image, "label": label}


def get_transforms(img_size=256):
    """Get train and test transforms for MVTec dataset"""

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


def split_train_valid(train_dataset, valid_dataset, valid_ratio=0.2, seed=42):
    valid_size = int(valid_ratio * len(valid_dataset))
    train_size = len(train_dataset) - valid_size

    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed))
    train_dataset = Subset(train_dataset, train_subset.indices)
    valid_dataset = Subset(valid_dataset, valid_subset.indices)
    return train_dataset, valid_dataset


def get_dataloader(dataset, batch_size, split):
    loader_params = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    if split == "train":
        dataloader = DataLoader(dataset, batch_size,
            shuffle=True, drop_last=True, **loader_params)
    else:
        dataloader = DataLoader(dataset, batch_size,
            shuffle=False, drop_last=False, **loader_params)
    return dataloader


if __name__ == "__main__":
    # Example usage
    pass