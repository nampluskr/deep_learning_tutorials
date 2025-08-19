"""
MVTec 데이터셋 로딩, 전처리, DataLoader 생성
"""

import os
from glob import glob

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T


def get_transforms(img_size=256, normalize=False):
    train_transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
        T.ToTensor(),
    ]
    test_transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ]
    if normalize:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms.append(T.Normalize(mean=mean, std=std))
        test_transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(train_transforms), T.Compose(test_transforms)


class MVTecDataset(Dataset):
    def __init__(self, data_dir, category, split, transform=None):
        self.data_dir = data_dir
        self.category = category
        self.split = split          # "train" or "test"
        self.transform = transform
        self.image_paths = []
        self.labels = []            # 0=normal, 1=anomaly
        self.defect_types = []

        category_path = os.path.join(self.data_dir, self.category, self.split)

        if self.split == "train":
            good_path = os.path.join(category_path, "good")
            for img_file in glob(os.path.join(good_path, "*.png")):
                self.image_paths.append(img_file)
                self.labels.append(0)
                self.defect_types.append("good")
        else:
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                label = 0 if subfolder == "good" else 1
                for img_file in glob(os.path.join(subfolder_path, "*.png")):
                    self.image_paths.append(img_file)
                    self.labels.append(label)
                    self.defect_types.append(subfolder)

        print(f"Loaded {self.category} {self.split}: {len(self.image_paths)} images")
        print(f">> Normal:  {sum(1 for l in self.labels if l == 0)}")
        print(f">> Anomaly: {sum(1 for l in self.labels if l == 1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label).long(),
            "defect_type": defect_type,
            "path": image_path
        }


def get_dataloaders(data_dir, category, batch_size, valid_ratio=0.2,
                    train_transform=None, test_transform=None):
    train_dataset = MVTecDataset(data_dir, category, "train", transform=train_transform)
    test_dataset = MVTecDataset(data_dir, category, "test", transform=test_transform)

    valid_size = int(valid_ratio * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    train_subset, valid_subset = random_split(
        train_dataset,
        [train_size, valid_size], 
        generator=torch.Generator().manual_seed(42)
    )
    valid_dataset = MVTecDataset(data_dir, category, "train", test_transform)
    valid_dataset = Subset(valid_dataset, valid_subset.indices)
    train_dataset = Subset(train_dataset, train_subset.indices)

    num_workers = 4
    params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False
    } if torch.cuda.is_available() else {"batch_size": batch_size}

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **params)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **params)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **params)

    return train_loader, valid_loader, test_loader