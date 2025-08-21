import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T
import cv2


class MVTec(Dataset):
    """MVTec anomaly detection dataset"""
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
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()
        return {"image": image, "label": label}


def create_transforms(img_size=256, augment=True):
    """Create data transforms for training and testing"""
    base_transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
    ]
    
    if augment:
        augment_transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
        transforms = T.Compose(base_transforms + augment_transforms + [T.ToTensor()])
    else:
        transforms = T.Compose(base_transforms + [T.ToTensor()])
    
    return transforms


def split_train_valid(dataset, valid_ratio=0.2, seed=42):
    """Split dataset into train and validation indices"""
    data_size = len(dataset)
    valid_size = int(data_size * valid_ratio)
    train_size = data_size - valid_size
    
    torch.manual_seed(seed)
    train_subset, valid_subset = random_split(dataset, [train_size, valid_size])
    return train_subset.indices, valid_subset.indices


def create_mvtec_loaders(data_dir, categories, img_size=256, batch_size=16, 
                        valid_ratio=0.2, num_workers=4, seed=42):
    """Create MVTec data loaders for training, validation, and testing"""
    
    # Create transforms
    train_transform = create_transforms(img_size, augment=True)
    test_transform = create_transforms(img_size, augment=False)
    
    # Create datasets
    train_dataset = MVTec(data_dir, categories, split="train", transform=train_transform)
    valid_dataset = MVTec(data_dir, categories, split="train", transform=test_transform)
    test_dataset = MVTec(data_dir, categories, split="test", transform=test_transform)
    
    # Split train/validation
    train_indices, valid_indices = split_train_valid(train_dataset, valid_ratio, seed)
    train_dataset = Subset(train_dataset, train_indices)
    valid_dataset = Subset(valid_dataset, valid_indices)
    
    # Create data loaders
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": True if num_workers > 0 else False
    }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, valid_loader, test_loader


def get_dataset_info(loader):
    """Get basic information about the dataset"""
    total_samples = len(loader.dataset)
    normal_count = 0
    anomaly_count = 0
    
    for data in loader:
        labels = data['label']
        normal_count += (labels == 0).sum().item()
        anomaly_count += (labels == 1).sum().item()
    
    return {
        'total_samples': total_samples,
        'normal_samples': normal_count,
        'anomaly_samples': anomaly_count,
        'anomaly_ratio': anomaly_count / total_samples if total_samples > 0 else 0
    }
