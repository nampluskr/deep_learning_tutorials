import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


class MVTecDataset(Dataset):
    """MVTec anomaly detection dataset with dict-based interface"""
    def __init__(self, data_dir, categories, split, transform=None, normal_only=False):
        super().__init__()
        self.transform = transform
        self.normal_only = normal_only
        self.image_paths = []
        self.labels = []

        for category in categories:
            category_path = os.path.join(data_dir, category, split)
            if split == "train":
                # Training split only contains normal samples
                label = 0
                for path in glob(os.path.join(category_path, "good", "*.png")):
                    self.image_paths.append(path)
                    self.labels.append(label)
            else:
                # Test split contains both normal and anomalous samples
                for subfolder in os.listdir(category_path):
                    label = 0 if subfolder == "good" else 1
                    
                    # Skip anomalous samples if normal_only is True
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
        
        # Return dict format for new interface
        return {
            "input": image,
            "target": image,  # For autoencoder, target is same as input
            "label": label,
            "path": path
        }


class MVTecNormalDataset(Dataset):
    """MVTec dataset that only returns normal samples for training"""
    def __init__(self, data_dir, categories, split, transform=None):
        super().__init__()
        self.transform = transform
        self.image_paths = []

        for category in categories:
            category_path = os.path.join(data_dir, category, split)
            if split == "train":
                # Training split - only good samples
                for path in glob(os.path.join(category_path, "good", "*.png")):
                    self.image_paths.append(path)
            else:
                # Test/validation split - only good samples for anomaly detection training
                good_path = os.path.join(category_path, "good")
                if os.path.exists(good_path):
                    for path in glob(os.path.join(good_path, "*.png")):
                        self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = read_image(path, mode=ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        # All samples are normal (label=0) by design
        label = torch.tensor(0).long()
        
        # Return dict format for new interface
        return {
            "input": image,
            "target": image,  # For autoencoder, target is same as input
            "label": label,
            "path": path
        }


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
    """Split train dataset into train and validation sets"""
    # Use the total length from train_dataset for splitting
    total_size = len(train_dataset)
    valid_size = int(valid_ratio * total_size)
    train_size = total_size - valid_size

    # Generate random split indices
    torch.manual_seed(seed)
    train_subset, valid_subset = random_split(
        range(total_size), [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create subsets
    train_subset = Subset(train_dataset, train_subset.indices)
    valid_subset = Subset(valid_dataset, valid_subset.indices)
    
    return train_subset, valid_subset


def get_dataloader(dataset, batch_size, split, num_workers=4):
    """Create dataloader with appropriate settings"""
    loader_params = {
        'num_workers': num_workers, 
        'pin_memory': True, 
        'persistent_workers': True if num_workers > 0 else False
    }
    
    if split == "train":
        dataloader = DataLoader(dataset, batch_size,
            shuffle=True, drop_last=True, **loader_params)
    else:
        dataloader = DataLoader(dataset, batch_size,
            shuffle=False, drop_last=False, **loader_params)
    return dataloader


def get_anomaly_detection_dataloaders(config):
    """Get dataloaders specifically for anomaly detection training"""
    
    # Get transforms
    train_transform, test_transform = get_transforms(img_size=config.img_size)
    
    # Create datasets - use MVTecNormalDataset for training (normal samples only)
    train_dataset = MVTecNormalDataset(
        config.data_dir, config.categories, 'train', transform=train_transform
    )
    valid_dataset = MVTecNormalDataset(
        config.data_dir, config.categories, 'train', transform=test_transform
    )
    
    # Split train/valid
    train_dataset, valid_dataset = split_train_valid(
        train_dataset, valid_dataset, 
        valid_ratio=config.valid_ratio, 
        seed=config.seed
    )
    
    # Create test dataset (includes both normal and anomalous for evaluation)
    test_dataset = MVTecDataset(
        config.data_dir, config.categories, 'test', transform=test_transform
    )
    
    # Create dataloaders
    train_loader = get_dataloader(train_dataset, config.batch_size, 'train')
    valid_loader = get_dataloader(valid_dataset, 16, 'valid')  # Smaller batch for validation
    test_loader = get_dataloader(test_dataset, 16, 'test')
    
    return train_loader, valid_loader, test_loader


def get_evaluation_dataloader(config):
    """Get test dataloader for anomaly detection evaluation"""
    _, test_transform = get_transforms(img_size=config.img_size)
    
    test_dataset = MVTecDataset(
        config.data_dir, config.categories, 'test', transform=test_transform
    )
    
    test_loader = get_dataloader(test_dataset, 16, 'test')
    return test_loader


if __name__ == "__main__":
    # Example usage and testing
    from config import Config
    
    config = Config(
        data_dir='/mnt/d/datasets/mvtec',
        categories=['bottle'],
        img_size=256,
        batch_size=32,
        valid_ratio=0.2,
        seed=42
    )
    
    # Test the new interface
    print("Testing new data interface...")
    
    # Get dataloaders
    train_loader, valid_loader, test_loader = get_anomaly_detection_dataloaders(config)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Valid dataset size: {len(valid_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test a single batch
    train_batch = next(iter(train_loader))
    print(f"\nTrain batch keys: {train_batch.keys()}")
    print(f"Input shape: {train_batch['input'].shape}")
    print(f"Target shape: {train_batch['target'].shape}")
    print(f"Labels: {train_batch['label']}")  # Should all be 0 (normal)
    
    test_batch = next(iter(test_loader))
    print(f"\nTest batch keys: {test_batch.keys()}")
    print(f"Input shape: {test_batch['input'].shape}")
    print(f"Target shape: {test_batch['target'].shape}")
    print(f"Labels: {test_batch['label']}")  # Mix of 0 (normal) and 1 (anomalous)
    
    print("\nData interface test completed!")