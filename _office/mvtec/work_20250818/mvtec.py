import os
from glob import glob
import warnings

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T


def get_transforms(img_size=256):
    """Get training and testing transforms for data augmentation"""
    
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
    return T.Compose(train_transforms), T.Compose(test_transforms)


class MVTecDataset(Dataset):
    """MVTec anomaly detection dataset loader"""
    
    def __init__(self, data_dir, category, split, transform=None):
        """
        Initialize MVTec dataset
        
        Args:
            data_dir: Root directory of MVTec dataset
            category: Product category (e.g., 'bottle', 'cable', etc.)
            split: Dataset split ('train' or 'test')
            transform: Transform pipeline to apply to images
        """
        self.data_dir = data_dir
        self.category = category
        self.split = split          # "train" or "test"
        self.transform = transform
        self.image_paths = []
        self.labels = []            # 0=normal, 1=anomaly
        self.defect_types = []

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset images and labels"""
        category_path = os.path.join(self.data_dir, self.category, self.split)
        
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Dataset path not found: {category_path}")

        if self.split == "train":
            self._load_train_data(category_path)
        else:
            self._load_test_data(category_path)
            
        if len(self.image_paths) == 0:
            warnings.warn(f"No images found in {category_path}")

    def _load_train_data(self, category_path):
        """Load training data (only normal samples)"""
        good_path = os.path.join(category_path, "good")
        if not os.path.exists(good_path):
            raise FileNotFoundError(f"Good samples path not found: {good_path}")
            
        for img_file in glob(os.path.join(good_path, "*.png")):
            self.image_paths.append(img_file)
            self.labels.append(0)
            self.defect_types.append("good")

    def _load_test_data(self, category_path):
        """Load test data (normal and anomalous samples)"""
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Test path not found: {category_path}")
            
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            label = 0 if subfolder == "good" else 1
            for img_file in glob(os.path.join(subfolder_path, "*.png")):
                self.image_paths.append(img_file)
                self.labels.append(label)
                self.defect_types.append(subfolder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single data sample"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image in case of error
            image = torch.zeros(3, 256, 256)
            
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {image_path}: {e}")
                # Return normalized tensor in case of transform error
                image = torch.zeros(3, 256, 256)

        return {
            "image": image,
            "label": torch.tensor(label).long(),
            "defect_type": defect_type,
            "path": image_path
        }



    def get_statistics(self):
        """Get dataset statistics"""
        normal_count = sum(1 for l in self.labels if l == 0)
        anomaly_count = sum(1 for l in self.labels if l == 1)
        defect_types = list(set(self.defect_types))
        
        return {
            'total_images': len(self.image_paths),
            'normal_samples': normal_count,
            'anomaly_samples': anomaly_count,
            'defect_types': defect_types,
            'category': self.category,
            'split': self.split
        }


def get_dataloaders(data_dir, category, batch_size, valid_ratio=0.2,
                    train_transform=None, test_transform=None):
    """Create data loaders for training, validation, and testing"""
    
    try:
        train_dataset = MVTecDataset(data_dir, category, "train", transform=train_transform)
        test_dataset = MVTecDataset(data_dir, category, "test", transform=test_transform)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        raise

    # Split training data into train and validation sets
    valid_size = int(valid_ratio * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    if train_size <= 0 or valid_size <= 0:
        raise ValueError(f"Invalid dataset split: train_size={train_size}, valid_size={valid_size}")

    train_subset, valid_subset = random_split(
        train_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset with test transforms
    valid_dataset = MVTecDataset(data_dir, category, "train", transform=test_transform)
    valid_dataset = Subset(valid_dataset, valid_subset.indices)
    train_dataset = Subset(train_dataset, train_subset.indices)

    # DataLoader parameters
    params = {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True
    } if torch.cuda.is_available() else {}

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        **params
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=32, 
        shuffle=False, 
        drop_last=False, 
        **params
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        drop_last=False, 
        **params
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/mvtec/dataset"
    category = "bottle"
    batch_size = 32

    try:
        train_transform, test_transform = get_transforms(img_size=256)
        train_loader, valid_loader, test_loader = get_dataloaders(
            data_dir, category, batch_size,
            train_transform=train_transform,
            test_transform=test_transform
        )

        # Print dataset statistics
        for loader_name, loader in [("Train", train_loader), ("Valid", valid_loader), ("Test", test_loader)]:
            print(f"{loader_name} loader: {len(loader)} batches")
            
        # Test a single batch
        for batch in train_loader:
            print(f"Batch shapes: {batch['image'].shape}, Labels: {batch['label'].shape}")
            print(f"Defect types: {batch['defect_type']}")
            break
            
    except Exception as e:
        print(f"Error in example usage: {e}")
