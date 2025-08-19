"""
MVTec Anomaly Detection Dataset module for data loading and preprocessing
Handles MVTec dataset loading, transforms, and dataloader creation
"""

import os
from glob import glob
import warnings

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class MVTecDataset(Dataset):
    """MVTec anomaly detection dataset loader"""

    def __init__(self, data_dir, category, split, transform=None):
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
            # PyTorch read_image 사용 (RGB 모드, 빠르고 안정적)
            image = read_image(image_path, mode=ImageReadMode.RGB)  # [C, H, W], uint8
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy tensor in case of error
            image = torch.zeros(3, 256, 256, dtype=torch.uint8)

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {image_path}: {e}")
                # Return normalized tensor in case of transform error
                image = torch.zeros(3, 256, 256, dtype=torch.float32)

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


if __name__ == "__main__":
    # Example usage
    pass
