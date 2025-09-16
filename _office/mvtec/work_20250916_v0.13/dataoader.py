import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MVTecDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []    # 0 = normal, 1 = anomaly
        self.mask_transform = mask_transform

        if split == 'train':
            good_dir = os.path.join(self.root_dir, 'train', 'good')
            if os.path.exists(good_dir):
                for fname in os.listdir(good_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(good_dir, fname)
                        self.img_paths.append(full_path)
                        self.labels.append(0)
        else:
            test_dir = os.path.join(self.root_dir, 'test')
            if os.path.exists(test_dir):
                for sub_name in os.listdir(test_dir):
                    sub_path = os.path.join(test_dir, sub_name)
                    if not os.path.isdir(sub_path):
                        continue
                    for fname in os.listdir(sub_path):
                        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            continue
                        full_path = os.path.join(sub_path, fname)
                        self.img_paths.append(full_path)
                        self.labels.append(0 if sub_name == 'good' else 1)

        print(f" > {split.capitalize()} set: {len(self.img_paths)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        height, width = image.shape[-2:]
        label = self.labels[idx]
        if label == 0:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_path = image_path.replace("test", "ground_truth").replace(".png", "_mask.png")
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = (np.array(mask) > 0).astype(np.uint8)

        label = torch.tensor(label).long()
        mask = torch.tensor(mask).long()
        name = os.path.basename(image_path)
        return dict(image=image, label=label, mask=mask, name=name)



def get_transforms(img_size=256):
    train_transform = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])
    test_transform = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])
    mask_transform = v2.Resize((img_size, img_size), antialias=True)
    return train_transform, test_transform, mask_transform


# def denormalize_imagenet(tensor):
#     """Convert ImageNet normalized tensor back to [0, 1] range for visualization"""
#     device = tensor.device
#     mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1, 1, 1)
#     return torch.clamp(tensor * std + mean, 0.0, 1.0)


# def normalize_to_imagenet_range(tensor):
#     """Convert [0, 1] tensor to ImageNet normalized range"""
#     device = tensor.device
#     mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1, 1, 1)
#     return (tensor - mean) / std


def get_dataloaders(root, category, batch_size=16, img_size=256):
    data_root = os.path.join(root, category)
    train_transform, test_transform, mask_transform = get_transforms(img_size=img_size)

    train_set = MVTecDataset(root_dir=data_root, split='train', transform=train_transform,
        mask_transform=mask_transform)
    test_set = MVTecDataset(root_dir=data_root, split='test', transform=test_transform,
        mask_transform=mask_transform)

    # dataloader_params = dict(num_workers=8, pin_memory=True, persistent_workers=True) # WSL
    dataloader_params = dict(num_workers=8, pin_memory=True, persistent_workers=False)  # NAMU
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        drop_last=True, **dataloader_params)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        drop_last=False, **dataloader_params)

    return train_loader, test_loader
