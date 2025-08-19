import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as T
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, random_split

import numpy as np
import random
import sys
from tqdm import tqdm
from time import time


def get_transforms(transform_type='default', **transform_params):
    """Get train and test transforms with GPU acceleration support"""
    available_types = ['light', 'default', 'heavy', 'custom', 'gpu_optimized']

    if transform_type not in available_types:
        raise ValueError(f"Unknown transform type: {transform_type}. Available: {available_types}")

    img_size = transform_params.get('img_size', 256)
    device = transform_params.get('device', None)

    # Base transforms
    base_transforms = [
        v2.Resize((img_size, img_size), antialias=True),
        v2.ConvertImageDtype(torch.float32),
    ]

    test_transforms = base_transforms.copy()

    if transform_type == 'light':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomRotation(degrees=5),
            v2.ColorJitter(brightness=0.05, contrast=0.05),
        ]
    elif transform_type == 'default':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ]
    elif transform_type == 'heavy':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=0.6),
            v2.RandomVerticalFlip(p=0.4),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            v2.RandomPerspective(distortion_scale=0.1, p=0.2),
            v2.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random erasing augmentation
        ]
    elif transform_type == 'gpu_optimized':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.RandomErasing(p=0.1),
        ]
    elif transform_type == 'custom':
        h_flip_p = transform_params.get('h_flip_p', 0.5)
        v_flip_p = transform_params.get('v_flip_p', 0.3)
        rotation_degrees = transform_params.get('rotation_degrees', 10)
        brightness = transform_params.get('brightness', 0.1)
        contrast = transform_params.get('contrast', 0.1)
        blur_kernel = transform_params.get('blur_kernel', 3)
        blur_sigma = transform_params.get('blur_sigma', (0.1, 0.5))
        blur_p = transform_params.get('blur_p', 0.0)
        erasing_p = transform_params.get('erasing_p', 0.0)

        aug_transforms = [
            v2.RandomHorizontalFlip(p=h_flip_p),
            v2.RandomVerticalFlip(p=v_flip_p),
            v2.RandomRotation(degrees=rotation_degrees),
            v2.ColorJitter(brightness=brightness, contrast=contrast),
        ]

        if blur_p > 0:
            aug_transforms.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)], p=blur_p)
            )

        if erasing_p > 0:
            aug_transforms.append(v2.RandomErasing(p=erasing_p))

    train_transforms = base_transforms + aug_transforms
    print(f" > Creating v2 transforms: type={transform_type}, img_size={img_size}")
    print(f" > Train augmentations: {len(aug_transforms)} transforms")
    if device:
        print(f" > GPU acceleration: {device}")
    if transform_params:
        print(f" > Transform parameters: {transform_params}")

    return v2.Compose(train_transforms), v2.Compose(test_transforms)