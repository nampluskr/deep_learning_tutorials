import os
import json
from glob import glob
import warnings
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


def get_transforms(img_size=256, normalize=False, augmentation_level='medium'):
    """
    Get training and testing transforms for data augmentation
    
    Args:
        img_size: Target image size
        normalize: Whether to apply ImageNet normalization
        augmentation_level: Level of augmentation ('light', 'medium', 'heavy', 'oled')
    """
    
    # Base transforms
    base_transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
    ]
    
    # Augmentation strategies
    augmentation_configs = {
        'light': {
            'horizontal_flip': 0.3,
            'vertical_flip': 0.1,
            'rotation': 5,
            'color_jitter': {'brightness': 0.05, 'contrast': 0.05, 'saturation': 0.02, 'hue': 0.01},
            'gaussian_blur': {'p': 0.1, 'kernel_size': 3, 'sigma': (0.1, 0.3)},
            'gaussian_noise': {'p': 0.05, 'std': 0.01}
        },
        'medium': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.3,
            'rotation': 10,
            'color_jitter': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.05, 'hue': 0.02},
            'gaussian_blur': {'p': 0.2, 'kernel_size': 3, 'sigma': (0.1, 0.5)},
            'gaussian_noise': {'p': 0.1, 'std': 0.02}
        },
        'heavy': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.3,
            'rotation': 15,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.05},
            'gaussian_blur': {'p': 0.3, 'kernel_size': 5, 'sigma': (0.1, 1.0)},
            'gaussian_noise': {'p': 0.15, 'std': 0.03},
            'elastic_transform': {'p': 0.1, 'alpha': 50, 'sigma': 5}
        },
        'oled': {  # OLED-specific augmentations
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,  # OLED patterns can be symmetric
            'rotation': 5,  # Small rotations for display alignment
            'color_jitter': {'brightness': 0.05, 'contrast': 0.05, 'saturation': 0.02},
            'gaussian_blur': {'p': 0.1, 'kernel_size': 3, 'sigma': (0.1, 0.3)},
            'gaussian_noise': {'p': 0.05, 'std': 0.005},  # Sensor noise simulation
            'gamma_correction': {'p': 0.2, 'gamma_range': (0.8, 1.2)}
        }
    }
    
    config = augmentation_configs.get(augmentation_level, augmentation_configs['medium'])
    
    # Training transforms with augmentation
    train_transforms = base_transforms.copy()
    
    # Add augmentations
    if config.get('horizontal_flip', 0) > 0:
        train_transforms.append(T.RandomHorizontalFlip(p=config['horizontal_flip']))
    
    if config.get('vertical_flip', 0) > 0:
        train_transforms.append(T.RandomVerticalFlip(p=config['vertical_flip']))
    
    if config.get('rotation', 0) > 0:
        train_transforms.append(T.RandomRotation(degrees=config['rotation']))
    
    if 'color_jitter' in config:
        cj = config['color_jitter']
        train_transforms.append(T.ColorJitter(
            brightness=cj.get('brightness', 0),
            contrast=cj.get('contrast', 0),
            saturation=cj.get('saturation', 0),
            hue=cj.get('hue', 0)
        ))
    
    if 'gaussian_blur' in config:
        gb = config['gaussian_blur']
        train_transforms.append(T.RandomApply([
            T.GaussianBlur(kernel_size=gb['kernel_size'], sigma=gb['sigma'])
        ], p=gb['p']))
    
    # Convert to tensor
    train_transforms.append(T.ToTensor())
    
    # Add Gaussian noise if specified
    if 'gaussian_noise' in config:
        gn = config['gaussian_noise']
        train_transforms.append(GaussianNoise(p=gn['p'], std=gn['std']))
    
    # Add gamma correction for OLED
    if 'gamma_correction' in config:
        gc = config['gamma_correction']
        train_transforms.append(RandomGammaCorrection(p=gc['p'], gamma_range=gc['gamma_range']))
    
    # Test transforms (no augmentation)
    test_transforms = base_transforms + [T.ToTensor()]
    
    # Apply normalization if requested
    if normalize:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms.append(T.Normalize(mean=mean, std=std))
        test_transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(train_transforms), T.Compose(test_transforms)


class GaussianNoise(object):
    """Add Gaussian noise to tensor"""
    
    def __init__(self, p=0.1, std=0.01):
        self.p = p
        self.std = std
    
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, 0, 1)
        return tensor


class RandomGammaCorrection(object):
    """Apply random gamma correction"""
    
    def __init__(self, p=0.2, gamma_range=(0.8, 1.2)):
        self.p = p
        self.gamma_range = gamma_range
    
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            gamma = torch.uniform(*self.gamma_range)
            return torch.pow(tensor, gamma)
        return tensor


class MVTecDataset(Dataset):
    """Enhanced MVTec anomaly detection dataset loader"""
    
    def __init__(self, data_dir, category, split, transform=None, 
                 img_size=256, load_masks=False, cache_images=False):
        """
        Initialize MVTec dataset
        
        Args:
            data_dir: Root directory of MVTec dataset
            category: Product category (e.g., 'bottle', 'cable', etc.) or list of categories
            split: Dataset split ('train' or 'test')
            transform: Transform pipeline to apply to images
            img_size: Target image size for error handling
            load_masks: Whether to load ground truth masks (for test set)
            cache_images: Whether to cache images in memory for faster loading
        """
        self.data_dir = data_dir
        self.category = category if isinstance(category, list) else [category]
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.load_masks = load_masks and split == 'test'
        self.cache_images = cache_images
        
        self.image_paths = []
        self.labels = []
        self.defect_types = []
        self.categories = []
        self.mask_paths = []
        self.image_cache = {} if cache_images else None

        self._load_dataset()
        self._validate_dataset()

    def _load_dataset(self):
        """Load dataset images and labels"""
        for cat in self.category:
            category_path = os.path.join(self.data_dir, cat, self.split)
            
            if not os.path.exists(category_path):
                warnings.warn(f"Dataset path not found: {category_path}")
                continue

            if self.split == "train":
                self._load_train_data(category_path, cat)
            else:
                self._load_test_data(category_path, cat)

    def _load_train_data(self, category_path, category):
        """Load training data (only normal samples)"""
        good_path = os.path.join(category_path, "good")
        if not os.path.exists(good_path):
            warnings.warn(f"Good samples path not found: {good_path}")
            return
            
        for img_file in sorted(glob(os.path.join(good_path, "*.png"))):
            self.image_paths.append(img_file)
            self.labels.append(0)
            self.defect_types.append("good")
            self.categories.append(category)
            if self.load_masks:
                self.mask_paths.append(None)  # No masks for normal samples

    def _load_test_data(self, category_path, category):
        """Load test data (normal and anomalous samples)"""
        if not os.path.exists(category_path):
            warnings.warn(f"Test path not found: {category_path}")
            return
            
        # Load ground truth path if masks are requested
        gt_path = os.path.join(self.data_dir, category, "ground_truth") if self.load_masks else None
            
        for subfolder in sorted(os.listdir(category_path)):
            subfolder_path = os.path.join(category_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            label = 0 if subfolder == "good" else 1
            
            for img_file in sorted(glob(os.path.join(subfolder_path, "*.png"))):
                self.image_paths.append(img_file)
                self.labels.append(label)
                self.defect_types.append(subfolder)
                self.categories.append(category)
                
                # Load corresponding mask path
                if self.load_masks:
                    if label == 0:  # Normal samples have no masks
                        self.mask_paths.append(None)
                    else:
                        # Find corresponding mask
                        img_name = os.path.basename(img_file)
                        mask_path = os.path.join(gt_path, subfolder, img_name)
                        self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)

    def _validate_dataset(self):
        """Validate dataset consistency"""
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for categories {self.category} in split {self.split}")
        
        # Check if all lists have same length
        lengths = [len(self.image_paths), len(self.labels), len(self.defect_types), len(self.categories)]
        if self.load_masks:
            lengths.append(len(self.mask_paths))
        
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("Inconsistent dataset: mismatched lengths of image paths, labels, and metadata")

    def _load_image_safe(self, image_path):
        """Safely load image with error handling"""
        try:
            # Try OpenCV first (faster for most cases)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("OpenCV failed to load image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            try:
                # Fallback to PIL
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
            except Exception as e2:
                print(f"Error loading image {image_path}: {e}, {e2}")
                # Create dummy image with appropriate size
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                
        return image

    def _load_mask_safe(self, mask_path):
        """Safely load mask with error handling"""
        if mask_path is None or not os.path.exists(mask_path):
            # Return empty mask
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.array(Image.open(mask_path).convert('L'))
            
            # Ensure binary mask
            mask = (mask > 128).astype(np.uint8) * 255
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single data sample"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        category = self.categories[idx]

        # Load from cache or disk
        if self.cache_images and image_path in self.image_cache:
            image = self.image_cache[image_path].copy()
        else:
            image = self._load_image_safe(image_path)
            if self.cache_images:
                self.image_cache[image_path] = image.copy()
            
        # Apply transforms
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {image_path}: {e}")
                # Create normalized tensor with correct size
                if hasattr(self.transform, 'transforms'):
                    # Check if normalization is applied
                    has_normalize = any(isinstance(t, T.Normalize) for t in self.transform.transforms)
                    if has_normalize:
                        # ImageNet normalized dummy image
                        image = torch.zeros(3, self.img_size, self.img_size)
                    else:
                        # Regular [0,1] dummy image
                        image = torch.zeros(3, self.img_size, self.img_size)
                else:
                    image = torch.zeros(3, self.img_size, self.img_size)

        sample = {
            "image": image,
            "label": torch.tensor(label).long(),
            "defect_type": defect_type,
            "category": category,
            "path": image_path
        }
        
        # Add mask if requested
        if self.load_masks:
            mask_path = self.mask_paths[idx]
            mask = self._load_mask_safe(mask_path)
            
            # Apply same spatial transforms to mask (without color transforms)
            if self.transform and mask_path:
                try:
                    # Create mask-specific transform (only spatial transforms)
                    mask_transforms = []
                    if hasattr(self.transform, 'transforms'):
                        for t in self.transform.transforms:
                            if isinstance(t, (T.ToPILImage, T.Resize, T.RandomHorizontalFlip, 
                                            T.RandomVerticalFlip, T.RandomRotation)):
                                mask_transforms.append(t)
                            elif isinstance(t, T.ToTensor):
                                mask_transforms.append(t)
                                break
                    
                    mask_transform = T.Compose(mask_transforms)
                    mask = mask_transform(mask)
                    
                    # Ensure mask is single channel
                    if mask.dim() == 3 and mask.size(0) == 3:
                        mask = mask[0:1]  # Take first channel
                    elif mask.dim() == 2:
                        mask = mask.unsqueeze(0)
                        
                except Exception as e:
                    print(f"Error applying transform to mask {mask_path}: {e}")
                    mask = torch.zeros(1, self.img_size, self.img_size)
            else:
                # Convert to tensor if no transform
                mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            
            sample["mask"] = mask

        return sample

    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_images': len(self.image_paths),
            'categories': list(set(self.categories)),
            'num_categories': len(set(self.categories)),
            'split': self.split
        }
        
        # Per-category statistics
        category_stats = defaultdict(lambda: {'normal': 0, 'anomaly': 0, 'defect_types': set()})
        
        for i, (cat, label, defect_type) in enumerate(zip(self.categories, self.labels, self.defect_types)):
            if label == 0:
                category_stats[cat]['normal'] += 1
            else:
                category_stats[cat]['anomaly'] += 1
            category_stats[cat]['defect_types'].add(defect_type)
        
        # Convert sets to lists for JSON serialization
        for cat in category_stats:
            category_stats[cat]['defect_types'] = sorted(list(category_stats[cat]['defect_types']))
        
        stats['category_breakdown'] = dict(category_stats)
        
        # Overall statistics
        stats['normal_samples'] = sum(1 for l in self.labels if l == 0)
        stats['anomaly_samples'] = sum(1 for l in self.labels if l == 1)
        stats['defect_types'] = sorted(list(set(self.defect_types)))
        stats['imbalance_ratio'] = stats['normal_samples'] / max(stats['anomaly_samples'], 1)
        
        return stats

    def compute_image_statistics(self, num_samples=1000) -> Dict:
        """Compute image-level statistics for normalization"""
        if len(self) == 0:
            return {}
        
        # Sample images for statistics
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        
        pixel_values = []
        image_sizes = []
        
        print(f"Computing image statistics from {len(indices)} samples...")
        
        for idx in indices:
            try:
                image = self._load_image_safe(self.image_paths[idx])
                image_sizes.append(image.shape[:2])  # (H, W)
                
                # Convert to [0, 1] range
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                
                pixel_values.append(image.reshape(-1, 3))
                
            except Exception as e:
                print(f"Error processing image {self.image_paths[idx]}: {e}")
                continue
        
        if not pixel_values:
            return {}
        
        # Concatenate all pixel values
        all_pixels = np.concatenate(pixel_values, axis=0)
        
        # Compute statistics
        stats = {
            'mean': all_pixels.mean(axis=0).tolist(),  # Per-channel mean
            'std': all_pixels.std(axis=0).tolist(),    # Per-channel std
            'min': all_pixels.min(axis=0).tolist(),    # Per-channel min
            'max': all_pixels.max(axis=0).tolist(),    # Per-channel max
            'num_samples_analyzed': len(indices),
            'image_sizes': {
                'unique_sizes': list(set(image_sizes)),
                'most_common_size': max(set(image_sizes), key=image_sizes.count),
                'size_distribution': {str(size): image_sizes.count(size) for size in set(image_sizes)}
            }
        }
        
        return stats

    def save_statistics(self, save_path: str):
        """Save dataset statistics to file"""
        stats = self.get_statistics()
        
        # Add image statistics if requested
        img_stats = self.compute_image_statistics()
        if img_stats:
            stats['image_statistics'] = img_stats
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics saved to {save_path}")


class MultiCategoryDataset(Dataset):
    """Dataset for training on multiple categories simultaneously"""
    
    def __init__(self, data_dir, categories, split, transform=None, **kwargs):
        """
        Initialize multi-category dataset
        
        Args:
            data_dir: Root directory of MVTec dataset
            categories: List of categories to include
            split: Dataset split ('train' or 'test')
            transform: Transform pipeline to apply to images
            **kwargs: Additional arguments for MVTecDataset
        """
        self.datasets = []
        
        for category in categories:
            dataset = MVTecDataset(
                data_dir=data_dir,
                category=category,
                split=split,
                transform=transform,
                **kwargs
            )
            self.datasets.append(dataset)
        
        # Combine all datasets
        self.combined_dataset = ConcatDataset(self.datasets)
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]
    
    def get_statistics(self):
        """Get combined statistics from all categories"""
        combined_stats = {
            'categories': [],
            'total_images': 0,
            'normal_samples': 0,
            'anomaly_samples': 0,
            'category_breakdown': {},
            'defect_types': set()
        }
        
        for dataset in self.datasets:
            stats = dataset.get_statistics()
            combined_stats['categories'].extend(stats['categories'])
            combined_stats['total_images'] += stats['total_images']
            combined_stats['normal_samples'] += stats['normal_samples']
            combined_stats['anomaly_samples'] += stats['anomaly_samples']
            combined_stats['category_breakdown'].update(stats['category_breakdown'])
            combined_stats['defect_types'].update(stats['defect_types'])
        
        combined_stats['categories'] = sorted(list(set(combined_stats['categories'])))
        combined_stats['defect_types'] = sorted(list(combined_stats['defect_types']))
        combined_stats['num_categories'] = len(combined_stats['categories'])
        combined_stats['imbalance_ratio'] = combined_stats['normal_samples'] / max(combined_stats['anomaly_samples'], 1)
        
        return combined_stats


def get_dataloaders(data_dir, category, batch_size, valid_ratio=0.2, 
                   train_transform=None, test_transform=None, 
                   img_size=256, load_masks=False, cache_images=False, 
                   num_workers=4, **kwargs):
    """Create data loaders for training, validation, and testing"""
    
    try:
        # Create datasets
        if isinstance(category, list):
            # Multi-category dataset
            train_dataset = MultiCategoryDataset(
                data_dir, category, "train", 
                transform=train_transform, img_size=img_size, 
                cache_images=cache_images
            )
            test_dataset = MultiCategoryDataset(
                data_dir, category, "test", 
                transform=test_transform, img_size=img_size, 
                load_masks=load_masks, cache_images=cache_images
            )
        else:
            # Single category dataset
            train_dataset = MVTecDataset(
                data_dir, category, "train", 
                transform=train_transform, img_size=img_size, 
                cache_images=cache_images
            )
            test_dataset = MVTecDataset(
                data_dir, category, "test", 
                transform=test_transform, img_size=img_size, 
                load_masks=load_masks, cache_images=cache_images
            )
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Split training data into train and validation sets
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    
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
    if isinstance(category, list):
        valid_dataset = MultiCategoryDataset(
            data_dir, category, "train",
            transform=test_transform, img_size=img_size
        )
    else:
        valid_dataset = MVTecDataset(
            data_dir, category, "train",
            transform=test_transform, img_size=img_size
        )
    
    valid_dataset = Subset(valid_dataset, valid_subset.indices)
    train_dataset = Subset(train_dataset, train_subset.indices)

    # DataLoader parameters
    params = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0
    }

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
        batch_size=min(batch_size, 32), 
        shuffle=False, 
        drop_last=False, 
        **params
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(batch_size, 32), 
        shuffle=False, 
        drop_last=False, 
        **params
    )

    return train_loader, valid_loader, test_loader


def analyze_dataset(data_dir, categories=None, save_path=None, plot_statistics=True):
    """Comprehensive dataset analysis"""
    
    if categories is None:
        # Auto-detect categories
        categories = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and 
                     not d.startswith('.')]
        categories = sorted(categories)
    
    print(f"Analyzing dataset with categories: {categories}")
    
    all_stats = {}
    
    for split in ['train', 'test']:
        print(f"\nAnalyzing {split} split...")
        
        if len(categories) == 1:
            dataset = MVTecDataset(data_dir, categories[0], split)
        else:
            dataset = MultiCategoryDataset(data_dir, categories, split)
        
        stats = dataset.get_statistics()
        img_stats = dataset.compute_image_statistics(num_samples=500)
        
        stats['image_statistics'] = img_stats
        all_stats[split] = stats
        
        # Print summary
        print(f"{split.upper()} Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Normal: {stats['normal_samples']}")
        print(f"  Anomaly: {stats['anomaly_samples']}")
        print(f"  Categories: {len(stats['categories'])}")
        print(f"  Defect types: {len(stats['defect_types'])}")
        
        if img_stats:
            print(f"  Image mean: {[f'{m:.3f}' for m in img_stats['mean']]}")
            print(f"  Image std: {[f'{s:.3f}' for s in img_stats['std']]}")
            print(f"  Most common size: {img_stats['image_sizes']['most_common_size']}")
    
    # Save statistics
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nDataset analysis saved to {save_path}")
    
    # Plot statistics
    if plot_statistics and len(categories) > 1:
        plot_dataset_statistics(all_stats)
    
    return all_stats


def plot_dataset_statistics(stats):
    """Plot dataset statistics"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Normal vs Anomaly by split
        splits = list(stats.keys())
        normal_counts = [stats[split]['normal_samples'] for split in splits]
        anomaly_counts = [stats[split]['anomaly_samples'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, normal_counts, width, label='Normal', alpha=0.8)
        axes[0, 0].bar(x + width/2, anomaly_counts, width, label='Anomaly', alpha=0.8)
        axes[0, 0].set_xlabel('Split')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].set_title('Normal vs Anomaly Samples by Split')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        
        # Plot 2: Category distribution (test set)
        test_stats = stats.get('test', stats.get('train', {}))
        if 'category_breakdown' in test_stats:
            categories = list(test_stats['category_breakdown'].keys())
            normal_by_cat = [test_stats['category_breakdown'][cat]['normal'] for cat in categories]
            anomaly_by_cat = [test_stats['category_breakdown'][cat]['anomaly'] for cat in categories]
            
            x = np.arange(len(categories))
            axes[0, 1].bar(x - width/2, normal_by_cat, width, label='Normal', alpha=0.8)
            axes[0, 1].bar(x + width/2, anomaly_by_cat, width, label='Anomaly', alpha=0.8)
            axes[0, 1].set_xlabel('Category')
            axes[0, 1].set_ylabel('Number of Samples')
            axes[0, 1].set_title('Samples by Category')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(categories, rotation=45)
            axes[0, 1].legend()
        
        # Plot 3: Image statistics - channel means
        if 'image_statistics' in test_stats and test_stats['image_statistics']:
            img_stats = test_stats['image_statistics']
            channels = ['Red', 'Green', 'Blue']
            means = img_stats['mean']
            stds = img_stats['std']
            
            x = np.arange(len(channels))
            axes[1, 0].bar(x, means, alpha=0.8, yerr=stds, capsize=5)
            axes[1, 0].set_xlabel('Channel')
            axes[1, 0].set_ylabel('Mean Pixel Value')
            axes[1, 0].set_title('Channel Statistics')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(channels)
        
        # Plot 4: Defect type distribution
        if 'category_breakdown' in test_stats:
            all_defect_types = set()
            for cat_info in test_stats['category_breakdown'].values():
                all_defect_types.update(cat_info['defect_types'])
            
            defect_types = sorted(list(all_defect_types))
            defect_counts = []
            
            for defect in defect_types:
                count = 0
                for cat_info in test_stats['category_breakdown'].values():
                    if defect in cat_info['defect_types']:
                        if defect == 'good':
                            count += cat_info['normal']
                        else:
                            count += cat_info['anomaly']
                defect_counts.append(count)
            
            axes[1, 1].bar(range(len(defect_types)), defect_counts, alpha=0.8)
            axes[1, 1].set_xlabel('Defect Type')
            axes[1, 1].set_ylabel('Number of Samples')
            axes[1, 1].set_title('Defect Type Distribution')
            axes[1, 1].set_xticks(range(len(defect_types)))
            axes[1, 1].set_xticklabels(defect_types, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    # Test dataset functionality
    data_dir = "path/to/mvtec/dataset"  # Update this path
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Please update the path.")
        print("Testing with dummy functionality...")
        
        # Test transforms
        print("\nTesting transforms:")
        train_transform, test_transform = get_transforms(
            img_size=256, normalize=True, augmentation_level='medium'
        )
        print(f"Train transform: {len(train_transform.transforms)} steps")
        print(f"Test transform: {len(test_transform.transforms)} steps")
        
        # Test OLED-specific transforms
        oled_train, oled_test = get_transforms(
            img_size=512, normalize=False, augmentation_level='oled'
        )
        print(f"OLED train transform: {len(oled_train.transforms)} steps")
        
    else:
        # Test with real data
        category = "bottle"
        batch_size = 16
        
        try:
            # Test data loading
            print("Testing data loading...")
            train_transform, test_transform = get_transforms(
                img_size=256, normalize=True, augmentation_level='medium'
            )
            
            train_loader, valid_loader, test_loader = get_dataloaders(
                data_dir, category, batch_size,
                train_transform=train_transform,
                test_transform=test_transform,
                load_masks=True
            )

            # Print dataset statistics
            for loader_name, loader in [("Train", train_loader), ("Valid", valid_loader), ("Test", test_loader)]:
                print(f"{loader_name} loader: {len(loader)} batches")
                
            # Test a single batch
            for batch in train_loader:
                print(f"Batch shapes: {batch['image'].shape}")
                print(f"Labels: {batch['label'].shape}")
                print(f"Categories: {batch['category'][:3]}...")  # Show first 3
                if 'mask' in batch:
                    print(f"Masks: {batch['mask'].shape}")
                break
            
            # Test dataset analysis
            print("\nTesting dataset analysis...")
            stats = analyze_dataset(data_dir, categories=[category], plot_statistics=False)
            
            print("\nTesting multi-category dataset...")
            multi_categories = ["bottle", "cable"]  # Adjust based on available categories
            try:
                multi_loader, _, _ = get_dataloaders(
                    data_dir, multi_categories, batch_size,
                    train_transform=train_transform,
                    test_transform=test_transform
                )
                print(f"Multi-category loader: {len(multi_loader)} batches")
            except Exception as e:
                print(f"Multi-category test failed: {e}")
                
        except Exception as e:
            print(f"Error in testing: {e}")

    print("\nMVTec dataset module test completed!")