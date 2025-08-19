"""
Enhanced image transformations for multi-resolution anomaly detection
Supports various OLED display resolutions with adaptive processing
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import math
import warnings
from typing import Tuple, Union, Optional, List


# =============================================================================
# Custom Transform Classes for Multi-Resolution Support
# =============================================================================

class PreserveAspectRatioResize:
    """Resize while preserving aspect ratio with intelligent padding"""
    
    def __init__(self, max_size=1024, pad_mode='reflect', fill_value=0, square_output=True):
        """
        Args:
            max_size: Maximum dimension size
            pad_mode: Padding mode ('constant', 'reflect', 'edge', 'symmetric')
            fill_value: Fill value for constant padding
            square_output: Whether to output square images
        """
        self.max_size = max_size
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.square_output = square_output
    
    def __call__(self, img):
        _, h, w = img.shape
        
        # Calculate scaling factor to fit within max_size
        scale = min(self.max_size / h, self.max_size / w)
        
        if scale < 1.0:
            # Resize if image is too large
            new_h, new_w = int(h * scale), int(w * scale)
            img = v2.Resize((new_h, new_w), antialias=True)(img)
            h, w = new_h, new_w
        
        # Add padding to make it square (if requested)
        if self.square_output:
            target_size = max(h, w)
            pad_h = target_size - h
            pad_w = target_size - w
        else:
            pad_h = max(0, self.max_size - h)
            pad_w = max(0, self.max_size - w)
        
        if pad_h > 0 or pad_w > 0:
            # Symmetric padding
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if self.pad_mode == 'constant':
                img = v2.Pad([pad_left, pad_top, pad_right, pad_bottom], 
                            fill=self.fill_value)(img)
            else:
                # Use PyTorch's F.pad for non-constant padding modes
                img = F.pad(img, [pad_left, pad_right, pad_top, pad_bottom], 
                           mode=self.pad_mode)
        
        return img


class AdaptiveSizeTransform:
    """Dynamically adjust image size based on content and autoencoder constraints"""
    
    def __init__(self, min_size=256, max_size=1024, size_multiple=32, 
                 preserve_ratio=True, smart_resize=True):
        """
        Args:
            min_size: Minimum dimension size
            max_size: Maximum dimension size
            size_multiple: Size must be multiple of this (for autoencoder compatibility)
            preserve_ratio: Whether to preserve aspect ratio
            smart_resize: Use intelligent resizing based on content
        """
        self.min_size = min_size
        self.max_size = max_size
        self.size_multiple = size_multiple
        self.preserve_ratio = preserve_ratio
        self.smart_resize = smart_resize
    
    def __call__(self, img):
        _, h, w = img.shape
        
        # Calculate optimal dimensions
        new_h, new_w = self._calculate_optimal_size(h, w)
        
        # Resize if dimensions changed
        if new_h != h or new_w != w:
            img = v2.Resize((new_h, new_w), antialias=True)(img)
        
        return img
    
    def _calculate_optimal_size(self, h, w):
        """Calculate optimal dimensions based on constraints"""
        
        if self.preserve_ratio:
            # Preserve aspect ratio
            scale = 1.0
            
            # Ensure minimum size
            if h < self.min_size or w < self.min_size:
                scale = max(self.min_size / h, self.min_size / w)
            
            # Ensure maximum size
            if h * scale > self.max_size or w * scale > self.max_size:
                scale = min(self.max_size / h, self.max_size / w)
            
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            # Independent scaling
            new_h = max(self.min_size, min(self.max_size, h))
            new_w = max(self.min_size, min(self.max_size, w))
        
        # Round to nearest multiple for autoencoder compatibility
        new_h = self._round_to_multiple(new_h, self.size_multiple)
        new_w = self._round_to_multiple(new_w, self.size_multiple)
        
        return new_h, new_w
    
    def _round_to_multiple(self, value, multiple):
        """Round value to nearest multiple"""
        return ((value + multiple - 1) // multiple) * multiple


class PatchBasedTransform:
    """Transform for patch-based processing of large images"""
    
    def __init__(self, patch_size=256, overlap=32, max_patches=16, 
                 min_coverage=0.8, adaptive_patches=True):
        """
        Args:
            patch_size: Size of each patch
            overlap: Overlap between patches
            max_patches: Maximum number of patches to extract
            min_coverage: Minimum coverage ratio for patch extraction
            adaptive_patches: Whether to adapt patch extraction based on image size
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_patches = max_patches
        self.min_coverage = min_coverage
        self.adaptive_patches = adaptive_patches
        self.stride = patch_size - overlap
    
    def __call__(self, img):
        _, h, w = img.shape
        
        # If image is smaller than patch size, pad it
        if h <= self.patch_size and w <= self.patch_size:
            return self._pad_small_image(img)
        
        # For large images, return original (patch extraction handled elsewhere)
        # or extract representative patches
        if self.adaptive_patches:
            return self._extract_representative_patches(img)
        
        return img
    
    def _pad_small_image(self, img):
        """Pad small images to patch size"""
        _, h, w = img.shape
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            img = F.pad(img, [pad_left, pad_right, pad_top, pad_bottom], 
                       mode='reflect')
        
        return img
    
    def _extract_representative_patches(self, img):
        """Extract representative patches from large images"""
        # For now, return the center crop
        _, h, w = img.shape
        
        # Center crop to patch size
        start_h = (h - self.patch_size) // 2
        start_w = (w - self.patch_size) // 2
        
        if start_h >= 0 and start_w >= 0:
            return img[:, start_h:start_h + self.patch_size, 
                      start_w:start_w + self.patch_size]
        
        return img


class MultiScaleTransform:
    """Multi-scale processing transform for enhanced feature extraction"""
    
    def __init__(self, scales=[1.0, 0.75, 0.5], target_size=256, 
                 fusion_method='concat'):
        """
        Args:
            scales: List of scale factors
            target_size: Target size for all scales
            fusion_method: How to fuse multi-scale features ('concat', 'average')
        """
        self.scales = scales
        self.target_size = target_size
        self.fusion_method = fusion_method
    
    def __call__(self, img):
        """Process image at multiple scales"""
        _, h, w = img.shape
        
        if self.fusion_method == 'concat':
            # Return original image (multi-scale processing in model)
            return img
        elif self.fusion_method == 'average':
            # Create averaged multi-scale representation
            scaled_imgs = []
            
            for scale in self.scales:
                scale_h, scale_w = int(h * scale), int(w * scale)
                scaled = v2.Resize((scale_h, scale_w), antialias=True)(img)
                scaled = v2.Resize((self.target_size, self.target_size), 
                                 antialias=True)(scaled)
                scaled_imgs.append(scaled)
            
            # Average the scaled images
            avg_img = torch.stack(scaled_imgs).mean(dim=0)
            return avg_img
        
        return img


class SmartCropTransform:
    """Intelligent cropping based on content analysis"""
    
    def __init__(self, target_size=256, crop_method='center', 
                 edge_threshold=0.1, variance_threshold=0.01):
        """
        Args:
            target_size: Target crop size
            crop_method: Cropping method ('center', 'random', 'smart')
            edge_threshold: Threshold for edge detection in smart cropping
            variance_threshold: Minimum variance threshold for content detection
        """
        self.target_size = target_size
        self.crop_method = crop_method
        self.edge_threshold = edge_threshold
        self.variance_threshold = variance_threshold
    
    def __call__(self, img):
        _, h, w = img.shape
        
        if h <= self.target_size and w <= self.target_size:
            return img
        
        if self.crop_method == 'center':
            return self._center_crop(img)
        elif self.crop_method == 'smart':
            return self._smart_crop(img)
        elif self.crop_method == 'random':
            return self._random_crop(img)
        
        return self._center_crop(img)
    
    def _center_crop(self, img):
        """Standard center crop"""
        _, h, w = img.shape
        start_h = (h - self.target_size) // 2
        start_w = (w - self.target_size) // 2
        
        return img[:, start_h:start_h + self.target_size, 
                  start_w:start_w + self.target_size]
    
    def _smart_crop(self, img):
        """Content-aware smart cropping"""
        # Simple implementation: find region with highest variance
        _, h, w = img.shape
        
        best_crop = None
        best_variance = 0
        
        # Try multiple crop positions
        step = min(32, (h - self.target_size) // 4, (w - self.target_size) // 4)
        step = max(1, step)
        
        for i in range(0, h - self.target_size + 1, step):
            for j in range(0, w - self.target_size + 1, step):
                crop = img[:, i:i + self.target_size, j:j + self.target_size]
                variance = torch.var(crop).item()
                
                if variance > best_variance:
                    best_variance = variance
                    best_crop = crop
        
        return best_crop if best_crop is not None else self._center_crop(img)
    
    def _random_crop(self, img):
        """Random crop (for training augmentation)"""
        _, h, w = img.shape
        
        max_start_h = h - self.target_size
        max_start_w = w - self.target_size
        
        start_h = torch.randint(0, max_start_h + 1, (1,)).item()
        start_w = torch.randint(0, max_start_w + 1, (1,)).item()
        
        return img[:, start_h:start_h + self.target_size, 
                  start_w:start_w + self.target_size]


# =============================================================================
# Factory Function for Transform Selection
# =============================================================================

def get_transforms(transform_type='adaptive_resize', **transform_params):
    """Enhanced transforms supporting variable resolutions and aspect ratios"""
    
    available_types = [
        'preserve_ratio',    # Preserve aspect ratio with padding
        'adaptive_resize',   # Smart adaptive resizing
        'patch_based',      # Patch-based processing
        'multi_scale',      # Multi-scale processing
        'smart_crop',       # Content-aware cropping
        'fixed_size'        # Legacy fixed size (backward compatibility)
    ]
    
    if transform_type not in available_types:
        raise ValueError(f"Unknown transform type: {transform_type}. Available: {available_types}")
    
    # Common parameters
    device = transform_params.get('device', None)
    dtype = transform_params.get('dtype', torch.float32)
    
    # Base transforms based on type
    base_transforms = [v2.ConvertImageDtype(dtype)]
    
    if transform_type == 'preserve_ratio':
        max_size = transform_params.get('max_size', 1024)
        pad_mode = transform_params.get('pad_mode', 'reflect')
        square_output = transform_params.get('square_output', True)
        
        base_transforms.append(
            PreserveAspectRatioResize(
                max_size=max_size,
                pad_mode=pad_mode,
                square_output=square_output
            )
        )
        
    elif transform_type == 'adaptive_resize':
        min_size = transform_params.get('min_size', 256)
        max_size = transform_params.get('max_size', 1024)
        size_multiple = transform_params.get('size_multiple', 32)
        preserve_ratio = transform_params.get('preserve_ratio', True)
        
        base_transforms.append(
            AdaptiveSizeTransform(
                min_size=min_size,
                max_size=max_size,
                size_multiple=size_multiple,
                preserve_ratio=preserve_ratio
            )
        )
        
    elif transform_type == 'patch_based':
        patch_size = transform_params.get('patch_size', 256)
        overlap = transform_params.get('overlap', 32)
        max_patches = transform_params.get('max_patches', 16)
        
        base_transforms.append(
            PatchBasedTransform(
                patch_size=patch_size,
                overlap=overlap,
                max_patches=max_patches
            )
        )
        
    elif transform_type == 'multi_scale':
        scales = transform_params.get('scales', [1.0, 0.75, 0.5])
        target_size = transform_params.get('target_size', 256)
        fusion_method = transform_params.get('fusion_method', 'concat')
        
        base_transforms.append(
            MultiScaleTransform(
                scales=scales,
                target_size=target_size,
                fusion_method=fusion_method
            )
        )
        
    elif transform_type == 'smart_crop':
        target_size = transform_params.get('target_size', 256)
        crop_method = transform_params.get('crop_method', 'smart')
        
        base_transforms.append(
            SmartCropTransform(
                target_size=target_size,
                crop_method=crop_method
            )
        )
        
    elif transform_type == 'fixed_size':
        # Legacy mode for backward compatibility
        img_size = transform_params.get('img_size', 256)
        if isinstance(img_size, int):
            target_size = (img_size, img_size)
        else:
            target_size = img_size
            
        base_transforms.append(
            v2.Resize(target_size, antialias=True)
        )
    
    # Augmentation transforms (applied only to training)
    aug_transforms = _get_augmentation_transforms(**transform_params)
    
    # Combine transforms
    train_transforms = base_transforms + aug_transforms
    test_transforms = base_transforms.copy()
    
    # Add device transfer if specified
    if device is not None:
        # Note: v2 transforms can work on GPU, but typically images are moved in DataLoader
        pass
    
    # Logging
    print(f"Creating enhanced transforms: type={transform_type}")
    if transform_params:
        key_params = {k: v for k, v in transform_params.items() 
                     if k in ['min_size', 'max_size', 'target_size', 'patch_size']}
        print(f"Key parameters: {key_params}")
    print(f"Train augmentations: {len(aug_transforms)} transforms")
    
    return v2.Compose(train_transforms), v2.Compose(test_transforms)


def _get_augmentation_transforms(**transform_params):
    """Get augmentation transforms based on parameters"""
    
    aug_level = transform_params.get('aug_level', 'default')
    disable_augmentation = transform_params.get('disable_augmentation', False)
    
    if disable_augmentation:
        return []
    
    # Custom augmentation parameters
    h_flip_p = transform_params.get('h_flip_p', None)
    v_flip_p = transform_params.get('v_flip_p', None)
    rotation_degrees = transform_params.get('rotation_degrees', None)
    brightness = transform_params.get('brightness', None)
    contrast = transform_params.get('contrast', None)
    blur_p = transform_params.get('blur_p', 0.0)
    erasing_p = transform_params.get('erasing_p', 0.0)
    
    aug_transforms = []
    
    if aug_level == 'none':
        return []
    elif aug_level == 'light':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=h_flip_p or 0.3),
            v2.RandomRotation(degrees=rotation_degrees or 5),
            v2.ColorJitter(brightness=brightness or 0.05, 
                          contrast=contrast or 0.05),
        ]
    elif aug_level == 'default':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=h_flip_p or 0.5),
            v2.RandomVerticalFlip(p=v_flip_p or 0.3),
            v2.RandomRotation(degrees=rotation_degrees or 10),
            v2.ColorJitter(brightness=brightness or 0.1, 
                          contrast=contrast or 0.1, 
                          saturation=0.05, hue=0.02),
        ]
    elif aug_level == 'heavy':
        aug_transforms = [
            v2.RandomHorizontalFlip(p=h_flip_p or 0.6),
            v2.RandomVerticalFlip(p=v_flip_p or 0.4),
            v2.RandomRotation(degrees=rotation_degrees or 15),
            v2.ColorJitter(brightness=brightness or 0.15, 
                          contrast=contrast or 0.15, 
                          saturation=0.1, hue=0.05),
            v2.RandomPerspective(distortion_scale=0.1, p=0.2),
        ]
    elif aug_level == 'custom':
        # Build custom augmentation from individual parameters
        if h_flip_p and h_flip_p > 0:
            aug_transforms.append(v2.RandomHorizontalFlip(p=h_flip_p))
        if v_flip_p and v_flip_p > 0:
            aug_transforms.append(v2.RandomVerticalFlip(p=v_flip_p))
        if rotation_degrees and rotation_degrees > 0:
            aug_transforms.append(v2.RandomRotation(degrees=rotation_degrees))
        if brightness or contrast:
            aug_transforms.append(v2.ColorJitter(
                brightness=brightness or 0,
                contrast=contrast or 0
            ))
    
    # Add optional augmentations
    if blur_p > 0:
        kernel_size = transform_params.get('blur_kernel', 3)
        sigma = transform_params.get('blur_sigma', (0.1, 0.5))
        aug_transforms.append(
            v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p=blur_p)
        )
    
    if erasing_p > 0:
        aug_transforms.append(v2.RandomErasing(p=erasing_p))
    
    return aug_transforms


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_optimal_size(height, width, max_size=1024, size_multiple=32, 
                          preserve_ratio=True):
    """Calculate optimal size for autoencoder processing"""
    
    if preserve_ratio:
        scale = min(max_size / height, max_size / width)
        new_h, new_w = int(height * scale), int(width * scale)
    else:
        new_h = min(max_size, height)
        new_w = min(max_size, width)
    
    # Round to nearest multiple
    new_h = ((new_h + size_multiple - 1) // size_multiple) * size_multiple
    new_w = ((new_w + size_multiple - 1) // size_multiple) * size_multiple
    
    return new_h, new_w


def analyze_dataset_sizes(dataset, max_samples=1000):
    """Analyze dataset size distribution for optimal transform selection"""
    
    sizes = []
    aspect_ratios = []
    
    # Sample a subset for analysis
    sample_indices = torch.randperm(len(dataset))[:max_samples]
    
    print("Analyzing dataset sizes...")
    for i in sample_indices:
        try:
            item = dataset[i]
            if isinstance(item, dict) and 'image' in item:
                img = item['image']
            else:
                img = item[0] if isinstance(item, (list, tuple)) else item
            
            if len(img.shape) == 3:
                _, h, w = img.shape
            else:
                h, w = img.shape[-2:]
            
            sizes.append((h, w))
            aspect_ratios.append(w / h)
            
        except Exception as e:
            continue
    
    if not sizes:
        warnings.warn("Could not analyze dataset sizes")
        return None
    
    sizes = torch.tensor(sizes, dtype=torch.float32)
    aspect_ratios = torch.tensor(aspect_ratios)
    
    analysis = {
        'num_samples': len(sizes),
        'height_stats': {
            'min': sizes[:, 0].min().item(),
            'max': sizes[:, 0].max().item(),
            'mean': sizes[:, 0].mean().item(),
            'std': sizes[:, 0].std().item(),
        },
        'width_stats': {
            'min': sizes[:, 1].min().item(),
            'max': sizes[:, 1].max().item(),
            'mean': sizes[:, 1].mean().item(),
            'std': sizes[:, 1].std().item(),
        },
        'aspect_ratio_stats': {
            'min': aspect_ratios.min().item(),
            'max': aspect_ratios.max().item(),
            'mean': aspect_ratios.mean().item(),
            'std': aspect_ratios.std().item(),
        },
        'common_sizes': _find_common_sizes(sizes),
        'recommended_transform': _recommend_transform_type(sizes, aspect_ratios)
    }
    
    return analysis


def _find_common_sizes(sizes):
    """Find most common image sizes in dataset"""
    unique_sizes, counts = torch.unique(sizes, dim=0, return_counts=True)
    
    # Get top 5 most common sizes
    top_indices = torch.argsort(counts, descending=True)[:5]
    common_sizes = []
    
    for idx in top_indices:
        h, w = unique_sizes[idx]
        count = counts[idx]
        common_sizes.append({
            'size': (int(h), int(w)),
            'count': int(count),
            'percentage': (count / len(sizes) * 100).item()
        })
    
    return common_sizes


def _recommend_transform_type(sizes, aspect_ratios):
    """Recommend optimal transform type based on dataset analysis"""
    
    max_size = max(sizes.max().item(), 1024)
    aspect_ratio_var = aspect_ratios.var().item()
    
    if aspect_ratio_var > 0.1:
        # High aspect ratio variation
        if max_size > 1024:
            return 'patch_based'
        else:
            return 'preserve_ratio'
    else:
        # Consistent aspect ratios
        if max_size > 1024:
            return 'adaptive_resize'
        else:
            return 'fixed_size'


def benchmark_transforms(img_tensor, transform_types=None, num_iterations=100):
    """Benchmark different transform types for performance comparison"""
    
    if transform_types is None:
        transform_types = ['fixed_size', 'adaptive_resize', 'preserve_ratio']
    
    results = {}
    
    print(f"Benchmarking transforms on image size: {img_tensor.shape}")
    
    for transform_type in transform_types:
        try:
            train_transform, _ = get_transforms(transform_type)
            
            # Warm up
            for _ in range(10):
                _ = train_transform(img_tensor)
            
            # Benchmark
            import time
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = train_transform(img_tensor)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            
            results[transform_type] = {
                'avg_time_ms': avg_time,
                'throughput_fps': 1000 / avg_time
            }
            
        except Exception as e:
            results[transform_type] = {'error': str(e)}
    
    return results


# =============================================================================
# Testing and Examples
# =============================================================================

if __name__ == "__main__":
    # Test different transform types with various input sizes
    test_sizes = [(256, 256), (512, 512), (1024, 768), (768, 1024), (2048, 1536)]
    
    print("Testing Enhanced Transform System")
    print("=" * 50)
    
    for h, w in test_sizes:
        print(f"\nTesting with input size: {h}x{w}")
        test_img = torch.randn(3, h, w)
        
        # Test different transform types
        transform_types = ['adaptive_resize', 'preserve_ratio', 'smart_crop']
        
        for transform_type in transform_types:
            try:
                train_transform, test_transform = get_transforms(
                    transform_type, 
                    max_size=512,
                    aug_level='light'
                )
                
                # Apply transforms
                train_result = train_transform(test_img)
                test_result = test_transform(test_img)
                
                print(f"  {transform_type:15s}: {test_img.shape} -> "
                      f"Train: {train_result.shape}, Test: {test_result.shape}")
                
            except Exception as e:
                print(f"  {transform_type:15s}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("Transform system ready for use!")