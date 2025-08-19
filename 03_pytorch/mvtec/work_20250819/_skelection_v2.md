## 1. `mvtec.py`

**Classes**
```python
class MVTecDataset(data_dir, category, split, transform=None):
```

## 2. `models.py`

**Functions**
```python
def get_model(model_type, device='auto', **model_params):
def show_model_info(model):
def save_model(model, config):
def load_weights(model, model_path):
def load_trained_model(model_path, device='cuda'):
```

### 2-1. `autoencoder.py`

**Classes**
```python
# Building Blocks
class ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
    norm=True, activation='leaky_relu'):

class DeconvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
    norm=True, activation='relu', dropout=False):

# Vanilla AutoEncoder
class VanillaEncoder(in_channels=3, latent_dim=512):
class VanillaDecoder(out_channels=3, latent_dim=512):
class VanillaAE(in_channels=3, out_channels=3, latent_dim=512):

# UNet-style Autoencoder with Skip Connections
class UnetEncoder(in_channels=3, latent_dim=512):
class UnetDecoder(out_channels=3, latent_dim=512):
class UnetAE(in_channels=3, out_channels=3, latent_dim=512):
```

### 2-2. `autoencoder_v2.py`

**Classes**
```python
# Enhanced Building Blocks for Multi-Resolution Support
class AdaptiveConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
    norm=True, activation='leaky_relu', adaptive=False):

class AdaptiveDeconvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
    norm=True, activation='relu', dropout=False, target_size=None):

# Adaptive Vanilla AutoEncoder
class AdaptiveVanillaEncoder(in_channels=3, latent_dim=512, min_size=16):
class AdaptiveVanillaDecoder(out_channels=3, latent_dim=512, target_size=256):
class AdaptiveVanillaAE(in_channels=3, out_channels=3, latent_dim=512, target_size=256):

# Multi-Scale UNet AutoEncoder
class MultiScaleUnetEncoder(in_channels=3, latent_dim=512):
class MultiScaleUnetDecoder(out_channels=3, latent_dim=512, target_size=256):
class MultiScaleUnetAE(in_channels=3, out_channels=3, latent_dim=512, target_size=256):

# Patch-based AutoEncoder for Memory Efficiency
class PatchBasedEncoder(in_channels=3, latent_dim=512, patch_size=256, overlap=32):
class PatchBasedDecoder(out_channels=3, latent_dim=512, target_size=512, patch_size=256):
class PatchBasedAE(in_channels=3, out_channels=3, latent_dim=512, 
    target_size=512, patch_size=256, overlap=32):
```

## 3. `metrics.py`

**Functions**
```python
# Factory Functions
def get_loss_fn(loss_type='combined', **loss_params):
def get_metrics(metric_names=None, **metric_params):

# Basic Reconstruction Loss Functions
def mse_loss(pred, target, reduction='mean'):
def l1_loss(pred, target, reduction='mean'):
def l2_loss(pred, target, reduction='mean'):
def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
def charbonnier_loss(pred, target, epsilon=1e-3, reduction='mean'):

# Perceptual Quality Metrics
def psnr(pred, target, max_val=1.0):
def ssim(pred, target, data_range=1.0, size_average=True):
def ssim_loss(pred, target, data_range=1.0, size_average=True):
def ms_ssim(pred, target, data_range=1.0, size_average=True):
def ms_ssim_loss(pred, target, data_range=1.0, size_average=True):

# Classification Metrics
def binary_accuracy(pred, target, threshold=0.5):
def pixel_accuracy(pred, target, threshold=0.5):

# Advanced Loss Functions
def combined_loss(pred, target, l1_weight=0.7, ssim_weight=0.3):
def perceptual_l1_loss(pred, target, l1_weight=0.8, perceptual_weight=0.2):
def robust_loss(pred, target, alpha=0.2):
def focal_mse_loss(pred, target, alpha=2.0, gamma=2.0):
def edge_preserving_loss(pred, target, edge_weight=0.1):

# Batch-wise Metrics for Efficiency
def batch_psnr(pred, target, max_val=1.0):
def batch_ssim(pred, target, data_range=1.0):
```

## 4. `transform.py`

**Classes**
```python
# Custom Transform Classes for Multi-Resolution Support
class PreserveAspectRatioResize(max_size=1024, pad_mode='reflect', fill_value=0, square_output=True):
class AdaptiveSizeTransform(min_size=256, max_size=1024, size_multiple=32, 
    preserve_ratio=True, smart_resize=True):
class PatchBasedTransform(patch_size=256, overlap=32, max_patches=16, 
    min_coverage=0.8, adaptive_patches=True):
class MultiScaleTransform(scales=[1.0, 0.75, 0.5], target_size=256, fusion_method='concat'):
class SmartCropTransform(target_size=256, crop_method='center', 
    edge_threshold=0.1, variance_threshold=0.01):
```

**Functions**
```python
# Factory Function
def get_transforms(transform_type='adaptive_resize', **transform_params):

# Utility Functions
def calculate_optimal_size(height, width, max_size=1024, size_multiple=32, preserve_ratio=True):
def analyze_dataset_sizes(dataset, max_samples=1000):
def benchmark_transforms(img_tensor, transform_types=None, num_iterations=100):

# Internal Helper Functions
def _get_augmentation_transforms(**transform_params):
def _find_common_sizes(sizes):
def _recommend_transform_type(sizes, aspect_ratios):
```

## 5. `train.py`

**Functions**
```python
def set_seed(seed=42, device='cpu'):
def split_dataset(train_dataset, valid_dataset, valid_ratio=0.2, seed=42):
def get_dataloader(loader_type='train', **loader_params):
def get_optimizer(optimizer_type='adamw', **optimizer_params):
def get_scheduler(scheduler_type='plateau', **scheduler_params):

def train_epoch(model, data_loader, criterion, optimizer, metrics={}):
def validate_epoch(model, data_loader, criterion, metrics={}):
```

**Classes**
```python
class Trainer(model, optimizer, loss_fn, metrics={}):
```

## 6. `config.py`

**Functions**
```python
def load_config(config_path):
def show_config(config, show_all=False):
def get_config_prefix(config):
```

**Classes**
```python
class Config():
    # Data configuration
    data_dir: str
    category: str
    batch_size: int
    img_size: int
    valid_ratio: float
    
    # Model configuration
    in_channels: int
    out_channels: int
    latent_dim: int
    model_type: str
    
    # Training configuration
    device: str
    seed: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    
    # Factory function default types
    loss_type: str
    optimizer_type: str
    scheduler_type: str
    transform_type: str
    loader_type: str
    metric_names: List[str]
    
    # Enhanced transform settings
    min_size: int
    max_size: int
    preserve_ratio: bool
    aug_level: str
    size_multiple: int
    
    # Multi-resolution settings
    target_size: int
    patch_size: int
    overlap: int
    
    # Smart crop settings
    crop_method: str
    
    # Saving configuration
    save_model: bool
    save_log: bool
    model_path: str
    config_path: str
    
    # Early Stopping configuration
    fine_tuning: bool
    early_stop: bool
    early_stop_patience: int
    evaluation: bool
    
    include_keys: Dict[str, str]
    exclude_keys: List[str]
```

## 7. `main.py`

**Functions**
```python
def run(config):
def save_log(config):
```

**Classes**
```python
class Logger(log_path):
```

## 8. `evaluate.py`

**Functions**
```python
def compute_anomaly_scores(model, data_loader, method="mse"):
def compute_threshold(scores, labels, method="percentile", percentile=95):
def evaluate_anomaly_detection(model, test_loader, method="mse",
    threshold_method="percentile", percentile=95):
def evaluate_model(model, test_loader, method="mse", percentile=95):
def normalize_scores(scores, method='minmax'):

def show_results(results):
def save_results(results, save_path):

def plot_roc_curve(y_true, y_scores, save_path=None, title="ROC Curve"):
def plot_precision_recall_curve(y_true, y_scores, save_path=None,
    title="Precision-Recall Curve"):
def plot_anomaly_score_distribution(scores, labels, save_path=None,
    title="Anomaly Score Distribution"):
def plot_confusion_matrix(cm, save_path=None, title="Confusion Matrix"):
def plot_reconstruction_samples(images, reconstructions, scores, labels,
    num_samples=8, save_path=None):

def create_evaluation_report(results, save_dir=None, model_name="model"):
def quick_evaluate(model_path, test_loader, method="mse", device='cuda'):
def compare_methods(model_path, test_loader, methods=["mse", "ssim", "combined"]):
```

## Summary of Major Updates

### **Factory Functions Standardization**
All factory functions now follow the `get_XXX(XXX_type, **XXX_params)` pattern:
- `get_model(model_type, device='auto', **model_params)`
- `get_loss_fn(loss_type='combined', **loss_params)`
- `get_metrics(metric_names=None, **metric_params)`
- `get_transforms(transform_type='adaptive_resize', **transform_params)`
- `get_dataloader(loader_type='train', **loader_params)`
- `get_optimizer(optimizer_type='adamw', **optimizer_params)`
- `get_scheduler(scheduler_type='plateau', **scheduler_params)`

### **New Files Added**
1. **`transform.py`**: Comprehensive transform system with multi-resolution support
2. **`autoencoder_v2.py`**: Advanced autoencoder architectures for variable resolutions

### **Enhanced Model Support**
- **Adaptive models**: `AdaptiveVanillaAE`, `MultiScaleUnetAE`, `PatchBasedAE`
- **Multi-resolution processing**: Support for various OLED display sizes
- **Memory-efficient processing**: Patch-based approach for large images

### **Advanced Transform Capabilities**
- **Variable resolution support**: No fixed resize requirement
- **Aspect ratio preservation**: Intelligent padding and scaling
- **Content-aware processing**: Smart cropping and multi-scale transforms
- **Performance optimization**: v2 transforms with GPU acceleration

### **Configuration Enhancements**
- **Transform parameters**: `transform_type`, `min_size`, `max_size`, `preserve_ratio`
- **Multi-resolution settings**: `target_size`, `patch_size`, `overlap`
- **Augmentation control**: `aug_level`, `crop_method`
- **Factory defaults**: Centralized default values for all factory functions

### **Utility Functions**
- **Dataset analysis**: `analyze_dataset_sizes()` for optimal transform selection
- **Performance benchmarking**: `benchmark_transforms()` for speed comparison
- **Size optimization**: `calculate_optimal_size()` for autoencoder compatibility