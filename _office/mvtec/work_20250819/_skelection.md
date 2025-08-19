## 1. `mvtec.py`

**Classes**
```python
class MVTecDataset(data_dir, category, split, transform=None):
```

## 2. `models.py`

**Functions**
```python
def get_model(model_type, **model_params):
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

## 3. `metrics.py`

**Functions**
```python
# Factory Functions
def get_loss_fn(loss_type='combined'):
def get_metrics(metric_names=None):

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
def batch_psnr(pred, target, max_val=1.0)
def batch_ssim(pred, target, data_range=1.0)
```

## 4. `train.py`

**Functions**
```python
def set_seed(seed=42, device='cpu'):
def get_transforms(img_size=256):
def split_dataset(train_dataset, valid_dataset, valid_ratio=0.2, seed=42):
def get_dataloader(dataset, batch_size, split, **loader_params):

def get_dataloader(dataset, batch_size, split, **loader_params):
def get_optimizer(model, optimizer_type, **optim_params):
def get_scheduler(optimizer, scheduler_type, **scheduler_params):    

def train_epoch(model, data_loader, criterion, optimizer, metrics={})
def validate_epoch(model, data_loader, criterion, metrics={})
```

**Classes**
```python
class Trainer(model, optimizer, loss_fn, metrics={}):
```

## 5. `config.py`

**Functions**
```python
def load_config(config_path):
def show_config(config, show_all=False):
def get_config_prefix(config):
```

**Classes**
```python
class Config():
```

## 6. `main.py`

**Functions**
```python
def run(config):
def save_log(config):
```

**Clases**
```python
class Logger(log_path)
```

## 7. `evaluate.py`

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
    title="Precision-RecallCurve"):
def plot_anomaly_score_distribution(scores, labels, save_path=None,
    title="Anomaly Score Distribution"):
def plot_confusion_matrix(cm, save_path=None, title="Confusion Matrix"):
def plot_reconstruction_samples(images, reconstructions, scores, labels,
    num_samples=8, save_path=None):

def create_evaluation_report(results, save_dir=None, model_name="model"):
def quick_evaluate(model_path, test_loader, method="mse", device='cuda'):
def compare_methods(model_path, test_loader, methods=["mse", "ssim", "combined"]):
```
