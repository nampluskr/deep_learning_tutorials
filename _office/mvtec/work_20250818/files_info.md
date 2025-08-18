## 1. Datasets
#### `mvtec.py`: imported in `main.py` 
```python
def get_transforms(img_size=256, normalize=False):
def get_dataloaders(data_dir, category, batch_size, valid_ratio=0.2,
                    train_transform=None, test_transform=None):
class MVTecDataset(torch.utils.data.Dataset):

## (usage)
train_transform, test_trainsform = get_transforms(img_size, normalize)
train_loader, valid_loader, test_loader = get_dataloaders(
	data_dir, category, batch_size=32,
	train_transform=train_transform, test_transform=test_transform)
```

## 2. Models
#### `models.py`: imported in `main.py` 
```python
def get_model(model_type, **model_params):

class VanillaAE(nn.Module):   # model_type: "vanilla_ae"
class UnetAE(nn.Module):      # model_type: "unnet_ae"

## (usage)
model = get_model(model_type="vanilla_ae")
model = model.to(config.device)
```

## 3. Metrics
#### `metrics.py`: imported in `main.py` 
```python
def get_loss_fn(loss_type, **loss_params):
def get_metrics(metric_names=[]):

## (usage)
loss_fn = get_loss_fn(loss_type="combined")
metrics = get_metrics(metric_names=["psnr", "ssim"])
```

## 4. Training
#### `train.py`: imported in `main.py` 
```python
def set_seed(seed=42, device='cpu'):
def train_model(model, train_loader, config, valid_loader=None):
def save_model(model, model_path):
def loads_weights(model, model_path):

def train_epoch(model, data_loader, criterion, optimizer, metrics={}):
def validate_epoch(model, data_loader, criterion, metrics={}):
```

## 5. Running
#### `config.py`
```python
@dataclass
class Config:
def print_config(config, show_all=False):
def save_config(config_path):
def load_config(config_path):

def get_config_prefix(config):
```

#### `main.py`
```python
def main(config):
class Logger:
def save_log(config):

# (usage)
config_list = [config1, config2, config3]
for config in config_list:
	main(config)
```

```python
def main(config):
	if config.save_log:
		set_seed(seed=config.seed, device=config.device)
		save_log(config):

    # =====================================================================
    # 1. Data Loading
    # =====================================================================
	train_transform, test_transform = get_transforms(
        img_size=config.img_size,
        normalize=config.normalize
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        category=config.category,
        batch_size=config.batch_size,
        valid_ratio=config.valid_ratio,
        train_transform=train_transform,
        test_transform=test_transform
    )

	# =====================================================================
    # 2. Model Loading
    # =====================================================================
    model = get_model(
        config.model_type,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_dim=config.latent_dim
    )
    model = model.to(config.device)

    # =====================================================================
    # 3. Model Training with Validation
    # =====================================================================
    start_time = time()
    history = train_model(model, train_loader, config, 
		valid_loader=valid_loader)
    elapsed_time = time() - start_time
    show_history(history)

    # =====================================================================
    # 4. Fine-tuning on Validation Data
    # =====================================================================
    if config.fine_tuning and valid_loader is not None:
        fine_tune_config = replace(config, num_epochs=5)
        train_model(model, valid_loader, fine_tune_config)

    # =====================================================================
    # 5. Evaluate Anomaly Detection Performance on Test Data
    # =====================================================================
    if config.evaluation and test_loader is not None:
        results = evaluate_model(model, test_loader, method='mse', 
			percentile=95)
		show_results(results)

    # =====================================================================
    # 6. Save Model
    # =====================================================================
    if config.save_model:
        save_model(model, config)
```

## 6. Evaluation
`evaluate.py`: imported in jupyter notebooks
```python
def evaluate_model(model, test_loader, method="mse", percentile=95):

# Metrics and score funcrions for anomaly detection
```
