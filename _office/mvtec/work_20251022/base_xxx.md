```python
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class BaseDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None, **kwargs):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.load_data(root_dir, split, **kwargs)

    def load_data(self, root_dir, split, **kwargs):
        raise NotImplementedError("Dataset must implement load_data()")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return dict(image=image, label=label)

class AnomalyDataset(BaseDataset):
    def __init__(self, root_dir, split=None, transform=None, mask_transform=None, **kwargs):
        self.mask_transform = mask_transform
        self.mask_paths = []
        self.defect_types = []
        self.categories = []
        self.has_mask = True

        super().__init__(root_dir, split, transform, **kwargs)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()

        mask_path = self.mask_paths[idx]
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = (np.array(mask) > 0).astype(np.uint8)
            mask = Image.fromarray(mask * 255, mode='L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            if self.mask_transform:
                dummy_mask = Image.new('L', (width, height), 0)
                mask = self.mask_transform(dummy_mask)
            else:
                mask = torch.zeros(1, height, width)

        defect_type = self.defect_types[idx]
        category = self.categories[idx]
        has_mask = torch.tensor(self.has_mask).bool()

        return dict(image=image, label=label, mask=mask, defect_type=defect_type, 
            category=category, has_mask=has_mask)

class MVTecDataset(AnomalyDataset):
    def load_data(self, root_dir, split="train", dataset_type="mvtec", category="bottle", **kwargs):
        self.has_mask = True
        category_dir = os.path.join(root_dir, dataset_type, category)

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "good")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

        elif split == "test":
            normal_image_dir = os.path.join(category_dir, "test", "good")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

            test_dir = os.path.join(category_dir, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                if defect_type == "good": continue

                anomaly_image_dir = os.path.join(test_dir, defect_type)
                anomaly_mask_dir = os.path.join(category_dir, "ground_truth", defect_type)
                for image_name in sorted(os.listdir(anomaly_image_dir)):
                    anomaly_image_path = os.path.join(anomaly_image_dir, image_name)
                    image_stem = os.path.splitext(image_name)[0]
                    anomaly_mask_path = os.path.join(anomaly_mask_dir, f"{image_stem}_mask.png")
                    self.image_paths.append(anomaly_image_path)
                    self.mask_paths.append(anomaly_mask_path)
                    self.labels.append(1)
                    self.defect_types.append(defect_type)
                    self.categories.append(category)
        else:
            raise ValueError(f" >> Invalid split: {split}. Expected 'train' or 'test'")

class Transform:
    def __init__(self, split="train", img_size=256, normalize=True):
        transforms = []
        if split == "train":
            transforms.extend([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(
                    brightness=(0.8, 1.2),  # Brightness factor between 0.8 and 1.2
                    contrast=(0.7, 1.3),    # Contrast factor between 0.7 and 1.3
                    saturation=(0.7, 1.3),  # Saturation factor between 0.7 and 1.3
                    hue=(-0.1, 0.1)         # Hue Bshift between -0.1 and 0.1
                ),
                T.ToTensor(),
            ])
        else:
            transforms.extend([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])

        if normalize and split != "mask":
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transforms = T.Compose(transforms)

    def __call__(self, x):
        return self.transforms(x)

class CollateFunction:
    def __init__(self, task="default"):
        pass

train_loader = DataLoader(
    dataset=MVTecDataset(root_dir="/home/namu/myspace/NAMU/datasets",
        split="train", dataset_type="mvtec", category="bottle",
        transform=Transform(split="train", normalize=True),
        mask_transform=Transform(split="mask", normalize=False)),
    batch_size=32, shuffle=True, drop_last=True,
    num_workers=8, pin_memory=True, persistent_workers=False)

test_loader = DataLoader(
    dataset=MVTecDataset(root_dir="/home/namu/myspace/NAMU/datasets",
        split="test", dataset_type="mvtec", category="bottle",
        transform=Transform(split="test", normalize=True),
        mask_transform=Transform(split="mask", normalize=False)),
    batch_size=32, shuffle=False, drop_last=False,
    num_workers=8, pin_memory=True, persistent_workers=False)
```

```python
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')

class BaseTrainer:
    def on_fit_start(self): pass
    def on_fit_end(self): pass

    def on_epoch_start(self): pass
    def on_epoch_end(self): pass

    def on_train_epoch_start(self):
        self.model.train()

    def on_train_epoch_end(self, train_results): pass
    def on_train_batch_start(self, batch, batch_idx): pass

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement training_step()")

    def on_before_backward(self, loss): pass
    def on_after_backward(self): pass

    def on_before_optimizer_step(self, optimizer): pass
    def on_train_batch_end(self, outputs, batch, batch_idx): pass

    def on_validation_epoch_start(self):
        self.model.eval()

    def on_validation_epoch_end(self, valid_results): pass

    def on_validation_batch_start(self, batch, batch_idx): pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement validation_step()")

    def on_validation_batch_end(self, outputs, batch, batch_idx): pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import time
import os
from collections import defaultdict

class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        metrics: Dict[str, Callable] = None,
        device: torch.device = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_config: Dict[str, Any] = None,  # 예: {"interval": "epoch", "frequency": 1}
        early_stopper_loss: Callable = None,
        early_stopper_metric: Callable = None,
        output_dir: str = "./results",
        log_to_file: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            raise ValueError("Model must be provided")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch", "frequency": 1}
        self.early_stopper_loss = early_stopper_loss  # e.g., EarlyStopping(loss 기준)
        self.early_stopper_metric = early_stopper_metric  # e.g., EarlyStopping(metric 기준)

        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.metrics = {}
        if metrics:
            for name, metric_fn in metrics.items():
                self.metrics[name] = metric_fn.to(self.device) if isinstance(metric_fn, nn.Module) else metric_fn

        self.eval_period = 1
        self.history = defaultdict(list)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = self._setup_logger(output_dir, log_to_file)

        # 상태 변수
        self.fit_start_time = None
        self.epoch_start_time = None
        self.current_epoch = 0
        self.max_epochs = None
        self.global_step = 0
        self.train_loader = None
        self.valid_loader = None

    def _setup_logger(self, output_dir: str, log_to_file: bool):
        import logging
        logger = logging.getLogger(f"Trainer_{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            if log_to_file:
                fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        return logger

    def on_fit_start(self):
        self.logger.info("Starting training...")
        if self.early_stopper_loss: self.early_stopper_loss.reset()
        if self.early_stopper_metric: self.early_stopper_metric.reset()

    def on_fit_end(self):
        self.logger.info("Training completed.")
        final_time = time.time() - self.fit_start_time
        self.logger.info(f"Total training time: {final_time:.2f}s")

    def on_epoch_start(self):
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {self.current_epoch + 1}/{self.max_epochs}:")

    def on_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.logger.info(f"Epoch {self.current_epoch + 1} completed in {epoch_time:.2f}s")

    def on_train_epoch_start(self):
        self.model.train()

    def on_train_epoch_end(self, train_results: Dict[str, float]):
        log_str = " | ".join([f"train_{k}: {v:.4f}" for k, v in train_results.items()])
        self.logger.info(f"Train Results: {log_str}")

    def on_train_batch_start(self, batch: Dict[str, Any], batch_idx: int):
        pass

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_idx: int):
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclass must implement training_step()")

    def on_before_backward(self, loss: torch.Tensor):
        pass

    def on_after_backward(self):
        pass

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        optimizer.step()

    def on_validation_epoch_start(self):
        self.model.eval()

    def on_validation_epoch_end(self, valid_results: Dict[str, float]):
        log_str = " | ".join([f"val_{k}: {v:.4f}" for k, v in valid_results.items()])
        self.logger.info(f"Validation Results: {log_str}")

    def on_validation_batch_start(self, batch: Dict[str, Any], batch_idx: int):
        pass

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_idx: int):
        pass

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclass must implement validation_step()")

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        def to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            elif isinstance(obj, dict):
                return {k: to_device(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_device(x) for x in obj]
            else:
                return obj
        return to_device(batch)

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        metric_results = {}
        for name, metric_fn in self.metrics.items():
            try:
                metric_results[name] = metric_fn(preds, targets)
            except Exception as e:
                metric_results[name] = torch.tensor(float("nan"))
                self.logger.warning(f"Metric {name} failed: {e}")
        return metric_results

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        valid_loader: Optional[DataLoader] = None,
        output_dir: str = None,
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.max_epochs = num_epochs
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        self.on_fit_start()
        self.fit_start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.on_epoch_start()
            self.on_train_epoch_start()

            train_results = self._run_train_epoch()
            self.on_train_epoch_end(train_results)

            if valid_loader and (epoch + 1) % self.eval_period == 0:
                self.on_validation_epoch_start()
                valid_results = self._run_validation_epoch()
                self.on_validation_epoch_end(valid_results)

                if self.early_stopper_loss:
                    current_loss = valid_results.get("loss", None)
                    if current_loss is not None:
                        self.early_stopper_loss(current_loss)
                        if self.early_stopper_loss.early_stop:
                            self.logger.info("Early stopping triggered (loss).")
                            break

                if self.early_stopper_metric:
                    metric_val = next(iter(valid_results.values()))
                    self.early_stopper_metric(metric_val)
                    if self.early_stopper_metric.early_stop:
                        self.logger.info("Early stopping triggered (metric).")
                        break

            if self.scheduler:
                interval = self.scheduler_config.get("interval", "epoch")
                frequency = self.scheduler_config.get("frequency", 1)
                if interval == "epoch" and (epoch + 1) % frequency == 0:
                    self.scheduler.step()

            self.on_epoch_end()

        self.on_fit_end()

    def _run_train_epoch(self) -> Dict[str, float]:
        epoch_outputs = []
        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)
            self.on_train_batch_start(batch, batch_idx)

            self.optimizer.zero_grad()
            outputs = self.training_step(batch, batch_idx)

            loss = outputs["loss"]
            self.on_before_backward(loss)
            loss.backward()
            self.on_after_backward()
            self.on_before_optimizer_step(self.optimizer)

            self.global_step += 1
            self.on_train_batch_end(outputs, batch, batch_idx)
            epoch_outputs.append({k: v.item() if isinstance(v, torch.Tensor) else v
                                  for k, v in outputs.items()})

        # 에폭 평균 계산
        avg_results = {k: np.mean([out[k] for out in epoch_outputs]) for k in epoch_outputs[0].keys()}
        for k, v in avg_results.items():
            self.history[f"train_{k}"].append(v)
        return avg_results

    @torch.no_grad()
    def _run_validation_epoch(self) -> Dict[str, float]:
        epoch_outputs = []
        for batch_idx, batch in enumerate(self.valid_loader):
            batch = self._move_batch_to_device(batch)
            self.on_validation_batch_start(batch, batch_idx)
            outputs = self.validation_step(batch, batch_idx)
            self.on_validation_batch_end(outputs, batch, batch_idx)
            epoch_outputs.append({k: v.item() if isinstance(v, torch.Tensor) else v
                                  for k, v in outputs.items()})

        avg_results = {k: np.mean([out[k] for out in epoch_outputs]) for k in epoch_outputs[0].keys()}
        for k, v in avg_results.items():
            self.history[f"val_{k}"].append(v)
        return avg_results

    def save_checkpoint(self, filepath: str, save_optimizer: bool = True):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, strict: bool = True):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info(f"Checkpoint loaded: {filepath}")

# --------------------------------------------------------------
#  BaseTrainer (Pure PyTorch, Lightning-style API)
# --------------------------------------------------------------
import os
import time
import logging
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# --------------------------------------------------------------
# 1) Trainer 설정을 한 번에 관리하기 위한 dataclass
# --------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerConfig:
    """Lightning‑style trainer 옵션을 한 곳에 모아 둔 dataclass."""
    max_epochs: int = 100
    eval_interval: int = 1                     # validation 수행 주기 (epoch 단위)
    gradient_accumulation_steps: int = 1       # gradient accumulation
    precision: str = "32"                      # "32" | "16" (AMP)
    log_to_file: bool = True
    output_dir: str = "./results"

    # ----- early‑stop 관련 옵션 -----
    early_stop_patience: Optional[int] = None          # loss 기준 patience
    early_stop_metric_name: Optional[str] = None       # validation metric 이름 (예: "acc")
    early_stop_metric_patience: Optional[int] = None   # metric 기준 patience (필요 시 별도 지정)

    # ----- scheduler 옵션 -----
    scheduler_interval: str = "epoch"          # "epoch" | "step"
    scheduler_frequency: int = 1



# ------------------------------------------------------------------
# 2) EarlyStopping 콜백
# ------------------------------------------------------------------
class EarlyStopper:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-3,
                 mode: str = 'max',
                 target_value: float | None = None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target_value = target_value
        self.reset()

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_reached = False

    def __call__(self, score: float) -> bool:
        # ① target value 도달 시 즉시 stop
        if self.target_value is not None:
            if self.mode == 'max' and score >= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True
            if self.mode == 'min' and score <= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True

        # ② 최초 호출 → best_score 초기화
        if self.best_score is None:
            self.best_score = score
            return False

        # ③ 개선 여부 판단
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        # ④ patience 초과 시 stop
        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False



# ------------------------------------------------------------------
# 3 BaseTrainer 구현
# ------------------------------------------------------------------
class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        early_stopper_loss: Optional[EarlyStopper] = None,
        early_stopper_metric: Optional[EarlyStopper] = None,
    ):
        # ---------- device ----------
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # ---------- optimizer / scheduler ----------
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.scheduler_interval = self.config.scheduler_interval
        self.scheduler_frequency = self.config.scheduler_frequency

        # ---------- loss & metrics ----------
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.metrics: Dict[str, Callable] = {}
        if metrics:
            for name, fn in metrics.items():
                self.metrics[name] = fn.to(self.device) if isinstance(fn, nn.Module) else fn

        # ---------- early‑stopping ----------
        # ① 사용자가 직접 만든 EarlyStopper 객체가 있으면 그대로 사용
        # ② 없으면 config 값으로 자동 생성
        if early_stopper_loss is not None:
            self.early_stopper_loss = early_stopper_loss
        elif self.config.early_stop_patience is not None:
            self.early_stopper_loss = EarlyStopper(
                patience=self.config.early_stop_patience,
                mode="min"
            )
        else:
            self.early_stopper_loss = None

        if early_stopper_metric is not None:
            self.early_stopper_metric = early_stopper_metric
        elif (self.config.early_stop_metric_name is not None and
              self.config.early_stop_metric_patience is not None):
            self.early_stopper_metric = EarlyStopper(
                patience=self.config.early_stop_metric_patience,
                mode="max"
            )
        else:
            self.early_stopper_metric = None

        # ---------- 로깅 ----------
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = self._setup_logger(self.output_dir, self.config.log_to_file)

        # ---------- 상태 변수 ----------
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.history: defaultdict = defaultdict(list)
        self.fit_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None

        # ---------- AMP ----------
        self.use_amp = self.config.precision == "16"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    # ------------------------------------------------------------------
    # 로거
    # ------------------------------------------------------------------
    def _setup_logger(self, output_dir: str, log_to_file: bool) -> logging.Logger:
        logger = logging.getLogger(f"BaseTrainer_{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            stream = logging.StreamHandler()
            stream.setFormatter(fmt)
            logger.addHandler(stream)

            if log_to_file:
                fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
                fh.setFormatter(fmt)
                logger.addHandler(fh)
        return logger

    # ------------------------------------------------------------------
    # Hook API (Lightning‑style)
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:
        self.logger.info("=== Training start ===")
        if self.early_stopper_loss:
            self.early_stopper_loss.reset()
        if self.early_stopper_metric:
            self.early_stopper_metric.reset()

    def on_fit_end(self) -> None:
        total = time.time() - self.fit_start_time
        self.logger.info(f"=== Training finished (elapsed {total:.2f}s) ===")

    def on_epoch_start(self) -> None:
        self.epoch_start_time = time.time()
        self.logger.info(f"\n--- Epoch {self.current_epoch + 1}/{self.config.max_epochs} ---")

    def on_epoch_end(self) -> None:
        elapsed = time.time() - self.epoch_start_time
        self.logger.info(f"Epoch {self.current_epoch + 1} finished in {elapsed:.2f}s")

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_train_epoch_end(self, train_metrics: Dict[str, float]) -> None:
        log = " | ".join([f"train_{k}: {v:.4f}" for k, v in train_metrics.items()])
        self.logger.info(f"[Train] {log}")

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def on_validation_epoch_end(self, val_metrics: Dict[str, float]) -> None:
        log = " | ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
        self.logger.info(f"[Valid] {log}")

    # ------------------------------------------------------------------
    # 사용자가 구현해야 하는 메서드 (abstract)
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _move_batch_to_device(self, batch: Any) -> Any:
        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            if isinstance(x, dict):
                return {k: _to(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(_to(v) for v in x)
            return x
        return _to(batch)

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        results = {}
        for name, fn in self.metrics.items():
            try:
                val = fn(preds, targets)
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                results[name] = float(val)
            except Exception as e:
                self.logger.warning(f"Metric '{name}' failed: {e}")
                results[name] = float("nan")
        return results

    # ------------------------------------------------------------------
    # 학습·검증 루프
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        if ckpt_path:
            self.load_checkpoint(ckpt_path)

        self.on_fit_start()
        self.fit_start_time = time.time()

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            self.on_epoch_start()

            # ---------- train ----------
            self.on_train_epoch_start()
            train_metrics = self._run_train_epoch()
            self.on_train_epoch_end(train_metrics)

            # ---------- validation ----------
            if valid_loader and ((epoch + 1) % self.config.eval_interval == 0):
                self.on_validation_epoch_start()
                val_metrics = self._run_validation_epoch()
                self.on_validation_epoch_end(val_metrics)

                # ----- early‑stop (loss) -----
                if self.early_stopper_loss:
                    loss = val_metrics.get("loss")
                    if loss is not None and self.early_stopper_loss(loss):
                        self.logger.info("Early stopping triggered (loss).")
                        break

                # ----- early‑stop (metric) -----
                if (self.early_stopper_metric and
                    self.config.early_stop_metric_name):
                    metric_val = val_metrics.get(self.config.early_stop_metric_name)
                    if metric_val is not None and self.early_stopper_metric(metric_val):
                        self.logger.info(
                            f"Early stopping triggered ({self.config.early_stop_metric_name}).")
                        break

            # ---------- scheduler ----------
            if self.scheduler:
                if self.scheduler_interval == "epoch":
                    if (epoch + 1 % self.scheduler_frequency == 0:
                        self.scheduler.step()

            self.on_epoch_end()

        self.on_fit_end()

    # ------------------------------------------------------------------
    # 내부 학습·검증 루프
    # ------------------------------------------------------------------
    def _run_train_epoch(self) -> Dict[str, float]:
        batch_metrics: List[Dict[str, float]] = []
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)
            self.on_train_batch_start(batch, batch_idx)

            # forward
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.training_step(batch, batch_idx)   # {"loss": loss, ...}
                loss = outputs["loss"] / accum_steps

            # backward
            self.on_before_backward(loss)
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            self.on_after_backward()

            # optimizer step (gradient accumulation)
            if (batch_idx + 1) % accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.on_before_optimizer_step(self.optimizer)

                self.optimizer.zero_grad()
                self.global_step += 1

            # metric 계산 (loss 제외)
            loss_val = outputs["loss"].detach().cpu().item()
            metric_vals = {}
            if "logits" in outputs and isinstance(batch, dict) and "targets" in batch:
                metric_vals = self._compute_metrics(outputs["logits"], batch["targets"])

            epoch_res = {"loss": loss_val, **metric_vals}
            batch_metrics.append(epoch_res)

            self.on_train_batch_end(outputs, batch, batch_idx)

        # epoch 평균
        avg = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0].keys()}
        for k, v in avg.items():
            self.history[f"train_{k}"].append(v)
        return avg

    @torch.no_grad()
    def _run_validation_epoch(self) -> Dict[str, float]:
        batch_metrics: List[Dict[str, float]] = []

        for batch_idx, batch in enumerate(self.valid_loader):
            batch = self._move_batch_to_device(batch)
            self.on_validation_batch_start(batch, batch_idx)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.validation_step(batch, batch_idx)

            loss_val = outputs["loss"].detach().cpu().item()
            metric_vals = {}
            if "logits" in outputs and isinstance(batch, dict) and "targets" in batch:
                metric_vals = self._compute_metrics(outputs["logits"], batch["targets"])

            epoch_res = {"loss": loss_val, **metric_vals}
            batch_metrics.append(epoch_res)

            self.on_validation_batch_end(outputs, batch, batch_idx)

        avg = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0].keys()}
        for k, v in avg.items():
            self.history[f"val_{k}"].append(v)
        return avg

    # ------------------------------------------------------------------
    # 체크포인트
    # ------------------------------------------------------------------
    def save_checkpoint(self, filepath: str, save_optimizer: bool = True) -> None:
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
        }
        if save_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, strict: bool = True) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        if "optimizer_state_dict" in ckpt and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.logger.info(f"Checkpoint loaded from {filepath}")

    # ------------------------------------------------------------------
    # History 조회
    # ------------------------------------------------------------------
    def get_history(self) -> Dict[str, List[float]]:
        return dict(self.history)
```

```python
def generic_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    - Tensor 필드 : default_collate (stack)  
    - List/None 필드 : 그대로 리스트화 (예: bbox 가 None 이거나 길이가 다른 경우)  
    - mask 가 Tensor 이지만 shape 가 다르면 (예: 이미지 크기 차이) -> 리스트 반환
    """
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]

        # Tensor 인 경우만 stack, 그 외는 리스트 유지
        if isinstance(values[0], torch.Tensor):
            # 모든 텐서 shape 가 동일하면 stack, 아니면 리스트 반환
            try:
                out[key] = default_collate(values)
            except Exception:  # shape mismatch
                out[key] = values
        else:
            out[key] = values
    return out


class PrintBatchSizeCallback:
    """배치가 로드될 때마다 배치 크기를 콘솔에 출력"""
    def on_batch_start(self, batch, batch_idx, dataloader):
        # batch 가 dict 형태라고 가정
        size = batch["image"].size(0) if isinstance(batch, dict) else len(batch)
        print(f"[Batch {batch_idx}] size = {size}")

    def on_dataloader_start(self, dataloader):
        print("=== Dataloader start ===")

    def on_dataloader_end(self, dataloader):
        print("=== Dataloader end ===")

dataloader_cfg = DataloaderConfig(
    batch_size=16,
    num_workers=8,
    callbacks=[PrintBatchSizeCallback()],
    collate_fn=generic_collate   # 앞서 만든 가변 길이 지원 collate
)

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Iterable, Iterator, List, Dict, Any


class BaseDataloader:
    """
    torch.utils.data.DataLoader 를 감싸는 래퍼.
    - DataloaderConfig 로 모든 옵션을 일괄 관리
    - 콜백을 통해 배치 전·후 로깅/통계/디버깅 가능
    - __iter__ 를 구현해 for-loop 에 바로 사용할 수 있음
    """
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        config: Optional[DataloaderConfig] = None,
        **override_kwargs,
    ):
        self.dataset = dataset
        self.config = config or DataloaderConfig()
        # 사용자가 직접 넘긴 kwargs 로 config 를 오버라이드
        for k, v in override_kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # sampler (DistributedSampler 등) 가 지정돼 있으면 shuffle 옵션을 무시
        if self.config.sampler is None and self.config.shuffle:
            self.sampler = torch.utils.data.RandomSampler(self.dataset)
        else:
            self.sampler = self.config.sampler

        # 실제 DataLoader 생성
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,                     # sampler 사용 시 shuffle=False 로 고정
            sampler=self.sampler,
            num_workers=self.config.num_workers,
            collate_fn=self.config.collate_fn,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            timeout=self.config.timeout,
        )

        # 콜백 (list 혹은 단일 객체) 를 내부에 저장
        self.callbacks = self.config.callbacks

    # ------------------------------------------------------------------
    # 콜백 헬퍼
    # ------------------------------------------------------------------
    def _call(self, hook_name: str, *args, **kwargs):
        """등록된 콜백들의 hook_name 메서드를 순차적으로 호출."""
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)
            if callable(hook):
                hook(*args, **kwargs)

    # ------------------------------------------------------------------
    # 외부에서 epoch 번호를 바꿔야 할 경우 (DistributedSampler 용)
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        """DistributedSampler 를 사용할 때 epoch 정보를 전달."""
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    # ------------------------------------------------------------------
    # iterator 구현 (for-loop 에 바로 사용)
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._call("on_dataloader_start", self)
        for batch_idx, batch in enumerate(self.loader):
            self._call("on_batch_start", batch, batch_idx, self)
            yield batch
            self._call("on_batch_end", batch, batch_idx, self)
        self._call("on_dataloader_end", self)

    # ------------------------------------------------------------------
    # DataLoader 의 주요 속성에 직접 접근할 수 있게 proxy
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f"<BaseDataloader len={len(self)} batch_size={self.config.batch_size}>"

    # ------------------------------------------------------------------
    # 명시적으로 워커를 종료하고 싶을 때 (멀티프로세싱 환경에서 권장)
    # ------------------------------------------------------------------
    def close(self):
        if hasattr(self.loader, "worker_init_fn"):
            # DataLoader 가 내부적으로 만든 워커를 정리
            self.loader._iterator._shutdown_workers()
```
