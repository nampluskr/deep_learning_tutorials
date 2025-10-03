import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, roc_curve
from skimage import measure
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from time import time
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments


class BaseTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.metrics = {}
        if metrics:
            for name, metric_fn in metrics.items():
                self.metrics[name] = metric_fn.to(self.device) if isinstance(metric_fn, nn.Module) else metric_fn

        self.epoch_period = 5           # period for evaluation of classification metrics
        self.history = {}               # training history

        # Temperary variables for training/validation process
        self.fit_start_time = None      # time at the start of training
        self.epoch_start_time = None    # time at the start of each epoch
        self.epoch = None               # current epoch number
        self.num_epochs = None          # total number of epochs for training
        self.train_info = None          # string for training info
        self.valid_info = None          # string for validation info

    #############################################################
    # Hooks for training process
    #############################################################

    def on_train_start(self, train_loader):
        self.model.train()

    @torch.enable_grad()
    def train_step(self, batch):
        raise NotImplementedError

    def train_epoch(self, train_loader):
        results = {"loss": 0.0, **{name: 0.0 for name in self.metrics}}
        total = 0
        with tqdm(train_loader, desc="> Training", leave=False, ascii=True) as pbar:
            for batch in pbar:
                batch_size = batch["image"].size(0)
                total += batch_size
                step_results = self.train_step(batch)

                for name, value in step_results.items():
                    results.setdefault(name, 0.0)
                    results[name] += value * batch_size

                pbar.set_postfix({name: f"{value/total:.3f}" for name, value in results.items()})
        return {name: value/total for name, value in results.items()}

    def on_train_end(self, train_results):
        self.train_info = ", ".join([f'{k}={v:.3f}' for k, v in train_results.items()])
        self.valid_info = None

        for name, value in train_results.items():
            self.history.setdefault(name, [])
            self.history[name].append(value)

    #############################################################
    # Hooks for validation process
    #############################################################

    def on_validation_start(self, valid_loader):
        self.model.eval()

    @torch.no_grad()
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model.predict(images)
        scores = predictions['pred_score'].cpu()
        if scores.ndim > 1:
            scores = scores.view(scores.size(0))
        return scores

    @torch.no_grad()
    def validation_epoch(self, loader):
        all_scores, all_labels = [], []
        with tqdm(loader, desc="> Validation", leave=False, ascii=True) as pbar:
            for batch in loader:
                scores = self.validation_step(batch)
                labels = batch["label"].cpu()

                all_scores.append(scores)
                all_labels.append(labels)

        scores = torch.cat(all_scores)   # shape = [N_images]
        labels = torch.cat(all_labels)   # shape = [N_images]
        results = {}
        results["auroc"] = roc_auc_score(labels, scores)
        results["aupr"] = average_precision_score(labels, scores)
        return results, scores, labels

    def on_validation_end(self, valid_results, scores, labels):
        self.valid_info = ", ".join([f'{k}={v:.3f}' for k, v in valid_results.items()])
        elapsed_time = time() - self.epoch_start_time
        epoch_info = f" [{self.epoch:3d}/{self.num_epochs}]"
        print(f" {epoch_info} {self.train_info} | {self.valid_info} ({elapsed_time:.1f}s)")
        
        for name, value in valid_results.items():
            self.history.setdefault(name, [])
            self.history[name].append(value)

        if self.epoch % self.epoch_period == 0:
            threshold = compute_threshold(scores, labels, method="f1")
            eval_results = evaluate_classification(scores, labels, threshold)
            eval_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_results.items() if isinstance(v, float)])
            eval_info2 = ", ".join([f"{k.upper()}={v}" for k, v in eval_results.items() if isinstance(v, int)])
            print(f" > {eval_info1} ({eval_info2})\n")

    #############################################################
    # Hooks for fitting process
    #############################################################

    def on_fit_start(self ):
        print("\n > Start training...\n")
        self.fit_start_time = time()

    def on_fit_end(self, weight_path=None):
        elapsed_time = time() - self.fit_start_time
        hours, reminder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(reminder, 60)
        print(f"\n > Training finished... in {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        self.save_model(weight_path)
        
    #############################################################
    # Hooks for epoch process
    #############################################################

    def on_epoch_start(self):
        self.epoch_start_time = time()

    def on_epoch_end(self):
        if self.valid_info is None:
            elapsed_time = time() - self.epoch_start_time
            epoch_info = f" [{self.epoch:3d}/{self.num_epochs}]"
            print(f" {epoch_info} {self.train_info} ({elapsed_time:.1f}s)")

    #############################################################
    # Main fit function
    #############################################################

    def fit(self, train_loader, num_epochs, valid_loader=None, weight_path=None):
        self.num_epochs = num_epochs
        self.on_fit_start()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            self.on_epoch_start()

            self.on_train_start(train_loader)
            train_results = self.train_epoch(train_loader)
            self.on_train_end(train_results)

            if valid_loader:
                self.on_validation_start(valid_loader)
                valid_results, scores, labels = self.validation_epoch(valid_loader)
                self.on_validation_end(valid_results, scores, labels)

            self.on_epoch_end()

        self.on_fit_end(weight_path)
        return self.history

    #############################################################
    # Model save / load
    #############################################################

    def save_model(self, weight_path):
        if weight_path is not None:
            output_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(output_dir, exist_ok=True)
            checkpoint = {"model": self.model.state_dict()}

            if self.optimizer is not None:
                checkpoint["optimizer"] = self.optimizer.state_dict()
            torch.save(checkpoint, weight_path)
            print(f" > Model (and optimizer) weights saved to: {weight_path}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f" > Model weights loaded from: {weight_path}")

            if self.optimizer is not None and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print(" > Optimizer state loaded.")
        else:
            print(f" > No model weights found at: {weight_path}\n")

    #############################################################
    # Model test (saving anomaly maps)
    #############################################################

    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
            skip_normal=False, skip_anomaly=False, num_max=-1, imagenet_normalize=True):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        num_images = 0
        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            images = batch["image"].to(self.device)
            masks = batch["mask"].cpu().numpy() if "mask" in batch else None

            prediction = self.model.predict(images)
            anomaly_maps = prediction["anomaly_map"].cpu().numpy()
            scores = prediction["pred_score"].cpu().numpy()

            for i in range(images.size(0)):
                label = int(labels[i])
                score = float(scores[i])

                if skip_normal and label == 0: continue
                if skip_anomaly and label == 1: continue
                if num_max > 0 and num_images >= num_max: continue
                num_images += 1

                img_tensor = images[i].cpu()
                if imagenet_normalize:
                    original = denormalize(img_tensor).clamp(0, 1)
                else:
                    original = img_tensor.clamp(0, 1)
                amap = anomaly_maps[i]
                anomaly_map = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                binary_mask = None
                if masks is not None:
                    mask = masks[i]
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    binary_mask = mask

                if binary_mask is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                    titles = [f"Original: {label}", "Mask", f"Anomaly: {score:.4f}"]
                    images_vis = [original, binary_mask, anomaly_map]
                    cmaps = [None, "gray", "jet"]
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                    titles = [f"Original: {label}", f"Anomaly: {score:.4f}"]
                    images_vis = [original, anomaly_map]
                    cmaps = [None, "jet"]

                for ax, img, title, cmap in zip(axes, images_vis, titles, cmaps):
                    ax.imshow(check_shape(img), cmap=cmap)
                    ax.set_title(title)
                    ax.axis("off")

                fig.tight_layout()
                if output_dir is not None:
                    label_name = "normal" if label == 0 else "anomaly"
                    file_name = f"{img_prefix}_{label_name}_{num_images:03d}.png"
                    fig.savefig(os.path.join(output_dir, file_name), dpi=150)
                if show_image:
                    plt.show()

                plt.close(fig)

#############################################################
# Helper functions
#############################################################

def compute_threshold(scores, labels, method="f1", percentile=95):
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()

    if method == "f1":
        thresholds = np.percentile(scores_np, np.linspace(1, 99.9, 200))
        best_f1, best_threshold = 0.0, 0.5
        for thr in thresholds:
            preds = (scores_np >= thr).astype(int)
            f1 = f1_score(labels_np, preds)
            if f1 > best_f1:
                best_f1, best_threshold = f1, thr
        return float(best_threshold)

    elif method == "roc":
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
        optimal_idx = np.argmax(tpr - fpr)
        return float(thresholds[optimal_idx])

    else:  # "percentile"
        normal_mask = labels_np == 0
        if normal_mask.sum() == 0:
            return float(np.percentile(scores_np, 95))
        normal_scores = scores_np[normal_mask]
        return float(np.percentile(normal_scores, percentile))


def evaluate_classification(scores, labels, threshold):
        preds = (scores >= threshold).long()
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        return {
            "th": float(threshold),
            "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
            "prec": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        }


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def check_shape(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 2:
        return img  # (H, W) grayscale
    if img.ndim == 3:
        # Case 1: (1, H, W) → (H, W)
        if img.shape[0] == 1:
            return img[0]
        # Case 2: (3, H, W) → (H, W, 3)
        elif img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0))
        # Case 3: (H, W, 3) → 그대로 반환
        elif img.shape[2] == 3:
            return img
        else:
            raise ValueError(f"Invalid shape for 3D array: {img.shape}")
    raise ValueError(f"Unsupported image shape: {img.shape}")


