import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, precision_recall_curve, roc_curve)
from types import SimpleNamespace
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from ssim import ssim


def get_config(category="tile"):
    config = SimpleNamespace(
        data_root="/home/namu/myspace/NAMU/datasets/mvtec",
        category=category,
        img_size=256,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,

        model_type="autoencoder",
        num_epochs=100,
        learning_rate= 1e-4,
        latent_dim=512,
        output_dir="./experiments"
    )
    config.weight_path = os.path.join(config.output_dir,
        f"model_{config.category}_{config.model_type}_epochs-{config.num_epochs}.pth")
    os.makedirs(os.path.dirname(config.weight_path), exist_ok=True)
    return config


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", transform=None, mask_transform=None):
        self.root = os.path.join(root, category, split)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.labels = []

        for label in sorted(os.listdir(self.root)):
            label_dir = os.path.join(self.root, label)
            if not os.path.isdir(label_dir):
                continue
            for img_name in sorted(os.listdir(label_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                image_path = os.path.join(label_dir, img_name)
                labels = 0 if label == "good" else 1
                self.image_paths.append(image_path)
                self.labels.append(labels)

        print(f" > {split.capitalize()} set: {len(self.image_paths)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
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


def get_dataloaders(config):
    train_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])

    test_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])
    mask_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
    ])

    train_set = MVTecDataset(root=config.data_root, category=config.category, split="train",
                             transform=train_transform, mask_transform=mask_transform)
    test_set  = MVTecDataset(root=config.data_root, category=config.category, split="test",
                             transform=test_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers)
    test_loader  = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers)

    return train_loader, test_loader


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=256 * 32 * 32, out_features=latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256 * 32 * 32),
            nn.Unflatten(dim=1, unflattened_size=(256, 32, 32)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def compute_anomaly_map(self, reconstructed, images):
        anomaly_map = torch.mean((images - reconstructed)**2, dim=1, keepdim=True)
        return anomaly_map

    def compute_anomaly_score(self, anomaly_map):
        # img_score = anomaly_maps.view(anomaly_map.size(0), -1).mean(dim=1)
        # pred_score = img_score.detach().cpu().numpy().tolist()
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        recon, *_ = self.forward(images)
        anomaly_map = self.compute_anomaly_map(recon, images)
        pred_score = self.compute_anomaly_score(anomaly_map)
        return {"anomaly_map": anomaly_map, "pred_score": pred_score}


class SSIMMetric(nn.Module):
    """Structural Similarity Index Measure metric."""
    
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, preds, targets):
        ssim_value = ssim(preds, targets, data_range=self.data_range, size_average=True)
        return ssim_value.item()


def train(model, loader, loss_fn, optimizer, metric_fn, device):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    for inputs in loader:
        images = inputs['image'].to(device)

        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = loss_fn(reconstructed, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            total_metric += metric_fn(reconstructed, images) * images.size(0)

    return total_loss / len(loader.dataset), total_metric / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []

    for inputs in loader:
        images = inputs['image'].to(device)
        labels = inputs['label'].to(device)

        reconstructed, _ = model(images)
        anomaly_maps = model.compute_anomaly_map(reconstructed, images)
        scores = model.compute_anomaly_score(anomaly_maps)

        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return all_scores, all_labels


def run_experiement(config):
    print(f"\n*** RUN EXPERIMENT: {config.model_type.upper()} - {config.category.upper()}")

    train_loader, test_loader = get_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(latent_dim=config.latent_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    metric_fn = SSIMMetric()

    best_auroc = 0.0
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_metric = train(model, train_loader, loss_fn, optimizer, metric_fn, device)
        scores, labels = validate(model, test_loader, device)

        auroc = roc_auc_score(labels, scores)
        aupr  = average_precision_score(labels, scores)

        print(f"Epoch [{epoch}/{config.num_epochs}] loss={train_loss:.4f}, ssim={train_metric:.4f}"
              f" | (val) auroc={auroc:.4f}, aupr={aupr:.4f}")

        # if auroc > best_auroc:
        #     best_auroc = auroc
        #     best_aupr = aupr
        #     torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "auroc": best_auroc,
        #         "aupr": best_aupr,
        #         "config": vars(config)
        #     }, config.weight_path)
        #     print(f"  > New best model saved (AUROC={best_auroc:.4f}, AUPR={best_aupr:.4f})")

    print("\n...Training finished.")
    # print(f"Best AUROC = {best_auroc:.4f}, AUPR = {best_aupr:.4f}")
    # print(f"Best model saved at: {config.weight_path}")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "auroc": best_auroc,
                # "aupr": best_aupr,
                "config": vars(config)
            }, config.weight_path)

    scores, labels = validate(model, test_loader, device)
    thresholds = get_thresholds(scores, labels)
    results = evaluate_thresholds(scores, labels, thresholds)
    print(results)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1,3,1,1)
    return tensor * std + mean


def get_thresholds(scores, labels, n_steps=100):
    # --------------------------------------------------------------
    # 0) 입력을 numpy 로 변환
    # --------------------------------------------------------------
    scores_np = np.asarray(scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    thresholds = {}

    # --------------------------------------------------------------
    # 1) ROC‑Youden (max TPR‑FPR)
    # --------------------------------------------------------------
    fpr, tpr, thr = roc_curve(labels_np, scores_np)
    optimal_idx = np.argmax(tpr - fpr)
    thresholds["roc"] = float(thr[optimal_idx])

    # --------------------------------------------------------------
    # 2) F1‑score maximization
    # --------------------------------------------------------------
    cand_thr = np.linspace(scores_np.min(),
                           scores_np.max(),
                           num=n_steps,
                           dtype=np.float32)

    best_f1, best_thr = -1.0, None
    for thr_val in cand_thr:
        preds = (scores_np >= thr_val).astype(np.float32)
        f1 = f1_score(labels_np, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr_val

    thresholds["f1"] = float(best_thr) if best_thr is not None else float(cand_thr.mean())

    # --------------------------------------------------------------
    # 3) Percentile of normal scores (95, 97, 99)
    # --------------------------------------------------------------
    normal_mask = labels_np == 0
    if normal_mask.sum() == 0:
        warnings.warn("No normal samples found - percentile thresholds set to NaN")
        for p in (95.0, 97.0, 99.0):
            thresholds[f"percentile_{int(p)}"] = np.nan
    else:
        normal_scores = scores_np[normal_mask]
        for p in (95.0, 97.0, 99.0):
            thresholds[f"percentile_{int(p)}"] = float(np.percentile(normal_scores, p))

    # --------------------------------------------------------------
    # 4) Fβ‑score (β = 2.0, 1.5, 1.0) – Precision‑Recall 기반
    # --------------------------------------------------------------
    precision, recall, thr_pr = precision_recall_curve(labels_np, scores_np)
    # thr_pr 길이는 precision/recall 길이보다 1 짧음 → 마지막 인덱스는 제외
    for beta in (2.0, 1.5, 1.0):
        beta2 = beta ** 2
        fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
        idx = np.nanargmax(fbeta)
        # idx 가 thr_pr 길이와 같다면 마지막 유효 threshold 사용
        if idx >= len(thr_pr):
            idx = len(thr_pr) - 1
        thresholds[f"fbeta_{beta:g}"] = float(thr_pr[idx])

    # --------------------------------------------------------------
    # 5) Cost‑sensitive threshold (FP·FN 비용 최소화)
    # --------------------------------------------------------------
    fnr = 1 - tpr                     # False‑Negative Rate
    cost = 1.0 * fpr + 5.0 * fnr      # cost_fp=1.0, cost_fn=5.0
    idx = np.argmin(cost)
    thresholds["cost_sensitive"] = float(thr[idx])

    # --------------------------------------------------------------
    # 6) SPC 3σ / 2σ / 1σ 규칙 (정규성 가정)
    # --------------------------------------------------------------
    mu = scores_np.mean()
    sigma = scores_np.std()
    thresholds["3sigma"] = mu + 3.0 * sigma
    thresholds["2sigma"] = mu + 2.0 * sigma
    thresholds["1sigma"] = mu + 1.0 * sigma

    # --------------------------------------------------------------
    # 7) Fixed false‑positive rate (target FPR = 0.01)
    # --------------------------------------------------------------
    idx = np.where(fpr <= 0.01)[0]
    if idx.size > 0:
        thresholds["fixed_fpr"] = float(thr[idx[0]])
    else:
        # 가장 작은 FPR에 해당하는 threshold 로 대체
        min_fpr_idx = np.argmin(fpr)
        thresholds["fixed_fpr"] = float(thr[min_fpr_idx])

    return thresholds


def evaluate_thresholds(scores, labels, thresholds):
    try:
        import torch
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
    except Exception:
        scores_np = np.asarray(scores)
        labels_np  = np.asarray(labels)

    auroc = roc_auc_score(labels_np, scores_np)
    aupr  = average_precision_score(labels_np, scores_np)
    normal_scores  = scores_np[labels_np == 0]
    anomaly_scores = scores_np[labels_np == 1]

    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        pooled_var = (
            (len(normal_scores) - 1) * normal_scores.var(ddof=1) +
            (len(anomaly_scores) - 1) * anomaly_scores.var(ddof=1)
        ) / (len(normal_scores) + len(anomaly_scores) - 2)
        pooled_std = np.sqrt(pooled_var)
        cohens_d   = (anomaly_scores.mean() - normal_scores.mean()) / pooled_std
    else:
        cohens_d   = np.nan

    rows = []
    for thr in thresholds:
        thr_val = thresholds[thr]
        pred = (scores_np >= thr_val).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(labels_np, pred, labels=[0, 1]).ravel()

        rows.append({
            "method":    thr,
            "threshold": thr_val,
            "auroc":     auroc,
            "aupr":      aupr,
            "cohens_d":  cohens_d,
            "accuracy":  accuracy_score(labels_np, pred),
            "precision": precision_score(labels_np, pred, zero_division=0),
            "recall":    recall_score(labels_np, pred, zero_division=0),
            "f1":        f1_score(labels_np, pred, zero_division=0),
            "mcc":       matthews_corrcoef(labels_np, pred),
            "kappa":     cohen_kappa_score(labels_np, pred),
            "tn":        tn,
            "fp":        fp,
            "fn":        fn,
            "tp":        tp,
        })
    return pd.DataFrame(rows)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1,3,1,1)
    return tensor * std + mean

def model_test(model, test_loader, device):
    for inputs in test_loader:
        images = inputs['image'].to(device)
        reconstructed, _ = model(images)
        anomaly_maps = model.compute_anomaly_map(reconstructed, images)
        scores = model.compute_anomaly_score(anomaly_maps)

        names = inputs['name']
        images = denormalize(images).cpu().numpy()
        reconstructed = denormalize(reconstructed).detach().cpu().numpy()
        scores = scores.detach().cpu().numpy().squeeze(axis=-1)
        masks = inputs['mask'].cpu().numpy()
        anomaly_maps = anomaly_maps.detach().cpu().numpy().squeeze(1)
        labels = inputs['label'].cpu().numpy()

        batch_size = images.shape[0]
        fig, axes = plt.subplots(batch_size, 4, figsize=(8, 2*batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for i in range(batch_size):
            axes[i, 0].imshow(images[i].transpose(1, 2, 0))
            axes[i, 0].set_title(f"{names[i]} ({labels[i]})")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(reconstructed[i].transpose(1, 2, 0))
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(masks[i], cmap="gray")
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis('off')

            im = axes[i, 3].imshow(anomaly_maps[i], cmap='hot', vmin=0, vmax=anomaly_maps[i].max())
            axes[i, 3].set_title(f"Anomaly ({scores[i]:.3e})")
            axes[i, 3].axis('off')
            # plt.colorbar(im, ax=axes[i, 2], shrink=0.8)

        plt.tight_layout()
        plt.show()
        # break

if __name__ == "__main__":

    config = get_config(category="tile")
    run_experiement(config)

    config = get_config(category="grid")
    run_experiement(config)
