#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b3
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ==============================================================
# 1️⃣ 기본 설정
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_dir = "/home/namu/myspace/work_20250915_v0.3/experiments"
os.makedirs(exp_dir, exist_ok=True)

# ==============================================================
# 2️⃣ Dataset
# ==============================================================

class MVTecDataset(Dataset):
    """
    (image_tensor, label, image_path) 를 반환
    label : 0 = good , 1 = defect
    """
    def __init__(self, root, category, split, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        base_dir = os.path.join(root, category, split)   # train / test
        good_dir = os.path.join(base_dir, "good")
        for f in sorted(glob.glob(os.path.join(good_dir, "*.*"))):
            self.images.append(f)
            self.labels.append(0)

        # defect 이미지 (mask는 별도 로드)
        gt_root = os.path.join(root, category, "ground_truth")
        for defect in sorted(os.listdir(gt_root)):
            img_dir = os.path.join(base_dir, defect)
            if not os.path.isdir(img_dir):
                continue
            for img_path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
                self.images.append(img_path)
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)          # (C, H, W) 0~1
        label = self.labels[idx]
        return img, label, self.images[idx]

# ==============================================================
# 3️⃣ Efficient AD (Mahalanobis) 모델
# ==============================================================

class EfficientAD(nn.Module):
    """
    EfficientNet‑B3 를 backbone 으로 사용하고,
    train‑set 전체에 대해 평균 μ 와 공분산 역행렬 Σ⁻¹ 을 추정한다.
    """
    def __init__(self, backbone_path):
        super().__init__()
        # ---- backbone 로드 ----
        self.backbone = efficientnet_b3(weights=None)
        self.backbone.load_state_dict(
            torch.load(backbone_path, map_location=device)
        )
        # classifier 를 Identity 로 교체 → feature extractor only
        self.backbone.classifier = nn.Identity()
        self.backbone = self.backbone.to(device).eval()

        # ---- 학습 후 저장될 통계 ----
        self.mean = None          # (C,)   – GPU에 저장
        self.cov_inv = None       # (C, C) – GPU에 저장

    # -----------------------------------------------------------------
    # 3‑1️⃣ feature extractor (spatial feature map 반환)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def extract(self, x):
        """
        EfficientNet‑B3 의 convolutional block 까지만 반환.
        shape : (B, C, H, W)
        """
        return self.backbone.features(x)   # avgpool, flatten, classifier 를 건너뜀

    # -----------------------------------------------------------------
    # 3‑2️⃣ train‑set 으로 평균·공분산 추정
    # -----------------------------------------------------------------
    @torch.no_grad()
    def fit(self, loader):
        print("[fit] 시작 – 전체 train‑set 으로 평균·공분산 추정")
        self.backbone.eval()
        feats = []                     # CPU에 임시 저장
        total = len(loader.dataset)
        processed = 0

        for batch_idx, (imgs, _, _) in enumerate(loader):
            imgs = imgs.to(device)                     # (B, C, H, W) → GPU
            f = self.extract(imgs)                     # (B, C, h, w)
            f = F.adaptive_avg_pool2d(f, 1)            # (B, C, 1, 1)
            f = f.view(f.size(0), -1)                  # (B, C)
            feats.append(f.cpu())
            processed += imgs.size(0)

            if (batch_idx + 1) % 10 == 0 or processed == total:
                print(f"[fit] {processed}/{total} 이미지 처리 완료")

        feats = torch.cat(feats, dim=0)                 # (N, C) – CPU

        # ----- 평균 & 공분산 (CPU) -----
        mean_cpu = feats.mean(dim=0)                    # (C,)
        cov_cpu  = torch.cov(feats.t())                 # (C, C) 혹은 scalar
        if cov_cpu.dim() == 0:                         # C == 1 인 경우
            cov_cpu = cov_cpu.view(1, 1)

        eps = 1e-6 * torch.eye(cov_cpu.size(0), device=cov_cpu.device)
        cov_inv_cpu = torch.inverse(cov_cpu + eps)      # (C, C)

        # ----- GPU 로 이동 -----
        self.mean    = mean_cpu.to(device)              # (C,) on GPU
        self.cov_inv = cov_inv_cpu.to(device)           # (C, C) on GPU

        print("[fit] 완료 – 평균·공분산 저장")
        print(f"    feature dim : {self.mean.size(0)}")
        print(f"    cov shape   : {self.cov_inv.shape}")

    # -----------------------------------------------------------------
    # 3‑3️⃣ 배치 차원 보장 (3‑D → 4‑D)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _ensure_batch_dim(self, x):
        if x.dim() == 3:          # (C, H, W)
            return x.unsqueeze(0) # (1, C, H, W)
        return x                  # 이미 (B, C, H, W)

    # -----------------------------------------------------------------
    # 3‑4️⃣ single‑image Mahalanobis score (scalar)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def score(self, img_tensor):
        img_tensor = self._ensure_batch_dim(img_tensor).to(device)   # (B, C, H, W)
        f = self.extract(img_tensor)                                 # (B, C, h, w)
        f = F.adaptive_avg_pool2d(f, 1).view(-1)                    # (C,)
        diff = f - self.mean                                         # GPU
        md = diff @ self.cov_inv @ diff
        return md.item()

    # -----------------------------------------------------------------
    # 3‑5️⃣ spatial Mahalanobis map (H, W) → 0~1 정규화
    # -----------------------------------------------------------------
    @torch.no_grad()
    def anomaly_map(self, img_tensor):
        img_tensor = self._ensure_batch_dim(img_tensor).to(device)   # (B, C, H, W)
        f = self.extract(img_tensor)                                 # (B, C, h, w)

        # fallback: (B, C) → 1×1
        if f.dim() == 2:                     # (B, C)
            B, C = f.shape
            f = f.view(B, C, 1, 1)           # (B, C, 1, 1)

        B, C, h, w = f.shape                 # (B, C, h, w) 보장
        f_flat = f.view(C, -1)                # (C, h*w)

        # Mahalanobis distance per pixel
        tmp = torch.matmul(self.cov_inv, f_flat)   # (C, h*w)
        md = (f_flat * tmp).sum(dim=0)              # (h*w,)

        md = md.view(1, 1, h, w)                    # (1,1,h,w)

        # 원본 이미지 크기로 up‑sample
        up = F.interpolate(md, size=img_tensor.shape[2:], mode='bilinear',
                           align_corners=False)
        amap = up.squeeze().cpu().numpy()            # (H, W)

        # 0~1 로 정규화 (상위 1% 클리핑)
        amap = np.clip(amap, 0, np.percentile(amap, 99))
        if amap.max() - amap.min() > 0:
            amap = (amap - amap.min()) / (amap.max() - amap.min())
        return amap                                 # (H, W)

# ==============================================================
# 4️⃣ 평가 함수
# ==============================================================

@torch.no_grad()
def evaluate(model, loader):
    print("[evaluate] 시작")
    scores, labels = [], []
    for img, label, _ in loader:
        scores.append(model.score(img))
        labels.append(label)

    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()

    auroc = roc_auc_score(labels, scores)
    aupr  = average_precision_score(labels, scores)
    thr   = np.percentile(scores, 95)
    preds = (scores >= thr).astype(int)

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "Threshold": thr,
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1": f1_score(labels, preds, zero_division=0),
        "Scores": scores,
        "Labels": labels,
        "MeanScore": scores.mean(),
    }

# ==============================================================
# 5️⃣ 시각화 함수 (요구사항에 맞게 재구성)
# ==============================================================

def plot_results(model, category,
                 good_img_path,
                 defect_img_paths,
                 gt_mask_paths,
                 metrics,
                 epoch,
                 timestamp):
    """
    - 1행 : 양품 (Original, Ground‑Truth, Anomaly‑Map)
    - 2행~ : 결함 이미지 각각 (Original, Ground‑Truth, Anomaly‑Map)
    - Anomaly‑Map 은 jet 컬러맵 오버레이 + max‑point 빨간 원 표시
    """
    n_defects = len(defect_img_paths)
    n_rows = 1 + n_defects          # 1 row for good, N rows for defects
    n_cols = 3                      # Original / GT / Anomaly‑Map

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows))
    # 1‑dim axes 를 2‑dim 리스트로 통일
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # --------------------------------------------------------------
    # 1️⃣ 양품 행
    # --------------------------------------------------------------
    good_img = Image.open(good_img_path).convert("RGB")
    # (0) Original
    axes[0, 0].imshow(good_img)
    axes[0, 0].set_title("Good – Original")
    axes[0, 0].axis("off")
    # (1) Ground‑Truth (양품은 전부 0 마스크 → 검은 이미지)
    axes[0, 1].imshow(np.zeros_like(np.array(good_img)), cmap="gray")
    axes[0, 1].set_title("Good – Ground Truth")
    axes[0, 1].axis("off")
    # (2) Anomaly‑Map (양품은 거의 0)
    amap_good = model.anomaly_map(T.ToTensor()(good_img))
    overlay_good = apply_jet_overlay(good_img, amap_good)
    axes[0, 2].imshow(overlay_good)
    axes[0, 2].set_title("Good – Anomaly Map")
    axes[0, 2].axis("off")

    # --------------------------------------------------------------
    # 2️⃣ 결함 행 (defect 이미지마다 한 행)
    # --------------------------------------------------------------
    for i, (orig_path, gt_path) in enumerate(zip(defect_img_paths, gt_mask_paths)):
        row = i + 1
        # ----- Original -----
        orig_img = Image.open(orig_path).convert("RGB")
        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title(f"Defect {i+1} – Original")
        axes[row, 0].axis("off")

        # ----- Ground Truth -----
        gt_img = Image.open(gt_path).convert("L")
        axes[row, 1].imshow(gt_img, cmap="gray")
        axes[row, 1].set_title(f"Defect {i+1} – Ground Truth")
        axes[row, 1].axis("off")

        # ----- Anomaly Map (jet overlay + max‑point) -----
        amap = model.anomaly_map(T.ToTensor()(orig_img))
        overlay = apply_jet_overlay(orig_img, amap, mark_max=True)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Defect {i+1} – Anomaly Map")
        axes[row, 2].axis("off")

    # 전체 타이틀
    fig.suptitle(f"Efficient AD – Category: {category}\n"
                 f"AUROC: {metrics['AUROC']:.3f} | AUPR: {metrics['AUPR']:.3f} | "
                 f"Mean Score: {metrics['MeanScore']:.3f}",
                 fontsize=16)

    # 저장
    fname = f"{timestamp}_epoch{epoch}_{category}_result.png"
    fig.savefig(os.path.join(exp_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_jet_overlay(pil_img, anomaly_map, mark_max=False):
    """
    - `anomaly_map` : (H, W) 0~1 정규화된 값
    - 반환값 : (H, W, 3) uint8 RGB 이미지 (jet overlay)
    - `mark_max=True` 인 경우 anomaly_map 의 최댓값 좌표에 빨간 원을 그림
    """
    # 1) jet 컬러맵 적용 → (H, W, 3) float [0,1]
    cmap = plt.get_cmap("jet")
    colored = cmap(anomaly_map)[:, :, :3]          # (H, W, 3)

    # 2) 원본 이미지와 같은 shape 로 변환
    img_np = np.array(pil_img).astype(np.float32) / 255.0   # (H, W, 3) 0~1

    # 3) 알파 블렌딩 (0.6 원본 + 0.4 anomaly)
    overlay = 0.6 * img_np + 0.4 * colored
    overlay = np.clip(overlay, 0, 1)

    # 4) 최댓값 좌표에 빨간 점 표시 (optional)
    if mark_max:
        y, x = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        # 빨간색 (1,0,0) 로 크게 표시
        cv_radius = max(5, int(0.02 * max(anomaly_map.shape)))   # 이미지 크기에 비례
        rr, cc = draw_circle(y, x, cv_radius, overlay.shape[:2])
        overlay[rr, cc] = [1.0, 0.0, 0.0]   # 빨간색

    # 5) uint8 로 변환
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8


def draw_circle(cy, cx, radius, shape):
    """
    간단한 원 그리기 (Bresenham 방식). 반환값은 (row_idx, col_idx) 튜플.
    `shape` 은 (H, W) 로, 이미지 경계를 넘어가지 않게 클리핑한다.
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    return np.where(mask)


# ==============================================================
# 6️⃣ 메인 – 전체 파이프라인
# ==============================================================

def main(categories=["tile", "carpet", "grid"],
         defect_per_category=1,
         epochs=30,
         load_path=None):
    """
    - 모든 카테고리 train‑set을 합쳐 하나의 Mahalanobis 통계(μ, Σ⁻¹) 를 학습
    - 학습된 모델을 하나의 파일(`efficient_ad_all_*.pth`) 로 저장/로드
    - 각 카테고리별 test‑set 에 대해 평가 + 시각화
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    backbone_path = "/home/namu/myspace/NAMU/backbones/efficientnet_b3_rwightman-b3899882.pth"
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    # --------------------------------------------------------------
    # 1️⃣ 전체 train‑set (ConcatDataset)
    # --------------------------------------------------------------
    train_sets = []
    for cat in categories:
        ds = MVTecDataset(
            "/home/namu/myspace/NAMU/datasets/mvtec", cat, "train", transform
        )
        train_sets.append(ds)
        print(f"[{cat}] train set : {len(ds)} 이미지")
    concat_train = ConcatDataset(train_sets)
    train_loader = DataLoader(
        concat_train,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"[전체] 합친 train set : {len(concat_train)} 이미지")

    # --------------------------------------------------------------
    # 2️⃣ 모델 학습 / 로드
    # --------------------------------------------------------------
    model = EfficientAD(backbone_path)

    if load_path and os.path.isfile(load_path):
        print("[load] 기존 모델 로드 :", load_path)
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        model.fit(train_loader)                     # μ, Σ⁻¹ 추정
        save_path = os.path.join(
            exp_dir,
            f"efficient_ad_all_epochs-{epochs}_{timestamp}.pth"
        )
        torch.save(model.state_dict(), save_path)
        print("[save] 전체 모델 저장 :", save_path)

    # --------------------------------------------------------------
    # 3️⃣ 카테고리별 테스트 / 평가 / 시각화
    # --------------------------------------------------------------
    for cat in categories:
        print(f"\n>>> [{cat}] 테스트 시작")
        test_set = MVTecDataset(
            "/home/namu/myspace/NAMU/datasets/mvtec", cat, "test", transform
        )
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # ---- 평가 ----
        metrics = evaluate(model, test_loader)

        # ---- 시각화에 사용할 이미지 경로 ----
        # 양품 (good) – test/good 폴더 중 첫 번째 이미지
        good_path = sorted(
            glob.glob(f"/home/namu/myspace/NAMU/datasets/mvtec/{cat}/test/good/*.*")
        )[0]

        # 결함 이미지 (defect) – 원하는 개수만큼 선택
        defect_all = sorted(
            glob.glob(f"/home/namu/myspace/NAMU/datasets/mvtec/{cat}/test/*/*.*")
        )
        defect_all = [p for p in defect_all if "good" not in p][:defect_per_category]

        # 각 결함에 대응하는 ground‑truth 마스크 경로
        gt_paths = []
        for d_path in defect_all:
            gt = d_path.replace("/test/", "/ground_truth/")
            gt = os.path.splitext(gt)[0] + "_mask.png"
            gt_paths.append(gt)

        # ---- 시각화 ----
        plot_results(
            model,
            cat,
            good_path,
            defect_all,
            gt_paths,
            metrics,
            epochs,
            timestamp,
        )
        print(f">>> [{cat}] 시각화 저장 완료")

    print("\n=== 전체 파이프라인 종료 ===")


# ==============================================================
# 7️⃣ 실행
# ==============================================================

if __name__ == "__main__":
    main(
        categories=["tile", "carpet", "grid"],   # 원하는 카테고리 리스트
        defect_per_category=7,                  # 시각화에 사용할 결함 이미지 수
        epochs=100,
        load_path=None,                         # 기존에 저장된 모델 파일 경로 (있다면 지정)
    )
