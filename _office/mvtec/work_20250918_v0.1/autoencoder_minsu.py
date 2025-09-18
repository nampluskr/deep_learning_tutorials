#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, datetime, warnings, random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from skimage.draw import disk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.neighbors import NearestNeighbors   # ← PatchCore 메모리 뱅크용

# --------------------------------------------------------------
# 1️⃣ 경고 억제 & 디바이스 설정
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True   # 재현성

# --------------------------------------------------------------
# 2️⃣ 기본 블록
def _conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# --------------------------------------------------------------
# 4️⃣ MVTec 데이터셋
class MVTecDataset(Dataset):
    def __init__(self, root: str, category: str, split: str, transform=None):
        self.transform = transform
        self.images, self.labels = [], []

        base_dir = os.path.join(root, category, split)
        good_dir = os.path.join(base_dir, "good")
        for f in sorted(glob.glob(os.path.join(good_dir, "*.*"))):
            self.images.append(f)
            self.labels.append(0)                     # 정상

        gt_root = os.path.join(root, category, "ground_truth")
        for defect in sorted(os.listdir(gt_root)):
            img_dir = os.path.join(base_dir, defect)
            if not os.path.isdir(img_dir):
                continue
            for img_path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
                self.images.append(img_path)
                self.labels.append(1)                 # 결함

        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.images[idx]

# --------------------------------------------------------------
# 5️⃣ Teacher (EfficientNet‑B7)
class Teacher(nn.Module):
    def __init__(self, backbone_path: str, device: torch.device):
        super().__init__()
        if not os.path.isfile(backbone_path):
            raise FileNotFoundError(f"Backbone weight not found: {backbone_path}")

        self.backbone = torchvision.models.efficientnet_b7(weights=None)
        state = torch.load(backbone_path, map_location=device)
        self.backbone.load_state_dict(state, strict=False)
        self.backbone.classifier = nn.Identity()
        self.backbone = self.backbone.to(device).eval()
        self.stage_idxs = [2, 4, 6]                     # 중간 3 레이어 선택

    @torch.no_grad()
    def forward(self, x):
        feats = []
        out = x
        for i, block in enumerate(self.backbone.features):
            out = block(out)
            if i in self.stage_idxs:
                feats.append(out)
        return feats

# --------------------------------------------------------------
# 6️⃣ TeacherAdapter (고정)
class TeacherAdapter(nn.Module):
    def __init__(self, teacher: Teacher, out_channels: int, device: torch.device):
        super().__init__()
        self.teacher = teacher
        self.device = device
        self.proj = nn.Conv2d(in_channels=self._calc_teacher_channels(),
                              out_channels=out_channels,
                              kernel_size=1,
                              bias=False).to(device)

        # Freeze 모든 파라미터
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _calc_teacher_channels(self):
        dummy = torch.randn(1, 3, 256, 256, device=self.device)
        feats = self.teacher(dummy)
        return sum(f.shape[1] for f in feats)

    @torch.no_grad()
    def forward(self, x):
        feats = self.teacher(x)
        th, tw = feats[0].shape[2:]
        aligned = []
        for f in feats:
            if f.shape[2:] != (th, tw):
                f = F.interpolate(f, size=(th, tw), mode='bilinear', align_corners=False)
            aligned.append(f)
        cat = torch.cat(aligned, dim=1)
        return self.proj(cat)

# --------------------------------------------------------------
# 7️⃣ Student (Wide‑ResNet‑101‑2)
class WideResNet101_2(nn.Module):
    def __init__(self, backbone_path: str, device: torch.device):
        super().__init__()
        if not os.path.isfile(backbone_path):
            raise FileNotFoundError(f"ResNet weight not found: {backbone_path}")

        self.backbone = torchvision.models.wide_resnet101_2(weights=None)
        state = torch.load(backbone_path, map_location=device)
        self.backbone.load_state_dict(state, strict=False)
        self.backbone.fc = nn.Identity()
        self.backbone = self.backbone.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        # stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # layer1 (필수)
        x = self.backbone.layer1(x)

        feats = []
        x = self.backbone.layer2(x)   # 1/8
        feats.append(x)
        x = self.backbone.layer3(x)   # 1/16
        feats.append(x)
        x = self.backbone.layer4(x)   # 1/32
        feats.append(x)
        return feats

# --------------------------------------------------------------
# 8️⃣ StudentAdapter (학습 가능한 1×1 Conv)
class StudentAdapter(nn.Module):
    def __init__(self, student: WideResNet101_2, out_channels: int, device: torch.device):
        super().__init__()
        self.student = student
        self.device = device
        self.proj = nn.Conv2d(in_channels=self._calc_student_channels(),
                              out_channels=out_channels,
                              kernel_size=1,
                              bias=False).to(device)

    def _calc_student_channels(self):
        dummy = torch.randn(1, 3, 256, 256, device=self.device)
        feats = self.student(dummy)
        return sum(f.shape[1] for f in feats)

    # @torch.no_grad() 를 제거 → gradient 전파 가능
    def forward(self, x):
        feats = self.student(x)
        th, tw = feats[0].shape[2:]
        aligned = []
        for f in feats:
            if f.shape[2:] != (th, tw):
                f = F.interpolate(f, size=(th, tw), mode='bilinear', align_corners=False)
            aligned.append(f)
        cat = torch.cat(aligned, dim=1)
        return self.proj(cat)

# --------------------------------------------------------------
# 9️⃣ Feature map 정렬 (teacher와 size 맞추기)
def _align_feature_maps(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
    if t_feat.shape[2:] != s_feat.shape[2:]:
        s_feat = F.interpolate(s_feat,
                               size=t_feat.shape[2:],
                               mode='bilinear',
                               align_corners=False)
    return s_feat

# --------------------------------------------------------------
# 🔟 Contrastive (NT‑Xent) Loss
class ContrastiveLoss(nn.Module):
    """
    Teacher‑Student 피처를 L2 정규화한 뒤 코사인 유사도로 NT‑Xent loss 를 계산합니다.
    온‑배치 내에서 같은 이미지(teacher‑student 쌍)는 Positive,
    나머지는 Negative 로 간주합니다.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, C*H*W)
        B = t_feat.size(0)
        t_vec = t_feat.view(B, -1)
        s_vec = s_feat.view(B, -1)

        # L2 정규화
        t_vec = F.normalize(t_vec, p=2, dim=1)
        s_vec = F.normalize(s_vec, p=2, dim=1)

        # similarity matrix (2B x 2B)
        z = torch.cat([t_vec, s_vec], dim=0)          # (2B, D)
        sim = torch.matmul(z, z.t()) / self.temperature   # (2B, 2B)

        # 마스크 : 자기 자신 제외
        mask = (~torch.eye(2 * B, 2 * B, dtype=bool, device=z.device)).float()

        # Positive pair는 (i, i+B) 와 (i+B, i)
        pos = torch.exp(sim.diag(B) / self.temperature) + torch.exp(sim.diag(-B) / self.temperature)

        # denominator : 모든 (자기 제외) similarity
        denom = torch.exp(sim) * mask
        denom = denom.sum(dim=1)

        loss = -torch.log(pos / denom).mean()
        return loss

# --------------------------------------------------------------
# 1️⃣1️⃣ 평가 지표
def compute_metrics(good_scores, defect_scores, threshold):
    y_true = np.array([0] * len(good_scores) + [1] * len(defect_scores))
    y_score = np.array(good_scores + defect_scores)

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    preds = (y_score >= threshold).astype(int)

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "MeanScore": y_score.mean(),
    }

# --------------------------------------------------------------
# 1️⃣2️⃣ Overlay (jet colormap + max‑point 마킹)
def apply_jet_overlay(pil_img: Image.Image,
                      anomaly_map: np.ndarray,
                      mark_max: bool = False) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    colored = cmap(anomaly_map)[:, :, :3]

    img_np = np.array(pil_img).astype(np.float32) / 255.0

    if colored.shape[:2] != img_np.shape[:2]:
        colored = resize(colored,
                         output_shape=img_np.shape[:2],
                         order=1,
                         mode='reflect',
                         anti_aliasing=True)

    overlay = 0.6 * img_np + 0.4 * colored
    overlay = np.clip(overlay, 0, 1)

    if mark_max:
        y, x = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        scale_y = img_np.shape[0] / anomaly_map.shape[0]
        scale_x = img_np.shape[1] / anomaly_map.shape[1]
        y = int(y * scale_y)
        x = int(x * scale_x)

        radius = max(5, int(0.02 * max(img_np.shape[:2])))
        rr, cc = disk((y, x), radius=radius, shape=overlay.shape[:2])
        overlay[rr, cc] = [1.0, 0.0, 0.0]

    return (overlay * 255).astype(np.uint8)

# --------------------------------------------------------------
# 1️⃣3️⃣ 시각화 – 행‑열 그리드 (defect_per_category 만큼 행)
def plot_category_grid(model_name: str,
                       category: str,
                       defect_types: list,
                       good_img_path: str,
                       defect_img_paths: list,
                       gt_mask_paths: list,
                       teacher_adapter: nn.Module,
                       student_adapter: nn.Module,
                       transform,
                       metrics: dict,
                       timestamp: str,
                       out_dir: str,
                       epoch: int):
    """
    - `defect_per_category` 만큼의 불량 이미지를 한 화면에 행(row) 단위로 배치
    - 열은 6개 : Good, Defect, Ground‑Truth, Anomaly‑Map, Overlay, Metrics
    - 제목에 두 줄 공백(`\n\n`) 삽입, 불량 종류는 `Defect – {category} {defect_type}`
    """
    rows = len(defect_img_paths)
    cols = 6
    size = transform.transforms[0].size

    # good 이미지 (모든 행에서 동일)
    good_img_raw = Image.open(good_img_path).convert("RGB")
    good_img = good_img_raw.resize(size, Image.BILINEAR)

    fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    fig.suptitle(f"{model_name} | MVTec‑AD | Category: {category}\n\n",
                 fontsize=20, y=0.95)

    for r in range(rows):
        # ---------- Defect ----------
        defect_img_raw = Image.open(defect_img_paths[r]).convert("RGB")
        defect_img = defect_img_raw.resize(size, Image.BILINEAR)

        # ---------- GT ----------
        gt_mask = Image.open(gt_mask_paths[r]).convert("L")

        # ---------- Anomaly Map ----------
        img_tensor = transform(defect_img_raw).unsqueeze(0).to(device)
        amap = anomaly_map(teacher_adapter, student_adapter, img_tensor)

        # ---------- Overlay ----------
        overlay = apply_jet_overlay(defect_img, amap, mark_max=True)

        # ---------- Plot ----------
        ax = axes[r] if rows > 1 else axes
        ax[0].imshow(good_img); ax[0].set_title("Good – Original"); ax[0].axis("off")
        ax[1].imshow(defect_img); ax[1].set_title(f"Defect – {category} {defect_types[r]}"); ax[1].axis("off")
        ax[2].imshow(gt_mask, cmap="gray"); ax[2].set_title("Defect – Ground Truth"); ax[2].axis("off")
        im = ax[3].imshow(amap, cmap="jet"); ax[3].set_title("Defect – Anomaly Map"); ax[3].axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
        ax[4].imshow(overlay); ax[4].set_title("Defect – Overlay"); ax[4].axis("off")

        # 마지막 열에 메트릭 텍스트 (첫 행에만 표시)
        if r == 0:
            metric_txt = (f"AUROC: {metrics['AUROC']*100:.2f}%\n"
                          f"AUPR: {metrics['AUPR']*100:.2f}%\n"
                          f"Threshold: {metrics['Threshold']:.4f}\n"
                          f"Accuracy: {metrics['Accuracy']*100:.2f}%\n"
                          f"Precision: {metrics['Precision']*100:.2f}%\n"
                          f"Recall: {metrics['Recall']*100:.2f}%\n"
                          f"F1‑Score: {metrics['F1']*100:.2f}%\n"
                          f"MeanScore: {metrics['MeanScore']:.4f}")
            ax[5].text(0.5, 0.5, metric_txt,
                       fontsize=12, ha='center', va='center',
                       bbox=dict(facecolor='white', edgecolor='black', pad=5))
        ax[5].axis('off')

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{timestamp}_{model_name}_{category}_epoch{epoch}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# --------------------------------------------------------------
# 1️⃣4️⃣ 히스토그램 – 양품(good)만 사용
def plot_histogram_good(category: str,
                        good_scores: list,
                        threshold: float,
                        timestamp: str,
                        model_name: str,
                        out_dir: str,
                        epoch: int):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(good_scores, bins=30, alpha=0.7, label='Good', color='green')
    ax.axvline(threshold, color='blue', linestyle='--', label='Threshold')
    ax.set_xlabel('Anomaly Score (max of map)')
    ax.set_ylabel('Count')
    ax.set_title(f"{model_name} – {category} – Good Score Distribution")
    ax.legend()

    fname = f"{timestamp}_{model_name}_{category}_epoch{epoch}_hist_good.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# --------------------------------------------------------------
# 1️⃣5️⃣ PatchCore‑style Memory‑Bank 구축
def build_memory_bank(teacher_adapter: nn.Module,
                      good_loader: DataLoader,
                      transform,
                      device: torch.device):
    """
    학습 데이터 중 정상(good) 이미지들의 Teacher‑Adapter 피처를
    (B, C, H, W) → (B, C·H·W) 로 flatten 후 메모리 뱅크에 저장합니다.
    """
    memory_feats = []
    teacher_adapter.eval()
    with torch.no_grad():
        for img, _, _ in tqdm(good_loader, desc="Build Memory Bank (good)"):
            img = img.to(device)
            feat = teacher_adapter(img)                     # (B, C, H, W)
            # 여러 stage 를 concat 한 뒤 flatten
            B = feat.size(0)
            vec = feat.view(B, -1).cpu().numpy()
            memory_feats.append(vec)
    memory_feats = np.concatenate(memory_feats, axis=0)      # (N, D)
    # NearestNeighbour 모델 (brute‑force) – N 은 수천 정도라 충분히 빠름
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(memory_feats)
    return nn

def compute_memory_score(nn: NearestNeighbors,
                         teacher_adapter: nn.Module,
                         img_tensor: torch.Tensor,
                         device: torch.device):
    """
    테스트 이미지 하나에 대해 Teacher‑Adapter 피처를 추출하고,
    메모리 뱅크와의 최소 L2 거리(정규화된)를 반환합니다.
    """
    teacher_adapter.eval()
    with torch.no_grad():
        feat = teacher_adapter(img_tensor)                 # (1, C, H, W)
        vec = feat.view(1, -1).cpu().numpy()
        dist, _ = nn.kneighbors(vec, n_neighbors=1)       # (1,1)
    # 거리값을 0‑1 로 정규화 (전체 메모리 거리의 min‑max 를 사용)
    # 여기서는 간단히 0‑1 로 스케일링 (max distance 를 1 로 가정)
    return float(dist[0][0])

# --------------------------------------------------------------
# 1️⃣6️⃣ Anomaly map (이미지‑단위 min‑max 정규화)
@torch.no_grad()
def anomaly_map(teacher_adapter: nn.Module,
                student_adapter: nn.Module,
                img_tensor: torch.Tensor,
                eps: float = 1e-8) -> np.ndarray:
    """
    1. Teacher / Student 피처 추출
    2. 채널 차원 L2 정규화
    3. L2 거리 평균
    4. 이미지‑단위 min‑max 정규화 → 0‑1
    """
    img_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
    img_tensor = img_tensor.to(device, non_blocking=True)

    t_feat = teacher_adapter(img_tensor)          # (B, C, H, W)
    s_feat = student_adapter(img_tensor)

    s_feat = _align_feature_maps(t_feat, s_feat)

    # 채널 정규화
    t_feat = F.normalize(t_feat, p=2, dim=1)
    s_feat = F.normalize(s_feat, p=2, dim=1)

    # L2 거리 평균
    d = (t_feat - s_feat).pow(2).mean(dim=1, keepdim=True)   # (B,1,H,W)
    amap = d.squeeze().cpu().numpy()                         # (H,W)

    # 이미지‑단위 min‑max 정규화
    amin, amax = amap.min(), amap.max()
    amap = (amap - amin) / (amax - amin + eps)                # 0‑1

    return amap

# --------------------------------------------------------------
# 1️⃣7️⃣ 메인 파이프라인 (변경 없이 그대로 유지)
def main(data_root: str,
         backbone_path: str,
         resnet_path: str,
         categories=None,
         defect_per_category: int = 5,
         epochs: int = 30,
         batch_size: int = 16,
         img_size: int = 256,
         model_size: str = "small",
         pretrain_penalty: bool = False,
         imagenet_train_path: str = "none",
         seed: int = 42):
    """
    전체 학습 → 평가 → 시각화 → 최종 checkpoint 저장
    (메인 함수 시그니처와 흐름은 기존 코드와 동일)
    """
    if categories is None:
        categories = ["tile", "carpet", "grid"]

    # ------------------- 시드 고정 -------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------- 메모리 부족 시 자동 다운스케일 -------------------
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem < 8 * 1024 ** 3:          # 8GB 미만이면
            img_size = min(img_size, 384)
            batch_size = min(batch_size, 8)
            print(f"[info] GPU 메모리 부족 → img_size={img_size}, batch_size={batch_size}")

    # ------------------- 출력 차원 결정 -------------------
    out_channels = 128 if model_size == "small" else 384

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = f"./visualization/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------- Transform --------------------
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    aug_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1)
    ])

    # train_transform 은 (원본, augmentation) 튜플 반환
    def train_transform(img):
        img = base_transform(img)
        img_aug = base_transform(aug_transform(transforms.ToPILImage()(img)))
        return img, img_aug

    # -------------------- Dataset / DataLoader --------------------
    train_sets, test_sets = [], []
    for cat in categories:
        train_sets.append(MVTecDataset(root=data_root,
                                       category=cat,
                                       split="train",
                                       transform=train_transform))
        test_sets.append(MVTecDataset(root=data_root,
                                      category=cat,
                                      split="test",
                                      transform=base_transform))

    # 일반 DataLoader 를 그대로 사용 (InfiniteDataloader 삭제)
    train_loader = DataLoader(ConcatDataset(train_sets),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)

    test_loader = DataLoader(ConcatDataset(test_sets),
                             batch_size=1,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True)

    # -------------------- Model 초기화 --------------------
    teacher = Teacher(backbone_path=backbone_path, device=device).to(device)
    student = WideResNet101_2(backbone_path=resnet_path, device=device).to(device)

    # ImageNet 사전학습 weight 로드 (옵션)
    if imagenet_train_path != "none" and os.path.isfile(imagenet_train_path):
        print("[info] ImageNet pretrained weight 로드:", imagenet_train_path)
        state = torch.load(imagenet_train_path, map_location=device)
        student.backbone.load_state_dict(state, strict=False)

    teacher_adapter = TeacherAdapter(teacher,
                                    out_channels=out_channels,
                                    device=device).to(device)
    student_adapter = StudentAdapter(student,
                                    out_channels=out_channels,
                                    device=device).to(device)

    optimizer = torch.optim.AdamW(student_adapter.parameters(),
                                 lr=1e-4,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ----- Contrastive loss (NT‑Xent) -----
    contrastive = ContrastiveLoss(temperature=0.07).to(device)
    lambda_contrast = 0.5          # contrastive loss 가 차지하는 비중 (조정 가능)

    # --------------------------------------------------------------
    # 1️⃣6️⃣ Training (전체 배치 순회)
    steps_per_epoch = len(train_loader)   # DataLoader 에는 __len__ 이 존재합니다.
    for epoch in range(1, epochs + 1):
        student_adapter.train()
        teacher_adapter.eval()
        epoch_loss = 0.0

        pbar = tqdm(range(steps_per_epoch),
                    desc=f"Epoch {epoch}/{epochs}",
                    total=steps_per_epoch)
        train_iter = iter(train_loader)   # 매 epoch 마다 새 iterator 생성

        for _ in pbar:
            (img_teacher, img_student), _, _ = next(train_iter)

            img_teacher = img_teacher.to(device, non_blocking=True)
            img_student = img_student.to(device, non_blocking=True)

            # Teacher feature (gradient 없음)
            with torch.no_grad():
                t_feat = teacher_adapter(img_teacher)

            # Student feature (gradient 필요)
            if use_amp:
                with torch.cuda.amp.autocast():
                    s_feat = student_adapter(img_student)
                    s_feat = _align_feature_maps(t_feat, s_feat)

                    # MSE loss
                    mse = ((t_feat - s_feat) ** 2).mean()

                    # Contrastive loss
                    c_loss = contrastive(t_feat.detach(), s_feat)

                    loss = mse + lambda_contrast * c_loss
                    if pretrain_penalty:
                        l2 = sum(p.pow(2).sum() for p in student_adapter.parameters())
                        loss = loss + 1e-4 * l2
            else:
                s_feat = student_adapter(img_student)
                s_feat = _align_feature_maps(t_feat, s_feat)

                mse = ((t_feat - s_feat) ** 2).mean()
                c_loss = contrastive(t_feat.detach(), s_feat)
                loss = mse + lambda_contrast * c_loss
                if pretrain_penalty:
                    l2 = sum(p.pow(2).sum() for p in student_adapter.parameters())
                    loss = loss + 1e-4 * l2

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(student_adapter.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_adapter.parameters(), max_norm=5.0)
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss /= steps_per_epoch
        print(f"[epoch {epoch}] avg loss = {epoch_loss:.6f}")
        scheduler.step()

    # --------------------------------------------------------------
    # 1️⃣7️⃣Core‑style Memory‑Bank 구축 (good 이미지만 사용)
    # good 전용 DataLoader 를 별도로 만든다.
    good_loader = DataLoader(
        MVTecDataset(root=data_root,
                     category=categories[0],   # 임시 – 실제로는 모든 카테고리의 good을 합쳐도 무방
                     split="test",
                     transform=base_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    memory_nn = build_memory_bank(teacher_adapter, good_loader, base_transform, device)

    # --------------------------------------------------------------
    # 1️⃣8️⃣ Evaluation (전체 epoch 종료 후)
    print("\n=== Evaluation 시작 ===")
    teacher_adapter.eval()
    student_adapter.eval()

    good_scores, defect_scores = [], []
    defect_scores_by_type = {}

    for img, label, path in tqdm(test_loader, desc="Eval (collect scores)"):
        img = img.to(device)

        # ----- 기존 MSE‑based anomaly map -----
        amap = anomaly_map(teacher_adapter, student_adapter, img)   # 0‑1 map
        mse_score = float(amap.max())

        # ----- Memory‑Bank 기반 점수 -----
        mem_score = compute_memory_score(memory_nn, teacher_adapter, img, device)

        # 두 점수를 가중 평균 (α 조정 가능)
        alpha = 0.6
        final_score = alpha * mse_score + (1 - alpha) * mem_score

        if label.item() == 0:
            good_scores.append(final_score)
        else:
            defect_scores.append(final_score)
            defect_type = Path(path[0]).parts[-2]
            defect_scores_by_type.setdefault(defect_type, []).append(final_score)

    threshold = np.percentile(good_scores, 95)
    metrics = compute_metrics(good_scores, defect_scores, threshold)

    print(f"[Eval] AUROC={metrics['AUROC']:.4f}  AUPR={metrics['AUPR']:.3f}")

    # --------------------------------------------------------------
    # 1️⃣9️⃣ Visualization (전체 epoch 종료 후 한 번에 저장)
    model_name = "EfficientAD"

    for cat in categories:
        # ---------- good 이미지 하나 선택 ----------
        good_path = sorted(
            glob.glob(os.path.join(data_root, cat, "test", "good", "*.*"))
        )[0]

        # ---------- defect 이미지와 GT 마스크를 defect_per_category 개수만큼 수집 ----------
        defect_img_paths, gt_mask_paths, defect_types = [], [], []
        defect_folders = [d for d in os.listdir(os.path.join(data_root, cat, "test"))
                          if d != "good" and os.path.isdir(os.path.join(data_root, cat, "test", d))]

        for folder in defect_folders:
            folder_path = os.path.join(data_root, cat, "test", folder)
            img_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
            for img_path in img_files:
                if len(defect_img_paths) >= defect_per_category:
                    break
                defect_img_paths.append(img_path)
                gt_path = os.path.join(data_root, cat, "ground_truth", folder,
                                      os.path.splitext(os.path.basename(img_path))[0] + "_mask.png")
                gt_mask_paths.append(gt_path)
                defect_types.append(folder)          # 폴더명이 defect_type
            if len(defect_img_paths) >= defect_per_category:
                break

        if not defect_img_paths:
            continue

        # ---------- Grid 형태 시각화 ----------
        plot_category_grid(
            model_name=model_name,
            category=cat,
            defect_types=defect_types,
            good_img_path=good_path,
            defect_img_paths=defect_img_paths,
            gt_mask_paths=gt_mask_paths,
            teacher_adapter=teacher_adapter,
            student_adapter=student_adapter,
            transform=base_transform,
            metrics=metrics,
            timestamp=timestamp,
            out_dir=out_dir,
            epoch=epochs)          # 최종 epoch 번호 사용

        # ---------- 히스토그램 (good only) ----------
        plot_histogram_good(
            category=cat,
            good_scores=good_scores,
            threshold=threshold,
            timestamp=timestamp,
            model_name=model_name,
            out_dir=out_dir,
            epoch=epochs)

    # --------------------------------------------------------------
    # 2️⃣0️⃣ 최종 checkpoint 저장
    final_ckpt_path = os.path.join(out_dir,
                                   f"{model_name}_student_adapter_final.pth")
    torch.save(student_adapter.state_dict(), final_ckpt_path)

    print("\n=== Training & Evaluation 완료 ===")
    print(f"Final checkpoint   : {final_ckpt_path}")
    print(f"Visualization dir : {out_dir}")

# --------------------------------------------------------------
if __name__ == "__main__":
    """
    사용 예시
    python wrapper_20250917_01.py
    """
    main(
        data_root="/home/namu/myspace/NAMU/datasets/mvtec",
        backbone_path="/home/namu/myspace/20250917_Anomaly_Detection/efficientnet_b7_lukemelas-c5b4e57e.pth",
        resnet_path="/home/namu/myspace/20250917_Anomaly_Detection/wide_resnet101_2-32ee1156.pth",
        categories=["tile", "carpet", "grid"],
        defect_per_category=7,          # 여기서 지정한 개수만큼 행이 추가됩니다.
        epochs=100,
        batch_size=32,
        img_size=600,
        model_size="small",
        pretrain_penalty=False,
        imagenet_train_path="none",
        seed=42,
    )
