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

    # gradient 전파 가능하도록 @torch.no_grad() 제거
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
# 🔟 Anomaly map (5 % border 마스킹 + 정규화)
@torch.no_grad()
def anomaly_map(teacher_adapter: nn.Module,
                student_adapter: nn.Module,
                img_tensor: torch.Tensor,
                eps: float = 1e-8) -> tuple:
    """
    Returns
    -------
    amap_masked : (H, W)   – 원본 L2 거리, 가장자리 5 % 마스킹 (nan 은 0 으로 대체)
    amap_norm   : (H, W)   – 시각화용 0~1 정규화된 지도 (nan 은 마스크 처리)
    mean_score  : float    – 마스킹된 영역을 제외한 평균 anomaly score
    max_score   : float    – 마스킹된 영역을 제외한 최대 anomaly score
    """
    img_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
    img_tensor = img_tensor.to(device, non_blocking=True)

    t_feat = teacher_adapter(img_tensor)
    s_feat = student_adapter(img_tensor)
    s_feat = _align_feature_maps(t_feat, s_feat)

    # 채널 정규화
    t_feat = F.normalize(t_feat, p=2, dim=1)
    s_feat = F.normalize(s_feat, p=2, dim=1)

    # L2 거리 (정규화 없음)
    d = (t_feat - s_feat).pow(2).mean(dim=1, keepdim=True)   # (B,1,H,W)
    amap = d.squeeze().cpu().numpy()                         # (H, W)

    # 5 % border 마스킹 (nan 으로 표시)
    h, w = amap.shape
    bh, bw = int(h * 0.05), int(w * 0.05)
    mask = np.ones_like(amap, dtype=bool)
    mask[:bh, :] = False
    mask[-bh:, :] = False
    mask[:, :bw] = False
    mask[:, -bw:] = False

    amap_masked = amap.copy()
    amap_masked[~mask] = np.nan                     # 시각화·통계에서 제외

    # 통계 (nan 제외)
    valid = amap_masked[mask]
    if valid.size == 0:
        mean_score = 0.0
        max_score  = 0.0
    else:
        mean_score = float(valid.mean())
        max_score  = float(valid.max())

    # 시각화용 정규화 (nan 은 그대로 유지)
    amin, amax = np.nanmin(amap_masked), np.nanmax(amap_masked)
    if (amax - amin) > eps:
        amap_norm = (amap_masked - amin) / (amax - amin + eps)
    else:
        amap_norm = amap_masked

    return amap_masked, amap_norm, mean_score, max_score

# --------------------------------------------------------------
# 1️⃣1️⃣ 평가 지표 (max / mean 모두 지원)
def compute_metrics(good_scores_dict, defect_scores_dict, score_type="max"):
    """
    score_type : "max" 혹은 "mean"
    """
    good_scores = good_scores_dict[score_type]
    defect_scores = defect_scores_dict[score_type]

    y_true = np.array([0] * len(good_scores) + [1] * len(defect_scores))
    y_score = np.array(good_scores + defect_scores)

    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)
    thresh = np.percentile(good_scores, 95)
    preds = (y_score >= thresh).astype(int)

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "Threshold": thresh,
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "MeanScore": float(np.mean(y_score)),
        "MaxScore": float(np.max(y_score)),
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
        y, x = np.unravel_index(np.nanargmax(anomaly_map), anomaly_map.shape)
        scale_y = img_np.shape[0] / anomaly_map.shape[0]
        scale_x = img_np.shape[1] / anomaly_map.shape[1]
        y = int(y * scale_y)
        x = int(x * scale_x)

        radius = max(5, int(0.02 * max(img_np.shape[:2])))
        rr, cc = disk((y, x), radius=radius, shape=overlay.shape[:2])
        overlay[rr, cc] = [1.0, 0.0, 0.0]

    return (overlay * 255).astype(np.uint8)

# --------------------------------------------------------------
# 1️⃣3️⃣ 시각화 – 행‑열 그리드 (제목 첫 줄에 빈 줄, 5 % border 반영)
def plot_category_grid(model_name: str,
                       category: str,
                       defect_types: list,
                       good_img_path: str,
                       defect_img_paths: list,
                       gt_mask_paths: list,
                       teacher_adapter: nn.Module,
                       student_adapter: nn.Module,
                       transform,
                       metrics_max: dict,
                       metrics_mean: dict,
                       timestamp: str,
                       out_dir: str,
                       epoch: int):
    rows = len(defect_img_paths)
    cols = 6
    size = transform.transforms[0].size

    good_img_raw = Image.open(good_img_path).convert("RGB")
    good_img = good_img_raw.resize(size, Image.BILINEAR)

    fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows))

    # 제목 첫 줄에 빈 줄 삽입
    fig.suptitle(f"\n{model_name} | MVTec‑AD | Category: {category}\n\n\n",
                 fontsize=24, weight='bold', y=0.96)

    for r in range(rows):
        defect_img_raw = Image.open(defect_img_paths[r]).convert("RGB")
        defect_img = defect_img_raw.resize(size, Image.BILINEAR)

        gt_mask = Image.open(gt_mask_paths[r]).convert("L")

        img_tensor = transform(defect_img_raw).unsqueeze(0).to(device)
        _, amap_norm, mean_score, max_score = anomaly_map(teacher_adapter,
                                                          student_adapter,
                                                          img_tensor)

        overlay = apply_jet_overlay(defect_img, amap_norm, mark_max=True)

        ax = axes[r] if rows > 1 else axes

        ax[0].imshow(good_img); ax[0].set_title("Good – Original"); ax[0].axis("off")
        ax[1].imshow(defect_img); ax[1].set_title(f"Defect – {category} {defect_types[r]}"); ax[1].axis("off")
        ax[2].imshow(gt_mask, cmap="gray"); ax[2].set_title("Defect – Ground Truth"); ax[2].axis("off")

        # nan 영역을 마스크해서 지도에 표시되지 않게 함
        amap_vis = np.ma.masked_invalid(amap_norm)
        im = ax[3].imshow(amap_vis, cmap="jet"); ax[3].set_title("Defect – Anomaly Map"); ax[3].axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

        ax[4].imshow(overlay); ax[4].set_title("Defect – Overlay"); ax[4].axis("off")

        if r == 0:
            metric_txt = (
                f"AUROC (Max): {metrics_max['AUROC']:.4f}\n\n"
                f"AUPR (Max): {metrics_max['AUPR']:.4f}\n\n"
                f"Accuracy: {metrics_max['Accuracy']:.4f}\n\n"
                f"Precision: {metrics_max['Precision']:.4f}\n\n"
                f"Recall: {metrics_max['Recall']:.4f}\n\n"
                f"F1-Score: {metrics_max['F1']:.4f}\n\n"
                f"Threshold (Max): {metrics_max['Threshold']:.4f}\n\n"
                f"Max Score: {metrics_max['MaxScore']:.6f}\n\n"
                f"Mean Score: {metrics_mean['MeanScore']:.6f}"
            )
            ax[5].text(0.5, 0.5, metric_txt,
                       fontsize=14, ha='center', va='center', weight='bold',
                       bbox=dict(facecolor='white', edgecolor='black', pad=10))
        ax[5].axis('off')

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{timestamp}_{model_name}_{category}_epoch{epoch}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# --------------------------------------------------------------
# 1️⃣4️⃣ 히스토그램 – Max / Mean 각각 별도 파일 생성
def plot_histogram_by_type(category: str,
                           good_scores: list,
                           defect_scores: list,
                           threshold: float,
                           score_type: str,
                           timestamp: str,
                           model_name: str,
                           out_dir: str,
                           epoch: int):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(good_scores, bins=30, alpha=0.7, label='Good', color='green')
    ax.hist(defect_scores, bins=30, alpha=0.7, label='Defect', color='red')
    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold (95%) = {threshold:.6f}')
    ax.set_xlabel(f'Anomaly Score ({score_type.capitalize()}, Edge 5% Excluded)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f"{model_name} – {category} – {score_type.capitalize()} Score Distribution",
                 fontsize=14, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fname = f"{timestamp}_{model_name}_{category}_epoch{epoch}_hist_{score_type}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='')
    plt.close(fig)

# --------------------------------------------------------------
# 1️⃣5️⃣ 메인 파이프라인 (변경 없이 그대로 유지)
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

    if imagenet_train_path != "none" and os.path.isfile(imagenet_train_path):
        print("[info] ImageNet pretrained weight 로드:", imagenet_train_path)
        state = torch.load(imagenet_train_path, map_location=device)
        student.backbone.load_state_dict(state, strict=False)

    teacher_adapter = TeacherAdapter(teacher, out_channels=out_channels, device=device).to(device)
    student_adapter = StudentAdapter(student, out_channels=out_channels, device=device).to(device)

    optimizer = torch.optim.AdamW(student_adapter.parameters(),
                                 lr=1e-4,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # --------------------------------------------------------------
    # 1️⃣6️⃣ Training
    steps_per_epoch = len(train_loader)
    for epoch in range(1, epochs + 1):
        student_adapter.train()
        teacher_adapter.eval()
        epoch_loss = 0.0

        pbar = tqdm(range(steps_per_epoch),
                    desc=f"Epoch {epoch}/{epochs}",
                    total=steps_per_epoch)
        train_iter = iter(train_loader)

        for _ in pbar:
            (img_teacher, img_student), _, _ = next(train_iter)

            img_teacher = img_teacher.to(device, non_blocking=True)
            img_student = img_student.to(device, non_blocking=True)

            with torch.no_grad():
                t_feat = teacher_adapter(img_teacher)

            if use_amp:
                with torch.cuda.amp.autocast():
                    s_feat = student_adapter(img_student)
                    s_feat = _align_feature_maps(t_feat, s_feat)
                    loss = ((t_feat - s_feat) ** 2).mean()
                    if pretrain_penalty:
                        l2 = sum(p.pow(2).sum() for p in student_adapter.parameters())
                        loss = loss + 1e-4 * l2
            else:
                s_feat = student_adapter(img_student)
                s_feat = _align_feature_maps(t_feat, s_feat)
                loss = ((t_feat - s_feat) ** 2).mean()
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
    # 1️⃣7️⃣ Evaluation (max / mean 점수 모두 수집)
    print("\n=== Evaluation 시작 ===")
    teacher_adapter.eval()
    student_adapter.eval()

    # 점수 저장용 dict
    good_scores = {"max": [], "mean": []}
    defect_scores = {"max": [], "mean": []}
    defect_scores_by_type = {}

    for img, label, path in tqdm(test_loader, desc="Eval (collect scores)"):
        img = img.to(device)
        _, _, mean_score, max_score = anomaly_map(teacher_adapter, student_adapter, img)

        if label.item() == 0:
            good_scores["max"].append(max_score)
            good_scores["mean"].append(mean_score)
        else:
            defect_scores["max"].append(max_score)
            defect_scores["mean"].append(mean_score)
            defect_type = Path(path[0]).parts[-2]
            defect_scores_by_type.setdefault(defect_type, []).append(max_score)

    # 전체 threshold (95% percentile of good max‑scores)
    threshold_max = np.percentile(good_scores["max"], 95)
    threshold_mean = np.percentile(good_scores["mean"], 95)

    # 점수 파일 저장
    scores_dir = os.path.join(out_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    np.save(os.path.join(scores_dir, "good_max.npy"),   np.array(good_scores["max"]))
    np.save(os.path.join(scores_dir, "defect_max.npy"), np.array(defect_scores["max"]))
    np.save(os.path.join(scores_dir, "good_mean.npy"),  np.array(good_scores["mean"]))
    np.save(os.path.join(scores_dir, "defect_mean.npy"),np.array(defect_scores["mean"]))

    # 메트릭 계산 (MAX AUROC 및 MAX AUPR 출력)
    metrics_max   = compute_metrics(good_scores, defect_scores, score_type="max")
    metrics_mean  = compute_metrics(good_scores, defect_scores, score_type="mean")
    print(f"[Eval] MAX AUROC={metrics_max['AUROC']:.4f}  MAX AUPR={metrics_max['AUPR']:.4f}")

    # --------------------------------------------------------------
    # 1️⃣8️⃣ Visualization (grid + 두 종류 히스토그램)
    model_name = "EfficientAD"

    for cat in categories:
        # good 이미지 경로
        good_path = sorted(glob.glob(os.path.join(data_root, cat, "test", "good", "*.*")))[0]

        # defect 샘플 (시각화용)
        defect_img_paths, gt_mask_paths, defect_types = [], [], []
        defect_folders = [d for d in os.listdir(os.path.join(data_root, cat, "test"))
                          if d != "good" and os.path.isdir(os.path.join(data_root, cat, "test", d))]

        for folder in defect_folders:
            folder_path = os.path.join(data_root, cat, "test", folder)
            img_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
            # defect_per_category 만큼 각 폴더에서 이미지 선택
            selected = 0
            for img_path in img_files:
                if selected >= defect_per_category:
                    break
                defect_img_paths.append(img_path)
                gt_path = os.path.join(data_root, cat, "ground_truth", folder,
                                      os.path.splitext(os.path.basename(img_path))[0] + "_mask.png")
                gt_mask_paths.append(gt_path)
                defect_types.append(folder)
                selected += 1

        if not defect_img_paths:
            continue

        # ----- grid 시각화 -----
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
            metrics_max=metrics_max,
            metrics_mean=metrics_mean,
            timestamp=timestamp,
            out_dir=out_dir,
            epoch=epochs)

        # ----- max‑score 히스토그램 -----
        plot_histogram_by_type(
            category=cat,
            good_scores=good_scores["max"],
            defect_scores=defect_scores["max"],
            threshold=threshold_max,
            score_type="max",
            timestamp=timestamp,
            model_name=model_name,
            out_dir=out_dir,
            epoch=epochs)

        # ----- mean‑score 히스토그램 -----
        plot_histogram_by_type(
            category=cat,
            good_scores=good_scores["mean"],
            defect_scores=defect_scores["mean"],
            threshold=threshold_mean,
            score_type="mean",
            timestamp=timestamp,
            model_name=model_name,
            out_dir=out_dir,
            epoch=epochs)

    # --------------------------------------------------------------
    # 1️⃣9️⃣ 최종 checkpoint 저장
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
        categories=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                    "transistor", "wood", "zipper"],
        # categories=["carpet"],          # 테스트하고 싶은 카테고리
        defect_per_category=3,
        epochs=10,
        batch_size=32,
        img_size=600,
        model_size="small",
        pretrain_penalty=False,
        imagenet_train_path="none",
        seed=42,
    )
