import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from time import time
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def run_epoch(self, train_loader, epoch, num_epochs, mode='train'):
        raise NotImplementedError

    def train(self, train_loader, epoch, num_epochs):
        self.model.train()
        desc = f" [{epoch:3d}/{num_epochs}] Training"
        return self.run_epoch(train_loader, mode='train', desc=desc)

    @torch.no_grad()
    def validate(self, valid_loader, epoch, num_epochs):
        self.model.eval()
        desc = f" [{epoch:3d}/{num_epochs}] Validation"
        return self.run_epoch(valid_loader, mode='valid', desc=desc)

    def fit(self, train_loader, num_epochs, valid_loader=None, weight_path=None):
        print("\n > Start training...\n")
        history = {name: [] for name in ['loss'] + list(self.metrics)}
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metrics)})

        train_start_time = time()
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time()
            train_results = self.train(train_loader, epoch, num_epochs)
            train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

            for name, value in train_results.items():
                history[name].append(value)

            if valid_loader is not None:
                valid_results = self.validate(valid_loader, epoch, num_epochs)
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

                for name, value in valid_results.items():
                    history[f"val_{name}"].append(value)

                print(f" [{epoch:3d}/{num_epochs}] {train_info} | (val) {valid_info} ({time() - epoch_start_time:.1f}s)")

                if epoch % 10 == 0:
                    for method in ["f1", "roc", "percentile"]:
                        eval_img = self.evaluate_image_level(valid_loader, method=method)
                        img_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_img.items() if isinstance(v, float)])
                        img_info2 = ", ".join([f"{k}={v:2d}" for k, v in eval_img.items() if isinstance(v, int)])
                        print(f" > Image-level: {img_info1} | {img_info2} ({method})")

                    print()
                    # eval_pix = self.evaluate_pixel_level(valid_loader, percentile=95)
                    # pix_info = ", ".join([f"{k}={v:.3f}" for k, v in eval_pix.items() if isinstance(v, (int, float))])
                    # print(f" > Pixel-level: {pix_info}\n")
            else:
                print(f" [{epoch:3d}/{num_epochs}] {train_info} ({time() - epoch_start_time:.1f}s)")

        elapsed_time = time() - train_start_time
        hours, reminder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(reminder, 60)
        print(f" > Training finished... in {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        self.save_model(weight_path)
        return history

    @torch.no_grad()
    def evaluate_image_level(self, loader, method="f1", percentile=95):
        """Evaluate image-level anomaly detection (with GT labels)."""
        all_scores, all_labels = [], []
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].cpu()

            prediction = self.model.predict(images)
            scores = prediction["pred_score"].cpu()
            if scores.ndim > 1:
                scores = scores.view(scores.size(0))

            all_scores.append(scores)
            all_labels.append(labels)

        scores = torch.cat(all_scores)   # shape = [N_images]
        labels = torch.cat(all_labels)   # shape = [N_images]

        results = {}
        results["auroc"] = roc_auc_score(labels, scores)
        results["aupr"] = average_precision_score(labels, scores)

        threshold = compute_threshold(scores, labels, method=method, percentile=percentile)
        preds = (scores >= threshold).long()

        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        results.update({
            "th": float(threshold),
            "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
            "prec": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp, 
        })
        return results

    @torch.no_grad()
    def evaluate_pixel_level(self, loader, percentile=95):
        """Evaluate pixel-level anomaly detection (requires GT masks)."""
        results = {}
        all_scores, all_labels = [], []
        iou_list, dice_list, pro_scores = [], [], []

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].cpu().numpy()
            prediction = self.model.predict(images)
            anomaly_maps = prediction["anomaly_map"].cpu().numpy()

            # anomaly_maps 차원 처리
            if anomaly_maps.ndim == 4:
                if anomaly_maps.shape[1] == 1:
                    anomaly_maps = anomaly_maps.squeeze(1)
                else:
                    anomaly_maps = anomaly_maps.mean(axis=1)

            # masks 차원 처리
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            # **수정된 부분: 각 이미지별로 개별 처리하여 크기 일치 보장**
            for i in range(len(masks)):
                amap = anomaly_maps[i]  # (H, W)
                gt_mask = masks[i]      # (H, W)

                # 크기 맞추기
                if amap.shape != gt_mask.shape:
                    from skimage.transform import resize
                    amap = resize(amap, gt_mask.shape, order=1, mode='reflect', anti_aliasing=True)

                # **크기가 일치하는 것을 확인한 후 픽셀별 점수 수집**
                all_scores.append(torch.from_numpy(amap.flatten()))
                all_labels.append(torch.from_numpy(gt_mask.flatten()))

                # threshold 계산 및 prediction mask 생성
                thr = np.percentile(amap.ravel(), percentile)
                pred_mask = (amap >= thr).astype(np.uint8)
                gt_mask_uint8 = gt_mask.astype(np.uint8)

                # IoU, Dice 계산
                inter = np.logical_and(pred_mask, gt_mask_uint8).sum()
                union = np.logical_or(pred_mask, gt_mask_uint8).sum()
                iou = inter / (union + 1e-8)
                dice = 2 * inter / (pred_mask.sum() + gt_mask_uint8.sum() + 1e-8)

                iou_list.append(iou)
                dice_list.append(dice)

                # PRO metric 계산
                from skimage import measure
                labeled_mask = measure.label(gt_mask_uint8, connectivity=2)
                thresholds = np.linspace(0, 1, 50)
                pros, fprs = [], []

                amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                for thr_ in thresholds:
                    pred = (amap_norm >= thr_).astype(np.uint8)
                    fp = np.logical_and(pred == 1, gt_mask_uint8 == 0).sum()
                    tn = (gt_mask_uint8 == 0).sum()
                    fpr = fp / (fp + tn + 1e-8)

                    region_overlaps = []
                    for rid in range(1, labeled_mask.max() + 1):
                        region = (labeled_mask == rid)
                        inter = np.logical_and(pred, region).sum()
                        region_overlaps.append(inter / (region.sum() + 1e-8))
                    pros.append(np.mean(region_overlaps) if region_overlaps else 0.0)
                    fprs.append(fpr)

                fprs, pros = np.array(fprs), np.array(pros)
                mask_valid = fprs <= 0.3
                if mask_valid.sum() > 1:
                    order = np.argsort(fprs[mask_valid])
                    fprs_sorted = fprs[mask_valid][order]
                    pros_sorted = pros[mask_valid][order]
                    auc_pro = np.trapz(pros_sorted, fprs_sorted) / 0.3
                else:
                    auc_pro = 0.0
                pro_scores.append(auc_pro)

        # 전체 배치 합쳐서 AUROC/AUPR 계산
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        # **디버깅용 출력 (선택사항)**
        # print(f"Debug: scores shape={scores.shape}, labels shape={labels.shape}")

        from sklearn.metrics import roc_auc_score, average_precision_score
        results["auroc"] = roc_auc_score(labels, scores)
        results["aupr"] = average_precision_score(labels, scores)

        # 평균 IoU, Dice, PRO
        results["iou"] = float(np.mean(iou_list)) if iou_list else float("nan")
        results["dice"] = float(np.mean(dice_list)) if dice_list else float("nan")
        results["pro"] = float(np.mean(pro_scores)) if pro_scores else float("nan")
        return results

    def save_model(self, weight_path):
        if weight_path is not None:
            output_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), weight_path)
            print(f" > Model weights saved to: {weight_path}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f" > Model weights loaded from: {weight_path}\n")
        else:
            print(f" > No model weights found at: {weight_path}\n")

    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        num_images = 0

        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            images = batch["image"].to(self.device)
            masks = batch["mask"].cpu().numpy() if "mask" in batch else None

            recon, *_ = self.model(images)
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
                original = denormalize(images[i].cpu()).clamp(0, 1)
                reconstructed = denormalize(recon[i].cpu()).clamp(0, 1)
                amap = anomaly_maps[i]
                anomaly_map = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                binary_mask = None
                if masks is not None:
                    mask = masks[i]
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    binary_mask = mask

                if binary_mask is not None:
                    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                    titles = [f"Original: {label}", "Reconstructed", "Mask", f"Anomaly: {score:.4f}"]
                    images_vis = [original, reconstructed, binary_mask, anomaly_map]
                    cmaps = [None, None, "gray", "jet"]
                else:
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    titles = [f"Original: {label}", "Reconstructed", f"Anomaly: {score:.4f}"]
                    images_vis = [original, reconstructed, anomaly_map]
                    cmaps = [None, None, "jet"]

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

    @torch.no_grad()
    def test_feature_based(self, test_loader, output_dir=None, 
        show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        num_images = 0

        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            images = batch["image"].to(self.device)
            masks = batch["mask"].cpu().numpy() if "mask" in batch else None

            model_output = self.model(images)

            if len(model_output) == 2:                     # STFPM, Manual
                teacher_features, student_features = model_output
                reconstructed = None
            elif len(model_output) == 3:                   # EfficientAD
                reconstructed, teacher_features, student_features = model_output
            else:
                raise ValueError(
                    f"Unexpected model output length {len(model_output)}. "
                    "Supported: 2 (teacher, student) or 3 (recon, teacher, student)."
                )

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
                original = denormalize(images[i].cpu()).clamp(0, 1)

                amap = anomaly_maps[i]
                if amap.ndim == 3 and amap.shape[0] == 1:
                    amap = amap[0]                     # (H, W)
                anomaly_map_vis = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                # teacher feature
                if isinstance(teacher_features, dict):
                    # dict → 첫 번째 레이어 선택
                    first_key = next(iter(teacher_features.keys()))
                    teacher_feat = teacher_features[first_key][i]   # (C, H, W) 혹은 (D,)
                elif isinstance(teacher_features, (list, tuple)):
                    teacher_feat = teacher_features[0][i]
                elif isinstance(teacher_features, torch.Tensor):
                    teacher_feat = teacher_features[i]
                else:
                    raise TypeError(f"Unsupported teacher feature type: {type(teacher_features)}")

                # student feature
                if isinstance(student_features, dict):
                    first_key = next(iter(student_features.keys()))
                    student_feat = student_features[first_key][i]
                elif isinstance(student_features, (list, tuple)):
                    student_feat = student_features[0][i]
                elif isinstance(student_features, torch.Tensor):
                    student_feat = student_features[i]
                else:
                    raise TypeError(f"Unsupported student feature type: {type(student_features)}")

                def to_2d(feat):
                    if feat.dim() == 3:                 # (C, H, W)
                        feat = torch.mean(feat, dim=0)  # (H, W)
                    elif feat.dim() == 4:               # (B, C, H, W)
                        feat = torch.mean(feat, dim=0).squeeze(0)
                    arr = feat.cpu().numpy()
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                    return arr

                teacher_feat_vis = to_2d(teacher_feat)
                student_feat_vis = to_2d(student_feat)

                binary_mask = None
                if masks is not None:
                    mask = masks[i]
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    binary_mask = mask

                if binary_mask is not None:
                    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                    titles = [
                        f"Original: {label}",
                        "Mask",
                        "Teacher Feature",
                        "Student Feature",
                        f"Anomaly: {score:.4f}",
                    ]
                    imgs = [
                        original,
                        binary_mask,
                        teacher_feat_vis,
                        student_feat_vis,
                        anomaly_map_vis,
                    ]
                    cmaps = [None, "gray", "viridis", "viridis", "jet"]
                else:
                    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                    titles = [
                        f"Original: {label}",
                        "Teacher Feature",
                        "Student Feature",
                        f"Anomaly: {score:.4f}",
                    ]
                    imgs = [
                        original,
                        teacher_feat_vis,
                        student_feat_vis,
                        anomaly_map_vis,
                    ]
                    cmaps = [None, "viridis", "viridis", "jet"]

                for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
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

        return {"num_images": num_images}


#############################################################
# Trainer for AutoEncoder Model
#############################################################

class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5 )
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metrics = metrics or {}

    def run_epoch(self, loader, mode='train', desc=""):
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0
        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                batch_size = images.size(0)
                num_images += batch_size

                recon, *_ = self.model(images)
                loss = self.loss_fn(recon, images)

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                results["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        results[name] += metric_fn(recon, images).item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})
        return {name: value / num_images for name, value in results.items()}



#############################################################
# Trainer for EfficientAD Model
#############################################################

class EfficientADTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        from model_efficientad import EfficientADLoss, EfficientADMetric

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        trainable_params = []
        trainable_params.extend(self.model.student_encoder.parameters())
        trainable_params.extend(self.model.decoder.parameters())

        self.optimizer = optimizer or optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        self.loss_fn = loss_fn or EfficientADLoss()
        self.metrics = metrics or {'efficientad_metric': EfficientADMetric()}

    def run_epoch(self, loader, mode='train', desc=""):
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0

        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                batch_size = images.size(0)
                num_images += batch_size

                reconstructed, teacher_features, student_features = self.model(images)
                loss = self.loss_fn(reconstructed, images, teacher_features, student_features)

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                results["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        if name == 'efficientad_metric':
                            metric_dict = metric_fn(reconstructed, images, teacher_features, student_features)
                            metric_value = metric_dict['recon_mse']
                        else:
                            metric_value = metric_fn(reconstructed, images, teacher_features, student_features)

                        results[name] += metric_value.item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})

        return {name: value / num_images for name, value in results.items()}

    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):
        return self.test_feature_based(test_loader, output_dir, show_image, img_prefix,
            skip_normal, skip_anomaly, num_max)


#############################################################
# Trainer for Manual EfficientAD Model
#############################################################

class ManualTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None,
                 use_amp=True, pretrain_penalty=False):
        from model_manual import ManualEfficientADLoss, ManualEfficientADMetric

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.pretrain_penalty = pretrain_penalty

        # Only optimize student adapter parameters (teacher is frozen)
        student_params = list(self.model.student_adapter.parameters())
        self.optimizer = optimizer or optim.AdamW(student_params, lr=1e-4, weight_decay=1e-5)
        self.loss_fn = loss_fn or ManualEfficientADLoss(pretrain_penalty=pretrain_penalty)
        self.metrics = metrics or {'similarity': ManualEfficientADMetric()}

        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def run_epoch(self, loader, mode='train', desc=""):
        """Run one epoch of training or validation."""
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0

        # Set model mode
        if mode == 'train':
            self.model.student_adapter.train()
            self.model.teacher_adapter.eval()
        else:
            self.model.eval()

        with tqdm(loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                else:
                    # Handle tuple format (img, label, path) or (img_teacher, img_student), label, path
                    if len(batch) == 3 and isinstance(batch[0], tuple):
                        # Training format with augmentation: ((img_teacher, img_student), label, path)
                        img_teacher, img_student = batch[0]
                        img_teacher = img_teacher.to(self.device, non_blocking=True)
                        img_student = img_student.to(self.device, non_blocking=True)
                        # Use student image for processing
                        images = img_student
                    else:
                        # Standard format: (img, label, path)
                        images = batch[0].to(self.device)

                batch_size = images.size(0)
                num_images += batch_size

                if mode == 'train' and self.use_amp:
                    with torch.cuda.amp.autocast():
                        teacher_features, student_features = self.model(images)
                        loss = self.loss_fn(teacher_features, student_features,
                                            model=self.model if self.pretrain_penalty else None)
                else:
                    teacher_features, student_features = self.model(images)
                    loss = self.loss_fn(teacher_features, student_features,
                                      model=self.model if self.pretrain_penalty else None)

                if mode == 'train':
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.student_adapter.parameters(), max_norm=5.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.student_adapter.parameters(), max_norm=5.0)
                        self.optimizer.step()

                results["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        metric_value = metric_fn(teacher_features, student_features)
                        results[name] += metric_value.item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})

        return {name: value / num_images for name, value in results.items()}

    @torch.no_grad()
    def test(self, test_loader, output_dir=None, show_image=False, img_prefix="img",
        skip_normal=False, skip_anomaly=False, num_max=-1):
        return self.test_feature_based(test_loader, output_dir, show_image, img_prefix,
            skip_normal, skip_anomaly, num_max)


#############################################################
# Helper functions
#############################################################

def compute_threshold(scores, labels, method="f1", percentile=95):
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()

    if method == "roc":
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
        optimal_idx = (tpr - fpr).argmax()
        return thresholds[optimal_idx]
    elif method == "f1":
        from sklearn.metrics import f1_score
        thresholds = torch.linspace(scores.min(), scores.max(), 100)
        best_f1, best_threshold = 0.0, 0.5
        for thr in thresholds:
            preds = (scores >= thr).float()
            f1 = f1_score(labels_np, preds.numpy())
            if f1 > best_f1:
                best_f1, best_threshold = f1, thr.item()
        return best_threshold
    else:
        normal_mask = labels_np == 0
        if normal_mask.sum() == 0:
            return 0.5
        normal_scores = scores_np[normal_mask]
        threshold = float(torch.quantile(torch.tensor(normal_scores), percentile / 100.0))
        return threshold


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def check_shape(img):
    """Ensure input is numpy array with shape (H, W) or (H, W, 3)."""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:      # (1,H,W)
        img = img[0]
    elif img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))       # (H,W,3)
    return img


