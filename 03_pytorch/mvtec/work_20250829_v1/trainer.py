# trainer.py
# Final Trainer: works with any Modeler (FastFlow, PaDiM, PatchCore, STFPM, AE, VAE)

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from metrics import (
    PSNR, SSIM, DeltaE2000,
    FFTPhaseDifference, WaveletDifference,
    LPIPSMetric
)


class Trainer:
    """Generic Trainer for Anomaly Detection (model-agnostic via Modeler API)"""

    def __init__(self, modeler, optimizer=None, scheduler=None,
                 logger=None, device="cuda", lpips_weights=None):
        self.modeler = modeler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device

        # Reconstruction-based metrics
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        self.deltaE = DeltaE2000().to(device)
        self.fft = FFTPhaseDifference().to(device)
        self.wavelet = WaveletDifference().to(device)
        self.lpips = LPIPSMetric(net="alex", weights_path=lpips_weights, device=device) \
                     if lpips_weights else None

    # -------------------------
    # Training Loop
    # -------------------------
    def fit(self, train_loader, valid_loader=None, num_epochs=10):
        for epoch in range(num_epochs):
            # ---- Training ----
            train_losses = []
            for batch in train_loader:
                if self.optimizer is not None:
                    self.optimizer.zero_grad()

                results = self.modeler.train_step(batch)   # model-specific
                loss = results["loss"]

                if loss is not None and loss.requires_grad:
                    loss.backward()
                    if self.optimizer is not None:
                        self.optimizer.step()
                train_losses.append(loss.item() if loss is not None else 0.0)

            # ---- Validation ----
            val_losses = []
            if valid_loader is not None:
                for batch in valid_loader:
                    results = self.modeler.validate_step(batch)
                    loss = results["val_loss"]
                    val_losses.append(loss.item() if loss is not None else 0.0)

            # ---- Scheduler Step ----
            if self.scheduler is not None:
                if val_losses:
                    self.scheduler.step(np.mean(val_losses))
                else:
                    self.scheduler.step()

            # ---- Logging ----
            log_msg = f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {np.mean(train_losses):.4f}"
            if val_losses:
                log_msg += f" | Val Loss: {np.mean(val_losses):.4f}"
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)

    # -------------------------
    # Evaluation
    # -------------------------
    @torch.no_grad()
    def evaluate(self, dataloader, threshold: float = 0.5):
        self.modeler.model.eval()
        all_scores, all_labels = [], []
        pixel_scores, pixel_labels = [], []

        psnr_vals, ssim_vals, deltaE_vals, fft_vals, wav_vals, lpips_vals = [], [], [], [], [], []

        for batch in dataloader:
            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(self.device)

            outputs = self.modeler.predict_step({"image": images})

            anomaly_map = outputs.get("anomaly_map")
            score = outputs.get("score")
            recon = outputs.get("reconstruction")

            # Image-level score
            if score is not None:
                all_scores.append(score.detach().cpu())
                all_labels.append(labels.detach().cpu())

            # Pixel-level score (mask 존재 시)
            if anomaly_map is not None and masks is not None:
                anomaly_map_np = anomaly_map.detach().cpu().numpy().reshape(-1)
                mask_np = masks.detach().cpu().numpy().reshape(-1)
                pixel_scores.append(anomaly_map_np)
                pixel_labels.append(mask_np)

            # Reconstruction quality metrics
            if recon is not None:
                psnr_vals.append(self.psnr(images, recon))
                ssim_vals.append(self.ssim(images, recon))

                x_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
                xh_np = recon.detach().cpu().numpy().transpose(0, 2, 3, 1)
                for xi, xhi in zip(x_np, xh_np):
                    deltaE_vals.append(self.deltaE(xi, xhi))
                    fft_vals.append(self.fft(xi, xhi))
                    wav_vals.append(self.wavelet(xi, xhi))
                if self.lpips:
                    lpips_vals.append(self.lpips(images, recon))

        # -------------------------
        # Aggregate results
        # -------------------------
        results = {}

        # Image-level AUROC
        if all_scores:
            all_scores = torch.cat(all_scores).numpy()
            all_labels = torch.cat(all_labels).numpy()
            results["image_auc"] = roc_auc_score(all_labels, all_scores)

        # Pixel-level AUROC, IoU, Dice
        if pixel_scores and pixel_labels:
            pixel_scores = np.concatenate(pixel_scores)
            pixel_labels = np.concatenate(pixel_labels)

            results["pixel_auc"] = roc_auc_score(pixel_labels, pixel_scores)
            pred_mask = (pixel_scores > threshold).astype(np.uint8)
            gt_mask = pixel_labels.astype(np.uint8)

            TP = np.sum((pred_mask == 1) & (gt_mask == 1))
            FP = np.sum((pred_mask == 1) & (gt_mask == 0))
            FN = np.sum((pred_mask == 0) & (gt_mask == 1))

            results["pixel_iou"] = TP / (TP + FP + FN + 1e-8)
            results["pixel_dice"] = 2 * TP / (2 * TP + FP + FN + 1e-8)

        # Reconstruction metrics
        if psnr_vals:
            results.update({
                "psnr": float(np.mean(psnr_vals)),
                "ssim": float(np.mean(ssim_vals)),
                "deltaE2000": float(np.mean(deltaE_vals)),
                "fft_phase": float(np.mean(fft_vals)),
                "wavelet": float(np.mean(wav_vals)),
            })
            if self.lpips and lpips_vals:
                results["lpips"] = float(np.mean(lpips_vals))

        return results
