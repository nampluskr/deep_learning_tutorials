import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from time import time


class BaseTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metrics = metrics or {}

    def train(self, train_loader):
        self.model.train()
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0
        with tqdm(train_loader, desc="Train", leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                batch_size = images.size(0)
                num_images += batch_size

                self.optimizer.zero_grad()
                recon, *_ = self.model(images)
                loss = self.loss_fn(recon, images)
                loss.backward()
                self.optimizer.step()

                results["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        results[name] += metric_fn(recon, images).item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})

        return {name: value / num_images for name, value in results.items()}

    @torch.no_grad()
    def validate(self, valid_loader):
        self.model.eval()
        results = {name: 0.0 for name in ["loss"] + list(self.metrics)}
        num_images = 0
        with tqdm(valid_loader, desc="Valid", leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                batch_size = images.size(0)
                num_images += batch_size

                recon, *_ = self.model(images)
                loss = self.loss_fn(recon, images)

                results["loss"] += loss.item() * batch_size
                for name, metric_fn in self.metrics.items():
                    results[name] += metric_fn(recon, images).item() * batch_size

                pbar.set_postfix({**{n: f"{v/num_images:.3f}" for n, v in results.items()}})

        return {name: value / num_images for name, value in results.items()}

    def fit(self, train_loader, num_epochs, valid_loader=None, output_dir="./output"):
        os.makedirs(output_dir, exist_ok=True)
        history = {name: [] for name in ['loss'] + list(self.metrics)}
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metrics)})

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            train_results = self.train(train_loader)
            train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

            for name, value in train_results.items():
                history[name].append(value)

            if valid_loader is not None:
                valid_results = self.validate(valid_loader)
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

                for name, value in valid_results.items():
                    history[f"val_{name}"].append(value)

                elapsed_time = time() - start_time
                print(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                print(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

        self.save_model(epoch, output_dir)
        return history

    # @torch.no_grad()
    # def test(self, test_loader, output_dir, threshold=0.5):
    #     output_dir = os.path.abspath(output_dir)
    #     os.makedirs(output_dir, exist_ok=True)

    #     self.model.eval()
    #     num_images = 0
    #     for batch in test_loader:
    #         images = batch['image'].to(self.device)
    #         predictions = self.model.predict(images)
    #         anomaly_maps = predictions["anomaly_map"]
    #         pred_scores = predictions["pred_score"]

    #         for i in range(images.size(0)):
    #             map = anomaly_maps[i].cpu()
    #             mask = (map > threshold).float()
    #             score = float(pred_scores[i].cpu().item())

    #             img_name = os.path.splitext(os.path.basename(names[i]))[0]
    #             map_path = os.path.join(output_dir, f"{img_name}_map.png")
    #             mask_path = os.path.join(output_dir, f"{img_name}_mask.png")
    #             score_path = os.path.join(output_dir, f"{img_name}_score.json")
    #             map_norm = (map - map.min()) / (map.max() - map.min() + 1e-8)

    #             to_pil_image(map_norm).save(map_path)
    #             to_pil_image(mask).save(mask_path)

    #             with open(score_path, "w") as f:
    #                 json.dump({"score": score}, f)
    #             num_images += 1
    #     return {"output_dir": output_dir, "num_images": num_images}

    @torch.no_grad()
    def evaluate(self, test_loader, pixel_level=True, image_level=True):
        self.model.eval()
        img_labels, img_scores = [], []
        pix_labels, pix_scores = [], []

        with tqdm(test_loader, desc="Evaluation", leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].cpu().numpy()
                masks = batch["mask"].cpu().numpy() if "mask" in batch else None

                predictions = self.model.predict(images)
                anomaly_maps = predictions["anomaly_map"]
                pred_scores = predictions["pred_score"]

                if image_level:
                    img_labels.extend(labels.tolist())
                    img_scores.extend(pred_scores.view(-1).cpu().tolist())
                    # results["img_label"] = img_labels
                    # results["img_score"] = img_scores

                if pixel_level and masks is not None:
                    maps = anomaly_maps.squeeze(1) if anomaly_maps.dim() == 4 else anomaly_maps
                    pix_labels.extend(masks.ravel().tolist())
                    pix_scores.extend(maps.view(-1).cpu().numpy().tolist())
                    # results["pix_label"] = img_labels
                    # results["pix_score"] = img_scores

        results = {}
        if image_level:
            if len(set(img_labels)) > 1:
                results["img_auroc"] = roc_auc_score(img_labels, img_scores)
                results["img_aupr"] = average_precision_score(img_labels, img_scores)
            else:
                results["img_auroc"] = float("nan")
                results["img_aupr"] = float("nan")

        if pixel_level and len(pix_labels) > 0:
            if len(set(pix_labels)) > 1:
                results["pix_auroc"] = roc_auc_score(pix_labels, pix_scores)
                results["pix_aupr"] = average_precision_score(pix_labels, pix_scores)
            else:
                results["pix_auroc"] = float("nan")
                results["pix_aupr"] = float("nan")

        return results

    def save_model(self, epoch, output_dir):
        weights_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch
        }, weights_path)
        return weights_path

    def load_model(self, weights_path):
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state"])
        if self.optimizer is not None and "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])