import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
import os
from time import time
from copy import deepcopy


class BaseTrainer(ABC):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, scheduler=None, stopper=None, logger=None):
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics or {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.logger = logger
        self.metric_names = list(self.metrics.keys())
        self.device = next(model.parameters()).device

    def log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        print(message)

    def _update_learning_rate(self, epoch, train_results, valid_results):
        if self.scheduler is not None:
            last_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                metric = valid_results.get('loss', train_results['loss'])
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            if abs(current_lr - last_lr) > 1e-12:
                self.log(f" > learning rate changed: {last_lr:.3e} => {current_lr:.3e}")

    def _check_stopping_condition(self, epoch, train_results, valid_results):
        if self.stopper is not None:
            current_loss = valid_results.get('loss', train_results['loss'])

            if hasattr(self.stopper, 'update_metrics'):
                current_metrics = {**train_results}
                if valid_results:
                    current_metrics.update(valid_results)
                self.stopper.update_metrics(current_metrics)

            should_stop = self.stopper(current_loss, self.model)
            if should_stop:
                self.log(f"Training stopped by stopper at epoch {epoch}")
                return True
        return False

    @abstractmethod
    def _run_epoch(self, data_loader, mode="train", desc=""):
        pass

    def train(self, train_loader, epoch, num_epochs):
        self.model.train()
        desc = f"Train [{epoch}/{num_epochs}]"
        return self._run_epoch(train_loader, mode="train", desc=desc)

    @torch.no_grad()
    def validate(self, valid_loader, epoch, num_epochs):
        self.model.eval()
        desc = f"Validate [{epoch}/{num_epochs}]"
        return self._run_epoch(valid_loader, mode="valid", desc=desc)

    def fit(self, train_loader, num_epochs=None, valid_loader=None):
        history = {'loss': []}
        history.update({name: [] for name in self.metric_names})
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in ['loss'] + list(self.metric_names)})

        self.log("\n > Training started...")
        for epoch in range(1, num_epochs + 1):
            start_time = time()
            train_results = self.train(train_loader, epoch, num_epochs)
            train_info = ", ".join([f'{key}={value:.3f}' for key, value in train_results.items()])

            for key, value in train_results.items():
                if key in history:
                    history[key].append(value)

            valid_results = {}
            if valid_loader is not None:
                valid_results = self.validate(valid_loader, epoch, num_epochs)
                valid_info = ", ".join([f'{key}={value:.3f}' for key, value in valid_results.items()])

                for key, value in valid_results.items():
                    val_key = f"val_{key}"
                    if val_key in history:
                        history[val_key].append(value)

                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} | (val) {valid_info} ({elapsed_time:.1f}s)")
            else:
                elapsed_time = time() - start_time
                self.log(f" [{epoch:2d}/{num_epochs}] {train_info} ({elapsed_time:.1f}s)")

            self._update_learning_rate(epoch, train_results, valid_results)

            if self._check_stopping_condition(epoch, train_results, valid_results):
                break

        self.log(" > Training completed!")
        return history

    @torch.no_grad()
    def predict(self, test_loader):
        self.model.eval()
        all_scores, all_labels = [], []

        with tqdm(test_loader, desc="Predict", leave=False, ascii=True) as pbar:
            for inputs in pbar:
                images = inputs['image'].to(self.device)
                predictions = self.model.evaluate(images)
                scores = predictions['pred_score']
                labels = inputs["label"]

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        scores_tensor = torch.cat(all_scores, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        return scores_tensor, labels_tensor

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


class AutoEncoderTrainer(BaseTrainer):

    def _run_epoch(self, data_loader, mode="train", desc=""):
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)

                if mode == "train":
                    self.optimizer.zero_grad()
                    reconstruction, *_ = self.model(images)
                    loss = self.loss_fn(reconstruction, images)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        reconstruction, *_ = self.model(images)
                        loss = self.loss_fn(reconstruction, images)

                batch_results = {"loss": loss.item()}
                with torch.no_grad():
                    for metric_name, metric_fn in self.metrics.items():
                        metric_val = metric_fn(reconstruction, images)
                        batch_results[metric_name] = float(metric_val)

                total_loss += batch_results["loss"]
                for name in self.metric_names:
                    total_metrics[name] += batch_results.get(name, 0.0)

                num_batches += 1
                avg_loss = total_loss / num_batches
                avg_metrics = {n: total_metrics[n] / num_batches for n in self.metric_names}
                pbar.set_postfix({"loss": f"{avg_loss:.3f}", 
                    **{n: f"{v:.3f}" for n, v in avg_metrics.items()}})

        results = {"loss": total_loss / max(num_batches, 1)}
        results.update({n: total_metrics[n] / max(num_batches, 1) for n in self.metric_names})
        return results


class STFPMTrainer(BaseTrainer):

    def _run_epoch(self, data_loader, mode="train", desc=""):
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)

                if mode == "train":
                    self.optimizer.zero_grad()
                    teacher_features, student_features = self.model(images)
                    loss = self.loss_fn(teacher_features, student_features)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        teacher_features, student_features = self.model(images)
                        loss = self.loss_fn(teacher_features, student_features)

                batch_results = {"loss": loss.item()}
                
                # Calculate metrics only once
                with torch.no_grad():
                    for metric_name, metric_fn in self.metrics.items():
                        metric_val = metric_fn(teacher_features, student_features)
                        batch_results[metric_name] = float(metric_val)

                total_loss += batch_results["loss"]
                for name in self.metric_names:
                    total_metrics[name] += batch_results.get(name, 0.0)

                num_batches += 1
                avg_loss = total_loss / num_batches
                avg_metrics = {n: total_metrics[n] / num_batches for n in self.metric_names}
                pbar.set_postfix({"loss": f"{avg_loss:.3f}", 
                    **{n: f"{v:.3f}" for n, v in avg_metrics.items()}})

        results = {"loss": total_loss / max(num_batches, 1)}
        results.update({n: total_metrics[n] / max(num_batches, 1) for n in self.metric_names})
        return results


class EfficientADTrainer(BaseTrainer):
    """EfficientAD trainer with three-loss training strategy."""

    def _run_epoch(self, data_loader, mode="train", desc=""):
        total_loss = 0.0
        total_loss_st = 0.0
        total_loss_ae = 0.0
        total_loss_stae = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)

                if mode == "train":
                    self.optimizer.zero_grad()
                    # EfficientAD returns three separate losses in training mode
                    loss_st, loss_ae, loss_stae = self.model(images)
                    total_batch_loss = loss_st + loss_ae + loss_stae
                    total_batch_loss.backward()
                    self.optimizer.step()
                    
                    # Record individual losses for monitoring
                    batch_results = {
                        "loss": total_batch_loss.item(),
                        "loss_st": loss_st.item(),
                        "loss_ae": loss_ae.item(),
                        "loss_stae": loss_stae.item()
                    }
                else:
                    # Validation mode - use model's internal loss computation
                    with torch.no_grad():
                        student_output, distance_st = self.model.compute_student_teacher_distance(images)
                        loss_st, loss_ae, loss_stae = self.model.compute_losses(images, None, distance_st)
                        total_batch_loss = loss_st + loss_ae + loss_stae
                        
                        batch_results = {
                            "loss": total_batch_loss.item(),
                            "loss_st": loss_st.item(),
                            "loss_ae": loss_ae.item(),
                            "loss_stae": loss_stae.item()
                        }

                # Calculate metrics if available
                with torch.no_grad():
                    for metric_name, metric_fn in self.metrics.items():
                        # EfficientAD specific metric calculation
                        if hasattr(self.model, 'compute_student_teacher_distance'):
                            student_output, distance_st = self.model.compute_student_teacher_distance(images)
                            metric_val = metric_fn(student_output, distance_st)
                            batch_results[metric_name] = float(metric_val)
                        else:
                            batch_results[metric_name] = 0.0

                total_loss += batch_results["loss"]
                total_loss_st += batch_results["loss_st"]
                total_loss_ae += batch_results["loss_ae"]
                total_loss_stae += batch_results["loss_stae"]
                for name in self.metric_names:
                    total_metrics[name] += batch_results.get(name, 0.0)

                num_batches += 1
                avg_loss = total_loss / num_batches
                avg_loss_st = total_loss_st / num_batches
                avg_loss_ae = total_loss_ae / num_batches
                avg_loss_stae = total_loss_stae / num_batches
                avg_metrics = {n: total_metrics[n] / num_batches for n in self.metric_names}
                
                pbar.set_postfix({
                    "loss": f"{avg_loss:.3f}",
                    "st": f"{avg_loss_st:.3f}",
                    "ae": f"{avg_loss_ae:.3f}",
                    "stae": f"{avg_loss_stae:.3f}",
                    **{n: f"{v:.3f}" for n, v in avg_metrics.items()}
                })

        results = {
            "loss": total_loss / max(num_batches, 1),
            "loss_st": total_loss_st / max(num_batches, 1),
            "loss_ae": total_loss_ae / max(num_batches, 1),
            "loss_stae": total_loss_stae / max(num_batches, 1),
        }
        results.update({n: total_metrics[n] / max(num_batches, 1) for n in self.metric_names})
        return results

    def fit(self, train_loader, num_epochs=None, valid_loader=None):
        """Override fit method to handle normalization parameter updates."""
        # Call parent fit method
        history = super().fit(train_loader, num_epochs, valid_loader)
        
        # Update normalization parameters after training
        self._update_normalization_parameters(train_loader)
        
        return history

    def _update_normalization_parameters(self, train_loader):
        """Update model normalization parameters using training data."""
        self.log("Computing normalization parameters...")
        self.model.eval()
        teacher_outputs = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing normalization params", leave=False):
                images = batch["image"].to(self.device)
                teacher_output = self.model.teacher(images)
                teacher_outputs.append(teacher_output)
        
        # Concatenate all teacher outputs
        all_teacher_outputs = torch.cat(teacher_outputs, dim=0)
        
        # Compute mean and std
        mean = torch.mean(all_teacher_outputs, dim=(0, 2, 3), keepdim=True)
        std = torch.std(all_teacher_outputs, dim=(0, 2, 3), keepdim=True)
        
        # Update model parameters
        self.model.mean_std["mean"].data = mean
        self.model.mean_std["std"].data = std + 1e-8  # Add small epsilon for stability
        
        self.log(f" > Updated normalization - Mean: {mean.mean().item():.4f}, Std: {std.mean().item():.4f}")

    @torch.no_grad()
    def update_quantiles(self, valid_loader):
        """Update quantiles for anomaly map normalization."""
        self.log("Computing quantiles for anomaly map normalization...")
        self.model.eval()
        maps_st = []
        maps_stae = []
        
        with tqdm(valid_loader, desc="Computing quantiles", leave=False) as pbar:
            for batch in pbar:
                images = batch["image"].to(self.device)
                student_output, distance_st = self.model.compute_student_teacher_distance(images)
                map_st, map_stae = self.model.compute_maps(images, student_output, distance_st, normalize=False)
                
                maps_st.append(map_st.flatten())
                maps_stae.append(map_stae.flatten())
        
        # Concatenate all maps
        all_maps_st = torch.cat(maps_st)
        all_maps_stae = torch.cat(maps_stae)
        
        # Compute quantiles (0.9 and 0.995)
        qa_st = torch.quantile(all_maps_st, 0.9)
        qb_st = torch.quantile(all_maps_st, 0.995)
        qa_ae = torch.quantile(all_maps_stae, 0.9)
        qb_ae = torch.quantile(all_maps_stae, 0.995)
        
        # Update model quantiles
        self.model.quantiles["qa_st"].data = qa_st
        self.model.quantiles["qb_st"].data = qb_st
        self.model.quantiles["qa_ae"].data = qa_ae
        self.model.quantiles["qb_ae"].data = qb_ae
        
        self.log(f" > Updated quantiles - ST: [{qa_st:.4f}, {qb_st:.4f}], AE: [{qa_ae:.4f}, {qb_ae:.4f}]")
