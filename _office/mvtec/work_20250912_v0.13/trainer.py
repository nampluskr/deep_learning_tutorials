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

    def to_device(self, inputs):
        device_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                device_inputs[key] = value.to(self.device, non_blocking=True)
            else:
                device_inputs[key] = value
        return device_inputs

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
        self.model.train()
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
                inputs = self.to_device(inputs)

                predictions = self.model(inputs['image'])
                scores = predictions['pred_score']
                labels = inputs["label"]

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        scores_tensor = torch.cat(all_scores, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        return scores_tensor, labels_tensor


class AutoEncoderTrainer(BaseTrainer):

    def _run_epoch(self, data_loader, mode="train", desc=""):
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                batch = self.to_device(batch)

                if mode == "train":
                    self.optimizer.zero_grad()
                    reconstruction, *_ = self.model(batch["image"])
                    loss = self.loss_fn(reconstruction, batch["image"])
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        reconstruction, *_ = self.model(batch["image"])
                        loss = self.loss_fn(reconstruction, batch["image"])

                batch_results = {"loss": loss.item()}
                with torch.no_grad():
                    for metric_name, metric_fn in self.metrics.items():
                        metric_val = metric_fn(reconstruction, batch["image"])
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
                batch = self.to_device(batch)

                if mode == "train":
                    self.optimizer.zero_grad()
                    teacher_features, student_features = self.model(batch["image"])
                    loss = self.loss_fn(teacher_features, student_features)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        teacher_features, student_features = self.model(batch["image"])
                        loss = self.loss_fn(teacher_features, student_features)

                batch_results = {"loss": loss.item()}
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

    def _run_epoch(self, data_loader, mode="train", desc=""):
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metric_names}
        num_batches = 0

        with tqdm(data_loader, desc=desc, leave=False, ascii=True) as pbar:
            for batch in pbar:
                batch = self.to_device(batch)

                if mode == "train":
                    self.optimizer.zero_grad()
                    loss_st, loss_ae, loss_stae = self.model(batch["image"])
                    loss = loss_st + loss_ae + loss_stae
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        loss_st, loss_ae, loss_stae = self.model(batch["image"])
                        loss = loss_st + loss_ae + loss_stae

                batch_results = {"loss": loss.item()}
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
