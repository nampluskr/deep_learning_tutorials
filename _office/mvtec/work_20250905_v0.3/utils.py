import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score,)
import pandas as pd


def show_data_info(data):
    print()
    print(f" > Dataset Type:      {data.data_dir}")
    print(f" > Categories:        {data.categories}")
    print(f" > Train data:        {len(data.train_loader().dataset)}")

    valid_loader = data.valid_loader()
    if valid_loader is not None:
        print(f" > Valid data:        {len(valid_loader.dataset)}")
    else:
        print(f" > Valid data:        None (no validation split)")

    print(f" > Test data:         {len(data.test_loader().dataset)}")


def show_model_info(model):
    print()
    # print(f" > Modeler Type:      {type(modeler).__name__}")
    # print(f" > Model Type:        {type(modeler.model).__name__}")
    print(f" > Total params.:     "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f" > Trainable params.: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # print(f" > Learning Type:     {modeler.learning_type}")
    # print(f" > Loss Function:     {type(modeler.loss_fn).__name__}")
    # print(f" > Metrics:           {list(modeler.metrics.keys())}")
    # print(f" > Device:            {modeler.device}")


def show_modeler_info(modeler):
    print()
    print(f" > Modeler Type:      {type(modeler).__name__}")
    print(f" > Model Type:        {type(modeler.model).__name__}")
    print(f" > Total params.:     "
          f"{sum(p.numel() for p in modeler.model.parameters()):,}")
    print(f" > Trainable params.: "
          f"{sum(p.numel() for p in modeler.model.parameters() if p.requires_grad):,}")
    print(f" > Learning Type:     {modeler.learning_type}")
    print(f" > Loss Function:     {type(modeler.loss_fn).__name__}")
    print(f" > Metrics:           {list(modeler.metrics.keys())}")
    print(f" > Device:            {modeler.device}")


def show_trainer_info(trainer):
    print()
    print(f" > Trainer Type:      {trainer.trainer_type}")
    print(f" > Optimizer:         {type(trainer.optimizer).__name__}")
    print(f" > Learning Rate:     {trainer.optimizer.param_groups[0]['lr']}")

    if trainer.scheduler is not None:
        print(f" > Scheduler:         {type(trainer.scheduler).__name__}")
    else:
        print(f" > Scheduler:         None")

    if trainer.stopper is not None:
        print(f" > Stopper:           {type(trainer.stopper).__name__}")
        if hasattr(trainer.stopper, 'patience'):
            print(f" > Patience:          {trainer.stopper.patience}")
        if hasattr(trainer.stopper, 'min_delta'):
            print(f" > Min Delta:         {trainer.stopper.min_delta}")
        if hasattr(trainer.stopper, 'max_epoch'):
            print(f" > Max Epochs:        {trainer.stopper.max_epoch}")
    else:
        print(f" > Stopper:           None")

    if trainer.logger is not None:
        print(f" > Logger:            {type(trainer.logger).__name__}")
    else:
        print(f" > Logger:            None")


def show_results_new(scores, labels):
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()

    results = {}
    results['auroc'] = roc_auc_score(labels, scores)
    results['aupr'] = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    results["threshold"] = best_threshold

    predictions = (scores >= best_threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    results['tn'] = tn
    results['fp'] = fp
    results['fn'] = fn
    results['tp'] = tp

    results["accuracy"] = accuracy_score(labels, predictions)
    results["precision"] = precision_score(labels, predictions)
    results["recall"] = recall_score(labels, predictions)
    results["f1"] = f1_score(labels, predictions)

    result_lines = [
        "-" * 60,
        f"EXPERIMENT RESULTS (NEW)",
        "-" * 60,
        f" > AUROC:             {results['auroc']:.4f}",
        f" > AUPR:              {results['aupr']:.4f}",
        f" > Threshold:         {results['threshold']:.3e}",
         "-" * 60,
        f" > Accuracy:          {results['accuracy']:.4f}",
        f" > Precision:         {results['precision']:.4f}",
        f" > Recall:            {results['recall']:.4f}",
        f" > F1-Score:          {results['f1']:.4f}",
        "-" * 60,
        f"                   Predicted",
        f"   Actual    Normal  Anomaly",
        f"   Normal    {results['tn']:6d}  {results['fp']:7d}",
        f"   Anomaly   {results['fn']:6d}  {results['tp']:7d}",
    ]
    for line in result_lines:
        print(line)


def show_results_old(scores, labels):
    results = {}
    results['auroc'] = roc_auc_score(labels, scores)
    results['aupr'] = average_precision_score(labels, scores)

    threshold = get_optimal_threshold(scores, labels, method="f1")
    predictions = (scores >= threshold).float()
    results["threshold"] = threshold

    results["accuracy"] = accuracy_score(labels, predictions)
    results["precision"] = precision_score(labels, predictions)
    results["recall"] = recall_score(labels, predictions)
    results["f1"] = f1_score(labels, predictions)

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    results['tn'] = tn
    results['fp'] = fp
    results['fn'] = fn
    results['tp'] = tp

    result_lines = [
        "-" * 60,
        f"EXPERIMENT RESULTS (OLD)",
        "-" * 60,
        f" > AUROC:             {results['auroc']:.4f}",
        f" > AUPR:              {results['aupr']:.4f}",
        f" > Threshold:         {results['threshold']:.3e}",
         "-" * 60,
        f" > Accuracy:          {results['accuracy']:.4f}",
        f" > Precision:         {results['precision']:.4f}",
        f" > Recall:            {results['recall']:.4f}",
        f" > F1-Score:          {results['f1']:.4f}",
        "-" * 60,
        f"                   Predicted",
        f"   Actual    Normal  Anomaly",
        f"   Normal    {results['tn']:6d}  {results['fp']:7d}",
        f"   Anomaly   {results['fn']:6d}  {results['tp']:7d}",
    ]
    for line in result_lines:
        print(line)

import torch

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve


def get_thresholds(scores, labels):
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()
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
    # 후보 threshold 를 CPU 텐서에 저장해 반복 시 GPU‑CPU 전송을 최소화
    cand_thr = torch.linspace(scores.min(), scores.max(), steps=100).cpu()
    best_f1, best_thr = 0.0, None
    for thr_val in cand_thr:
        preds = (scores >= thr_val).float().cpu().numpy()
        f1 = f1_score(labels_np, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr_val.item()
    thresholds["f1"] = best_thr if best_thr is not None else float(cand_thr.mean().item())

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
    mu = scores.mean().item()
    sigma = scores.std().item()
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

def evaluate_thresholds(scores, label, thresholds):
    try:
        import torch
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)
        if isinstance(label, torch.Tensor):
            label_np = label.detach().cpu().numpy()
        else:
            label_np = np.asarray(label)
    except Exception:
        scores_np = np.asarray(scores)
        label_np  = np.asarray(label)

    auroc = roc_auc_score(label_np, scores_np)
    aupr  = average_precision_score(label_np, scores_np)
    normal_scores  = scores_np[label_np == 0]
    anomaly_scores = scores_np[label_np == 1]

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
        tn, fp, fn, tp = confusion_matrix(label_np, pred, labels=[0, 1]).ravel()

        rows.append({
            "method":    thr,
            "threshold": thr_val,
            "auroc":     auroc,
            "aupr":      aupr,
            "cohens_d":  cohens_d,
            "accuracy":  accuracy_score(label_np, pred),
            "precision": precision_score(label_np, pred, zero_division=0),
            "recall":    recall_score(label_np, pred, zero_division=0),
            "f1":        f1_score(label_np, pred, zero_division=0),
            "mcc":       matthews_corrcoef(label_np, pred),
            "kappa":     cohen_kappa_score(label_np, pred),
            "tn":        tn,
            "fp":        fp,
            "fn":        fn,
            "tp":        tp,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    pass
