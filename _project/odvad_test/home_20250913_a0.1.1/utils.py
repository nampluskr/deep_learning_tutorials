import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, confusion_matrix,
)

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

def get_optimal_threshold(scores, labels, method="f1", percentile=95):
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()

    if method == "roc":
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
        optimal_idx = (tpr - fpr).argmax()
        return thresholds[optimal_idx]

    elif method == "f1":
        thresholds = torch.linspace(scores.min(), scores.max(), 100)
        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds:
            preds = (scores >= threshold).float()
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold.item()

        return best_threshold

    elif method == "percentile":
        normal_mask = labels_np == 0
        if normal_mask.sum() == 0:
            return 0.5
        normal_scores = scores_np[normal_mask]
        threshold = float(torch.quantile(torch.tensor(normal_scores), self.percentile / 100.0))
        return threshold

    else:
        return 0.5

if __name__ == "__main__":
    pass