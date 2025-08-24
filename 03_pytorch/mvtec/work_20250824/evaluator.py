import torch
from tqdm import tqdm

class Evaluator:
    """Evaluator wrapper for anomaly detection models"""
    def __init__(self, modeler):
        self.modeler = modeler

    @torch.no_grad()
    def predict(self, test_loader):
        all_scores, all_labels = [], []

        for inputs in tqdm(test_loader, desc="Evaluate"):
            scores = self.modeler.predict_step(inputs)
            labels = inputs["label"]

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

        return {
            "score": torch.cat(all_scores, dim=0),
            "label": torch.cat(all_labels, dim=0)
        }
