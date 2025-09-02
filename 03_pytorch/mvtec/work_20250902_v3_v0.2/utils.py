from metrics.metrics_base import AUROCMetric, AUPRMetric, OptimalThresholdMetric
from metrics.metrics_base import AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric


def count_labels(dataset):
    from torch.utils.data import Subset, ConcatDataset
    
    def extract_labels(ds):
        if isinstance(ds, Subset):
            original_labels = ds.dataset.labels
            return [original_labels[i] for i in ds.indices]
        elif isinstance(ds, ConcatDataset):
            all_labels = []
            for constituent in ds.datasets:
                all_labels.extend(extract_labels(constituent))
            return all_labels
        else:
            return ds.labels
    
    labels = extract_labels(dataset)
    anomaly_count = sum(labels)
    normal_count = len(labels) - anomaly_count
    return normal_count, anomaly_count


def show_data_info(data):
    print()
    print(f" > Dataset Type:      {data.data_dir}")
    print(f" > Categories:        {data.categories}")

    train = data.train_loader().dataset    
    normal, anomal = count_labels(train)
    print(f" > Train data:        {len(train)} (normal={normal}, anomaly={anomal})")

    valid = None if data.valid_loader() is None else data.valid_loader().dataset
    if valid is not None:
        normal, anomal = count_labels(valid)
        print(f" > Valid data:        {len(valid)} (normal={normal}, anomaly={anomal})")

    test = data.test_loader().dataset
    normal, anomal = count_labels(test)
    print(f" > Test data:         {len(test)} (normal={normal}, anomaly={anomal})")


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


def show_results(scores, labels):
    auroc = AUROCMetric()(labels, scores)
    aupr = AUPRMetric()(labels, scores)

    threshold = OptimalThresholdMetric(method="f1")(labels, scores)
    predictions = (scores >= threshold).float()

    accuracy = AccuracyMetric()(labels, predictions)
    precision = PrecisionMetric()(labels, predictions)
    recall = RecallMetric()(labels, predictions)
    f1 = F1Metric()(labels, predictions)

    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f" > AUROC:             {auroc:.4f}")
    print(f" > AUPR:              {aupr:.4f}")
    print(f" > Threshold:         {threshold:.3e}")
    print("-"*50)
    print(f" > Accuracy:          {accuracy:.4f}")
    print(f" > Precision:         {precision:.4f}")
    print(f" > Recall:            {recall:.4f}")
    print(f" > F1-Score:          {f1:.4f}")
    print()


if __name__ == "__main__":
    pass