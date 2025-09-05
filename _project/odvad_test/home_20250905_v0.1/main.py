import os
import random
import numpy as np
import torch

# ===================================================================
# Configs
# ===================================================================

from types import SimpleNamespace

BASE_CONFIGS = SimpleNamespace(
    backbone_dir="/mnt/d/github/deep_learning_tutorials/_project/odvad_test/backbones",
    output_dir="./experiments",
    num_epochs=10,
    seed=42,
    verbose=True,
    show_dataloader=True,
    show_modeler=True,
    show_trainer=True,
    show_memory=True,
)

DATA_CONFIGS = {
    "mvtec": SimpleNamespace(
        dataloader_type="mvtec",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/mvtec",
            categories=["bottle", "cable", "capsule"],
            train_batch_size=4, test_batch_size=2,
            test_ratio=0.15, valid_ratio=0.15,
            num_workers=4, pin_memory=True, persistent_workers=True,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
}

MODEL_CONFIGS = {
    "vanilla_ae": SimpleNamespace(
        modeler_type="ae",
        trainer_type="reconstruction",

        model_type="vanilla_ae",
        model_params=dict(),
        loss_type="ae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "unet_ae": SimpleNamespace(
        modeler_type="ae",
        trainer_type="reconstruction",

        model_type="unet_ae",
        model_params=dict(),
        loss_type="ae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "vanilla_vae": SimpleNamespace(
        modeler_type="vae",
        trainer_type="reconstruction",

        model_type="vanilla_vae",
        model_params=dict(),
        loss_type="vae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "unet_vae": SimpleNamespace(
        modeler_type="vae",
        trainer_type="reconstruction",

        model_type="unet_vae",
        model_params=dict(),
        loss_type="vae",
        loss_params=dict(),
        metric_list=[("psnr", dict())],   # (metric_type, matric_params)
    ),
    "stfpm": SimpleNamespace(
        modeler_type="stfpm",
        trainer_type="reconstruction",

        model_type="stfpm",
        model_params=dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"]),
        loss_type="stfpm",
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],   # (metric_type, matric_params)
    ),
    "fastflow": SimpleNamespace(
        modeler_type="fastflow",
        trainer_type="reconstruction",
        
        model_type="fastflow",
        model_params=dict(
            backbone="resnet18", 
            flow_steps=8, 
            conv3x3_only=False, 
            hidden_ratio=1.0
        ),
        loss_type="fastflow",
        loss_params=dict(),
        metric_list=[("likelihood", dict())],
    ),
    "draem": SimpleNamespace(
        modeler_type="draem",
        trainer_type="reconstruction",
        model_type="draem",
        model_params=dict(sspcab=False),
        loss_type="draem",
        loss_params=dict(),
        metric_list=[("auroc", dict()), ("aupr", dict())],
    ),
}

TRAIN_CONFIGS = {
    "reconstruction": SimpleNamespace(
        trainer_type="reconstruction",
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-5, weight_decay=1e-5),
        scheduler_type="plateau",
        scheduler_params=dict(),
        stopper_type="early",
        stopper_params=dict(patience=5, min_delta=1e-4),
    ),
}


def build_config(data_type, model_type):
    config = SimpleNamespace()
    config = merge_configs(config, BASE_CONFIGS)
    config = merge_configs(config, DATA_CONFIGS[data_type])

    model_config = MODEL_CONFIGS[model_type]
    config = merge_configs(config, model_config)

    trainer_type = model_config.trainer_type
    config = merge_configs(config, TRAIN_CONFIGS[trainer_type])

    if model_type in ["patchcore", "padim"]:    # memory-based models
        config.num_epochs = 1
        config.dataloader_params["train_batch_size"] = 4
        config.dataloader_params["test_batch_size"] = 2
        config.dataloader_params["valid_ratio"] = 0.0
        config.dataloader_params["train_shuffle"] = False
        config.dataloader_params["test_shuffle"] = False
        config.dataloader_params["train_drop_last"] = False
        config.dataloader_params["test_drop_last"] = False
    return config


def merge_configs(destination, source):
    for key, value in source.__dict__.items():
        if key.endswith("_params") and hasattr(destination, key):
            existing_params = getattr(destination, key)
            if isinstance(existing_params, dict) and isinstance(value, dict):
                existing_params.update(value)
            else:
                setattr(destination, key, value)
        else:
            setattr(destination, key, value)
    return destination


# ===================================================================
# Dataloaders
# ===================================================================

from dataloader import TrainTransform, TestTransform
from dataloader import MVTecDataloader

DATALOADER_REGISTRY = {
    "mvtec": MVTecDataloader,
}

TRANSFORM_REGISTRY = {
    "train": TrainTransform,
    "test": TestTransform,
}

# ===================================================================
# Models / Losses
# ===================================================================

import torch.nn as nn

from model_ae import VanillaAE, UNetAE, AELoss
from model_vae import VanillaVAE, UNetVAE, VAELoss
from model_stfpm import STFPMModel, STFPMLoss
from model_fastflow import FastFlowModel, FastFlowLoss
from model_draem import DRAEMModel, DRAEMLoss

MODEL_REGISTRY = {
    "vanilla_ae": VanillaAE,
    "unet_ae": UNetAE,
    "vanilla_vae": VanillaVAE,
    "unet_vae": UNetVAE,
    "stfpm": STFPMModel,
    "fastflow": FastFlowModel,
    "draem": DRAEMModel,
}

LOSS_REGISTRY = {
    "ae": AELoss,
    "vae": VAELoss,
    "stfpm": STFPMLoss,
    "fastflow": FastFlowLoss,
    "draem": DRAEMLoss,
}

# ===================================================================
# Metrics
# ===================================================================

from metrics import AUROCMetric, AUPRMetric, AccuracyMetric, PrecisionMetric
from metrics import RecallMetric, F1Metric, OptimalThresholdMetric
from metrics import PSNRMetric, FeatureSimilarityMetric
from metrics import LikelihoodMetric

METRIC_REGISTRY = {
    "auroc": AUROCMetric,
    "aupr": AUPRMetric,
    "acc": AccuracyMetric,
    "prec": PrecisionMetric,
    "recall": RecallMetric,
    "f1": F1Metric,
    "thershold": OptimalThresholdMetric,
    "psnr": PSNRMetric,
    "feature_sim": FeatureSimilarityMetric,
    "likelihood": LikelihoodMetric
}


# ===================================================================
# Modelers
# ===================================================================

from modeler import AEModeler, VAEModeler, STFPMModeler, FastFlowModeler, DRAEMModeler

MODELER_REGISTRY = {
    "ae": AEModeler,
    "vae": VAEModeler,
    "stfpm": STFPMModeler,
    "fastflow": FastFlowModeler,
    "draem": DRAEMModeler,
}


# ===================================================================
# Trainers / Optimizers / Schedulers / Stoppers
# ===================================================================

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from trainer import ReconstructionTrainer, EarlyStopper

TRAINER_REGISTRY = {
    "reconstruction": ReconstructionTrainer,
}

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}

SCHEDULER_REGISTRY = {
    "plateau": lr_scheduler.ReduceLROnPlateau,
}

STOPPER_REGISTRY = {
    "early": EarlyStopper,
}


# ===================================================================
# Factory Functions
# ===================================================================

def build_transform(transform_type, **transform_params):
    if transform_type is None or transform_type.lower() == "none":
        return None

    transform_type = transform_type.lower()
    if transform_type not in TRANSFORM_REGISTRY:
        available_transforms = list(TRANSFORM_REGISTRY.keys())
        raise ValueError(f"Unknown transform: {transform_type}. Available transforms: {available_transforms}")

    transform = TRANSFORM_REGISTRY.get(transform_type)
    params = {}
    params.update(transform_params)
    return transform(**params)


def build_dataloader(dataloader_type, **dataloader_params):
    dataloader_type = dataloader_type.lower()
    if dataloader_type not in DATALOADER_REGISTRY:
        available_dataloaders = list(DATALOADER_REGISTRY.keys())
        raise ValueError(f"Unknown dataloader: {dataloader_type}. Available dataloaders: {available_dataloaders}")

    dataloader = DATALOADER_REGISTRY.get(dataloader_type)
    params = {}
    params.update(dataloader_params)
    return dataloader(**params)


def build_model(model_type, **model_params):
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_type}. Available models: {available_models}")

    model = MODEL_REGISTRY.get(model_type)
    params = {}
    params.update(model_params)
    return model(**params)


def build_loss_fn(loss_type, **loss_params):
    if loss_type is None or loss_type.lower() == "none":
        return None

    loss_type = loss_type.lower()
    if loss_type not in LOSS_REGISTRY:
        available_losses = list(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss: {loss_type}. Available losses: {available_losses}")

    loss = LOSS_REGISTRY.get(loss_type)
    params = {}
    params.update(loss_params)
    return loss(**params)


def build_metric(metric_type, **metric_params):
    metric_type = metric_type.lower()
    if metric_type not in METRIC_REGISTRY:
        available_metrics = list(METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric: {metric_type}. Available metrics: {available_metrics}")

    metric = METRIC_REGISTRY.get(metric_type)
    params = {}
    params.update(metric_params)
    return metric(**params)


def build_metrics(metric_list):
    metrics = {}
    for metric_type, metric_params in metric_list:
        metrics[metric_type] = build_metric(metric_type, **metric_params)
    return metrics


def build_modeler(modeler_type, **modeler_params):
    modeler_type = modeler_type.lower()
    if modeler_type not in MODELER_REGISTRY:
        available_modelers = list(MODELER_REGISTRY.keys())
        raise ValueError(f"Unknown modeler: {modeler_type}. Available modelers: {available_modelers}")

    modeler = MODELER_REGISTRY.get(modeler_type)
    model = modeler_params.pop('model', None)
    loss_fn = modeler_params.pop('loss_fn', None)
    metrics = modeler_params.pop('metrics', None)
    return modeler(model, loss_fn=loss_fn, metrics=metrics, **modeler_params)


def build_trainer(trainer_type, **trainer_params):
    trainer_type = trainer_type.lower()
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer: {trainer_type}. Available trainers: {available_trainers}")

    trainer = TRAINER_REGISTRY.get(trainer_type)
    modeler = trainer_params.pop('modeler', None)
    optimizer = trainer_params.pop('optimizer', None)

    if modeler is None:
        raise ValueError("Modeler is required for trainer")
    if optimizer is None:
        raise ValueError("Optimizer is required for trainer")

    scheduler = trainer_params.pop('scheduler', None)
    stopper = trainer_params.pop('stopper', None)
    params = {}
    params.update(trainer_params)
    return trainer(modeler, optimizer, scheduler=scheduler, stopper=stopper, **params)


def build_optimizer(optimizer_type, **optimizer_params):
    if optimizer_type is None or optimizer_type.lower() == "none":
        return None

    optimizer_type = optimizer_type.lower()
    if optimizer_type not in OPTIMIZER_REGISTRY:
        available_optimizers = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Available optimizers: {available_optimizers}")

    optimizer = OPTIMIZER_REGISTRY.get(optimizer_type)
    model = optimizer_params.pop('model', None)
    if model is None:
        raise ValueError("Model is required for optimizer")

    params = {"lr": 1e-3}
    params.update(optimizer_params)
    return optimizer(model.parameters(), **params)


def build_scheduler(scheduler_type, **scheduler_params):
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None

    scheduler_type = scheduler_type.lower()
    if scheduler_type not in SCHEDULER_REGISTRY:
        available_schedulers = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Available schedulers: {available_schedulers}")

    scheduler = SCHEDULER_REGISTRY.get(scheduler_type)
    optimizer = scheduler_params.pop('optimizer', None)
    if optimizer is None:
        raise ValueError("Optimizer is required for scheduler")

    params = {}
    params.update(scheduler_params)
    return scheduler(optimizer, **params)


def build_stopper(stopper_type, **stopper_params):
    if stopper_type is None or stopper_type.lower() == "none":
        return None

    stopper_type = stopper_type.lower()
    if stopper_type not in STOPPER_REGISTRY:
        available_stoppers = list(STOPPER_REGISTRY.keys())
        raise ValueError(f"Unknown stopper: {stopper_type}. Available stoppers: {available_stoppers}")

    stopper = STOPPER_REGISTRY.get(stopper_type)
    default_params = {
        "early": {"patience": 10, "min_delta": 1e-4},
    }
    params = default_params.get(stopper_type, {})
    params.update(stopper_params)
    return stopper(**params)


# ===================================================================
# Main orchestrator
# ===================================================================

def run_experiment(config):
    try:
        torch.cuda.reset_peak_memory_stats()
        set_seed(config.seed)

        print("\n" + "="*60)
        print(f"RUN EXPERIMENT: {config.dataloader_type.upper()} - "
              f"{config.model_type.upper()} - {config.trainer_type.upper()} Trainer")
        print("="*60)

        ## 1. Data loaders
        train_transform = build_transform(config.train_transform_type, **config.train_transform_params)
        test_transform = build_transform(config.test_transform_type, **config.test_transform_params)
        data = build_dataloader(config.dataloader_type, **config.dataloader_params,
            train_transform=train_transform, test_transform=test_transform)
        show_dataloader_info(data, verbose=config.show_dataloader)

        ## 2. Modeler
        model = build_model(config.model_type, **config.model_params)
        loss_fn = build_loss_fn(config.loss_type, **config.loss_params)
        metrics = build_metrics(config.metric_list)
        modeler = build_modeler(config.modeler_type, model=model, loss_fn=loss_fn, metrics=metrics)
        show_modeler_info(modeler, verbose=config.show_modeler)

        ## 3. Trainer
        optimizer = build_optimizer(config.optimizer_type, model=model, **config.optimizer_params)
        scheduler = build_scheduler(config.scheduler_type, optimizer=optimizer, **config.scheduler_params)
        stopper = build_stopper(config.stopper_type, **config.stopper_params)
        trainer = build_trainer(config.trainer_type, modeler=modeler, optimizer=optimizer,
            scheduler=scheduler, stopper=stopper)
        show_trainer_info(trainer, verbose=config.show_trainer)

        ## 4. Training & Evaluation
        history = trainer.fit(data.train_loader(), config.num_epochs, valid_loader=data.valid_loader())
        results = trainer.test(data.test_loader())
        show_results(results)

    finally:
        # Phase 1: High-level components (dependency holders)
        if 'trainer' in locals():         del trainer
        if 'scheduler' in locals():       del scheduler
        if 'stopper' in locals():         del stopper

        # Phase 2: Model-dependent components (before deleting model)
        if 'modeler' in locals():
            if hasattr(modeler, 'model'): modeler.model = None
            del modeler

        if 'optimizer' in locals():
            if hasattr(optimizer, 'param_groups'): optimizer.param_groups.clear()
            del optimizer

        # Phase 3: Core components
        if 'metrics' in locals():         del metrics
        if 'loss_fn' in locals():         del loss_fn
        if 'model' in locals():
            if hasattr(model, 'cpu'): model.cpu()
            del model

        # Phase 4: Data components
        if 'data' in locals():            del data
        if 'test_transform' in locals():  del test_transform
        if 'train_transform' in locals(): del train_transform

        import gc
        gc.collect()

        torch.cuda.empty_cache()
        show_gpu_memory(config.show_memory)


# ===================================================================
# Utility Functions
# ===================================================================

def show_gpu_memory(verbose=True):
    if verbose:
        print("\n" + "-"*60)
        print("GPU MEMORY")
        print("-"*60)
        print(f" > Max allocated:     {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f" > Max reserved:      {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
        print("-"*60)
        print(f" > Current allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f" > Current reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


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


def show_dataloader_info(dataloader, verbose=True):
    if verbose:
        print()
        print(f" > Dataset Type:      {dataloader.data_dir}")
        print(f" > Categories:        {dataloader.categories}")

        train = dataloader.train_loader().dataset
        normal, anomal = count_labels(train)
        print(f" > Train data:        {len(train)} (normal={normal}, anomaly={anomal})")

        valid = None if dataloader.valid_loader() is None else dataloader.valid_loader().dataset
        if valid is not None:
            normal, anomal = count_labels(valid)
            print(f" > Valid data:        {len(valid)} (normal={normal}, anomaly={anomal})")

        test = dataloader.test_loader().dataset
        normal, anomal = count_labels(test)
        print(f" > Test data:         {len(test)} (normal={normal}, anomaly={anomal})")


def show_modeler_info(modeler, verbose=True):
    if verbose:
        print()
        print(f" > Modeler Type:      {type(modeler).__name__}")
        print(f" > Model Type:        {type(modeler.model).__name__}")
        print(f" > Total params.:     "
            f"{sum(p.numel() for p in modeler.model.parameters()):,}")
        print(f" > Trainable params.: "
            f"{sum(p.numel() for p in modeler.model.parameters() if p.requires_grad):,}")
        # print(f" > Learning Type:     {modeler.learning_type}")
        print(f" > Loss Function:     {type(modeler.loss_fn).__name__}")
        print(f" > Metrics:           {list(modeler.metrics.keys())}")
        print(f" > Device:            {modeler.device}")


def show_trainer_info(trainer, verbose=True):
    if verbose:
        print()
        # print(f" > Trainer Type:      {trainer.trainer_type}")
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


def show_results(results):
    print("\n" + "-"*60)
    print("EXPERIMENT RESULTS")
    print("-"*60)
    print(f" > AUROC:             {results['auroc']:.4f}")
    print(f" > AUPR:              {results['aupr']:.4f}")
    print(f" > Threshold:         {results['threshold']:.3e}")
    print("-"*60)
    print(f" > Accuracy:          {results['accuracy']:.4f}")
    print(f" > Precision:         {results['precision']:.4f}")
    print(f" > Recall:            {results['recall']:.4f}")
    print(f" > F1-Score:          {results['f1']:.4f}")


def run(data_type, model_type, categories=[], verbose=False):
    config = build_config(data_type, model_type)
    config.dataloader_params["categories"] = categories
    config.verbose = verbose
    config.show_dataloader=True
    config.show_modeler=True
    config.show_trainer=True
    config.show_memory=True
    config.num_epochs = 5
    
    if model_type in ["fastflow"]:
        config.optimizer_params=dict(lr=1e-5, weight_decay=1e-5)
    
    run_experiment(config)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    categories=["bottle", "tile"]
    # run("mvtec", "vanilla_ae", categories=categories)
    # run("mvtec", "unet_ae", categories=categories)
    # run("mvtec", "vanilla_vae", categories=categories)
    # run("mvtec", "unet_vae", categories=categories)
    # run("mvtec", "stfpm", categories=categories)

    # run("mvtec", "fastflow", categories=categories)
    run("mvtec", "draem", categories=categories)