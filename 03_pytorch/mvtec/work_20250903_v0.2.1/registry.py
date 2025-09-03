from types import SimpleNamespace
import warnings

# ===================================================================
# Dataloaders
# ===================================================================

from dataloaders.dataloader_base import TrainTransform, TestTransform
from dataloaders.dataloader_mvtec import MVTecDataloader
from dataloaders.dataloader_btad import BTADDataloader
from dataloaders.dataloader_visa import VisADataloader

DATALOADER_REGISTRY = {
    "mvtec": MVTecDataloader,
    "btad": BTADDataloader,
    "visa": VisADataloader,
}

TRANSFORM_REGISTRY = {
    "train": TrainTransform,
    "test": TestTransform,
}

# ===================================================================
# Models / Losses
# ===================================================================

import torch.nn as nn

from models.model_ae import VanillaAE, UNetAE, AELoss, AECombinedLoss
from models.model_stfpm import STFPMModel, STFPMLoss
from models.model_patchcore import PatchcoreModel
from models.model_padim import PadimModel
from models.model_fastflow import FastflowModel, FastflowLoss
from models.model_draem import DraemModel, DraemLoss

MODEL_REGISTRY = {
    "vanilla_ae": VanillaAE,
    "unet_ae": UNetAE,
    "stfpm": STFPMModel,
    "patchcore": PatchcoreModel,
    "padim": PadimModel,
    "fastflow": FastflowModel,
    "draem": DraemModel,
}

LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    "ae": AELoss,
    "combined": AECombinedLoss,
    "stfpm": STFPMLoss,
    "fastflow": FastflowLoss,
    "draem": DraemLoss,
}

# ===================================================================
# Metrics
# ===================================================================

from metrics.metrics_base import AUROCMetric, AUPRMetric, AccuracyMetric, PrecisionMetric
from metrics.metrics_base import RecallMetric, F1Metric, OptimalThresholdMetric
from metrics.metrics_flow import LogLikelihoodMetric, BPDMetric, FlowJacobianMetric
from metrics.metrics_gradient import PSNRMetric, SSIMMetric, LPIPSMetric, FeatureSimilarityMetric
from metrics.metrics_memory import MahalanobisDistanceMetric, MemoryEfficiencyMetric

METRIC_REGISTRY = {
    "auroc": AUROCMetric,
    "aupr": AUPRMetric,
    "acc": AccuracyMetric,
    "prec": PrecisionMetric,
    "recall": RecallMetric,
    "f1": F1Metric,
    "thershold": OptimalThresholdMetric,
    "psnr": PSNRMetric,
    "ssim": SSIMMetric,
    "lpips": LPIPSMetric,
    "feature_sim": FeatureSimilarityMetric,
}


# ===================================================================
# Modelers
# ===================================================================

from modelers.modeler_ae import AEModeler
from modelers.modeler_stfpm import STFPMModeler
from modelers.modeler_patchcore import PatchcoreModeler
from modelers.modeler_padim import PadimModeler
from modelers.modeler_fastflow import FastflowModeler
from modelers.modeler_draem import DraemModeler

MODELER_REGISTRY = {
    "ae": AEModeler,
    "stfpm": STFPMModeler,
    "patchcore": PatchcoreModeler,
    "padim": PadimModeler,
    "fastflow": FastflowModeler,
    "draem": DraemModeler,
}


# ===================================================================
# Trainers / Optimizers / Schedulers / Stoppers
# ===================================================================

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from trainers.trainer_gradient import GradientTrainer
from trainers.trainer_memory import MemoryTrainer
from trainers.trainer_flow import FlowTrainer
from trainers.trainer_base import EarlyStopper, EpochStopper

TRAINER_REGISTRY = {
    "gradient": GradientTrainer,
    "memory": MemoryTrainer,
    "flow": FlowTrainer,
}

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

SCHEDULER_REGISTRY = {
    "step": lr_scheduler.StepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
    "plateau": lr_scheduler.ReduceLROnPlateau,
    "exponential": lr_scheduler.ExponentialLR,
    "multistep": lr_scheduler.MultiStepLR,
}

STOPPER_REGISTRY = {
    "early": EarlyStopper,
    "epoch": EpochStopper,
}


# ===================================================================
# Factory Functions
# ===================================================================

def build_dataloader(dataloader_type, **dataloader_params):
    dataloader_type = dataloader_type.lower()
    if dataloader_type not in DATALOADER_REGISTRY:
        available_dataloaders = list(DATALOADER_REGISTRY.keys())
        raise ValueError(f"Unknown dataloader: {dataloader_type}. Available dataloaders: {available_dataloaders}")

    dataloader = DATALOADER_REGISTRY.get(dataloader_type)
    params = {}
    params.update(dataloader_params)
    return dataloader(**params)


def build_transform(transform_type, **transform_params):
    transform_type = transform_type.lower()
    if transform_type not in TRANSFORM_REGISTRY:
        available_transforms = list(TRANSFORM_REGISTRY.keys())
        raise ValueError(f"Unknown transform: {transform_type}. Available transforms: {available_transforms}")

    transform = TRANSFORM_REGISTRY.get(transform_type)
    params = {}
    params.update(transform_params)
    return transform(**params)
    

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
    params = {}
    params.update(modeler_params)
    return modeler(**params)


def build_trainer(modeler, trainer_type, **trainer_params):
    trainer_type = trainer_type.lower()
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer: {trainer_type}. Available trainers: {available_trainers}")

    trainer = TRAINER_REGISTRY.get(trainer_type)
    params = {}
    params.update(trainer_params)
    return trainer(modeler, **params)


def build_optimizer(model, optimizer_type, **optimizer_params):
    if optimizer_type is None or optimizer_type.lower() == "none":
        return None

    optimizer_type = optimizer_type.lower()
    if optimizer_type not in OPTIMIZER_REGISTRY:
        available_optimizers = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Available optimizers: {available_optimizers}")

    optimizer = OPTIMIZER_REGISTRY.get(optimizer_type)
    params = {"lr": 1e-3}
    params.update(optimizer_params)
    return optimizer(model.parameters(), **params)


def build_scheduler(optimizer, scheduler_type, **scheduler_params):
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None

    scheduler_type = scheduler_type.lower()
    if scheduler_type not in SCHEDULER_REGISTRY:
        available_schedulers = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Available schedulers: {available_schedulers}")

    scheduler = SCHEDULER_REGISTRY.get(scheduler_type)
    params = {}
    if scheduler_type == "step":
        params = {"step_size": 30, "gamma": 0.1}
    elif scheduler_type == "cosine":
        params = {"T_max": 100}
    elif scheduler_type == "exponential":
        params = {"gamma": 0.9}
    elif scheduler_type == "multistep":
        params = {"milestones": [30, 60], "gamma": 0.1}

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
    params = {}
    if stopper_type == "early":
        params = {"patience": 10, "min_delta": 1e-4}
    elif stopper_type == "epoch":
        params = {"max_epoch": 100}

    params.update(stopper_params)
    return stopper(**params)

# ===================================================================
# Helper functions
# ===================================================================

# ---- Helper functions for dataloader parameters ----
def _get_dataloader_params(model_type, dataloader_type, base_params):
    dataloader_params = base_params.copy()

    if model_type in ["patchcore", "padim"]:
        dataloader_params["shuffle_train"] = False
        dataloader_params["shuffle_test"] = False
    else:
        dataloader_params["shuffle_train"] = True
        dataloader_params["shuffle_test"] = False

    return dataloader_params


# ===================================================================
# Main orchestrator
# ===================================================================

def run_experiment(dataloader_type, model_type, config):
    dataset_loader_class = DATALOADER_REGISTRY.get(dataloader_type.lower())
    if dataset_loader_class is None:
        available_dataloaders = list(DATALOADER_REGISTRY.keys())
        raise ValueError(f"Unknown dataloader: {dataloader_type}. Available datasets: {available_dataloaders}")

    if model_type.lower() not in MODELER_REGISTRY:
        available_models = list(MODELER_REGISTRY.keys())
        raise ValueError(f"Unknown model_type: {model_type}. Available: {available_models}")

    base_dataloader_params = {
        "train_batch_size": config.dataloader["train_batch_size"],
        "test_batch_size": config.dataloader["test_batch_size"],
        "num_workers": config.dataloader["num_workers"],
        "pin_memory": config.dataloader["pin_memory"],
    }
    dataloader_params = _get_dataloader_params(model_type, dataloader_type, base_dataloader_params)

    dataset_instance = dataset_loader_class(
        dataloader_type=dataloader_type,
        dataloader_params=getattr(config, "dataloader_params", {}),
        categories=getattr(config, "categories", None),
        img_size=config.dataloader["img_size"],
        train_batch_size=dataloader_params["train_batch_size"],
        test_batch_size=dataloader_params["test_batch_size"],
        num_workers=dataloader_params["num_workers"],
        shuffle_train=dataloader_params.get("shuffle_train", True),
        shuffle_test=dataloader_params.get("shuffle_test", False),
        pin_memory=dataloader_params.get("pin_memory", True),
    )

    train_loader = dataset_instance.get_train_loader()
    valid_loader = dataset_instance.get_valid_loader()
    test_loader = dataset_instance.get_test_loader()

    model_params = getattr(config, "model_params", {})
    modeler_params = getattr(config, "modeler_params", {})
    modeler_params.update({"model_params": model_params})

    try:
        modeler = build_modeler(model_type, **modeler_params)
    except Exception as e:
        raise RuntimeError(f"Failed to create modeler for '{model_type}': {e}")

    if config.loss_type and config.loss_type != "none":
        try:
            loss_function = build_loss(config.loss_type, **config.loss_params)
            if hasattr(modeler, "set_loss_fn"):
                modeler.set_loss_fn(loss_function)
        except Exception as e:
            warnings.warn(f"Failed to create loss function: {e}")

    optimizer = None
    if hasattr(modeler, "parameters") and config.optimizer_type and config.optimizer_type != "none":
        try:
            optimizer = build_optimizer(modeler, config.optimizer_type, **config.optimizer_params)
        except Exception as e:
            warnings.warn(f"Failed to create optimizer: {e}")

    scheduler = None
    if optimizer is not None and config.scheduler_type and config.scheduler_type != "none":
        try:
            scheduler = build_scheduler(optimizer, config.scheduler_type, **config.scheduler_params)
        except Exception as e:
            warnings.warn(f"Failed to create scheduler: {e}")

    stopper = None
    if hasattr(config, "stopper_type") and config.stopper_type and config.stopper_type != "none":
        try:
            stopper = build_stopper(config.stopper_type, **getattr(config, "stopper_params", {}))
        except Exception as e:
            warnings.warn(f"Failed to create stopper: {e}")

    try:
        trainer = build_trainer(modeler, config.trainer_type, **config.trainer_params)
    except Exception as e:
        raise RuntimeError(f"Failed to create trainer: {e}")

    if hasattr(trainer, "set_optimizer") and optimizer is not None:
        trainer.set_optimizer(optimizer)
    if hasattr(trainer, "set_scheduler") and scheduler is not None:
        trainer.set_scheduler(scheduler)
    if hasattr(trainer, "set_stopper") and stopper is not None:
        trainer.set_stopper(stopper)

    epochs = config.trainer_params.get("epochs", 1)

    print(f"  Training {model_type} on {dataloader_type} for {epochs} epochs...")
    history = trainer.fit(train_loader, num_epochs=epochs, valid_loader=valid_loader)

    print(f"  Predicting on test set...")
    scores, labels = trainer.predict(test_loader)

    return {
        "history": history,
        "scores": scores,
        "labels": labels,
        "config_summary": {
            "train_batch": dataloader_params["train_batch_size"],
            "test_batch": dataloader_params["test_batch_size"],
            "epochs": epochs,
            "shuffle_train": dataloader_params.get("shuffle_train", True),
        }
    }

def get_available_combinations():
    available_dataloaders = list(DATALOADER_REGISTRY.keys())
    available_models = list(MODELER_REGISTRY.keys())
    return available_dataloaders, available_models
