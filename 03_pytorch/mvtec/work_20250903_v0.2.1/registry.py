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

LOGGER_REGISTRY = {}


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


def build_modeler(model, modeler_type, loss_fn=None, metrics=None, device=None, **kwargs):
    modeler_type = modeler_type.lower()
    if modeler_type not in MODELER_REGISTRY:
        available_modelers = list(MODELER_REGISTRY.keys())
        raise ValueError(f"Unknown modeler: {modeler_type}. Available modelers: {available_modelers}")

    modeler_class = MODELER_REGISTRY.get(modeler_type)

    modeler_params = {
        'loss_fn': loss_fn,
        'metrics': metrics,
        'device': device,
    }
    modeler_params.update(kwargs)

    # Remove None values to let modeler use defaults
    modeler_params = {k: v for k, v in modeler_params.items() if v is not None}

    return modeler_class(model, **modeler_params)


def build_trainer(modeler, optimizer, trainer_type, **trainer_params):
    trainer_type = trainer_type.lower()
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer: {trainer_type}. Available trainers: {available_trainers}")

    trainer = TRAINER_REGISTRY.get(trainer_type)
    params = {}
    params.update(trainer_params)
    return trainer(modeler, optimizer, **params)


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


def build_logger(logger_type, **logger_params):
    if logger_type is None or logger_type.lower() == "none":
        return None

    logger_type = logger_type.lower()
    if logger_type not in LOGGER_REGISTRY:
        available_stoppers = list(LOGGER_REGISTRY.keys())
        raise ValueError(f"Unknown logger: {logger_type}. Available loggers: {available_stoppers}")

    logger = LOGGER_REGISTRY.get(logger_type)
    params = {}
    params.update(logger_params)
    return logger(**params)


