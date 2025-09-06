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
from model_efficientad import EfficientADModel

MODEL_REGISTRY = {
    "vanilla_ae": VanillaAE,
    "unet_ae": UNetAE,
    "vanilla_vae": VanillaVAE,
    "unet_vae": UNetVAE,
    "stfpm": STFPMModel,
    "fastflow": FastFlowModel,
    "draem": DRAEMModel,
    "efficientad": EfficientADModel,
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

from modeler import AEModeler, VAEModeler
from modeler import STFPMModeler, FastFlowModeler, DRAEMModeler, EfficientADModeler

MODELER_REGISTRY = {
    "ae": AEModeler,
    "vae": VAEModeler,
    "stfpm": STFPMModeler,
    "fastflow": FastFlowModeler,
    "draem": DRAEMModeler,
    "efficientad": EfficientADModeler,
}


# ===================================================================
# Trainers / Optimizers / Schedulers / Stoppers
# ===================================================================

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from trainer import ReconstructionTrainer, EarlyStopper
from trainer import DistillationTrainer, FlowTrainer, MemoryTrainer

TRAINER_REGISTRY = {
    "reconstruction": ReconstructionTrainer,
    "distillation": DistillationTrainer,
    "flow": FlowTrainer,
    "memory": MemoryTrainer,
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
# Factory Functions - builders
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
    # if optimizer is None:
    #     raise ValueError("Optimizer is required for trainer")

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