import os
import random
import numpy as np
import torch
from datetime import datetime

# ===================================================================
# Global Configuration
# ===================================================================

from types import SimpleNamespace

BASE_CONFIGS = SimpleNamespace(
    backbone_dir="/mnt/d/backbones",
    output_dir="./experiments",
    log_name="experiment.log",
    num_epochs=10,
    seed=42,

    verbose=True,
    show_dataloader=True,
    show_modeler=True,
    show_trainer=True,
    show_memory=True,

    save_log=True,
    save_config=True,
    save_model=True,
    save_history=True,
    save_results=True,

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
    "efficientad": SimpleNamespace(
        modeler_type="efficientad",
        trainer_type="reconstruction",

        model_type="efficientad",
        model_params=dict(
            model_size="small",  # "small" or "medium"
            teacher_out_channels=384,
            padding=False,
            pad_maps=True,
            use_imagenet_penalty=True,
        ),
        loss_type=None,  # EfficientAD는 자체 loss 계산
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],
    ),
}

TRAIN_CONFIGS = {
    "reconstruction": SimpleNamespace(
        trainer_type="reconstruction",
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-4, weight_decay=1e-5),
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = os.path.join(config.output_dir, f"{timestamp}_{data_type.lower()}_{model_type.lower()}")
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
# Main pipe line
# ===================================================================

from model_base import set_backbone_dir, check_backbone_files

def run_experiment(config):
    logger = None
    try:
        ## 0. Experiment setup
        torch.cuda.reset_peak_memory_stats()
        if config.save_log:
            logger = get_logger(config.output_dir, log_name=config.log_name)

        show_header(config, logger=logger)
        set_seed(config.seed)
        set_backbone_dir(config.backbone_dir)
        check_backbone_files(config.backbone_dir)

        ## 1. Data loaders
        train_transform = build_transform(config.train_transform_type, **config.train_transform_params)
        test_transform = build_transform(config.test_transform_type, **config.test_transform_params)
        data = build_dataloader(config.dataloader_type, **config.dataloader_params,
            train_transform=train_transform, test_transform=test_transform)
        show_dataloader_info(data, verbose=config.show_dataloader, logger=logger)

        ## 2. Modeler
        model = build_model(config.model_type, **config.model_params)
        loss_fn = build_loss_fn(config.loss_type, **config.loss_params)
        metrics = build_metrics(config.metric_list)
        modeler = build_modeler(config.modeler_type, model=model, loss_fn=loss_fn, metrics=metrics)
        show_modeler_info(modeler, verbose=config.show_modeler, logger=logger)

        ## 3. Trainer
        optimizer = build_optimizer(config.optimizer_type, model=model, **config.optimizer_params)
        scheduler = build_scheduler(config.scheduler_type, optimizer=optimizer, **config.scheduler_params)
        stopper = build_stopper(config.stopper_type, **config.stopper_params)
        trainer = build_trainer(config.trainer_type, modeler=modeler, optimizer=optimizer,
            scheduler=scheduler, stopper=stopper, logger=logger)
        show_trainer_info(trainer, verbose=config.show_trainer, logger=logger)

        ## 4. Training & Evaluation
        history = trainer.fit(data.train_loader(), config.num_epochs, valid_loader=data.valid_loader())
        results = trainer.test(data.test_loader())
        show_results(results, logger=logger)

        ## 5. Save Model and Results
        if config.save_config: save_config(config, config.output_dir, logger=logger)
        if config.save_model: save_model(model, config.output_dir, logger=logger)
        if config.save_history: save_history(history, config.output_dir, logger=logger)
        if config.save_results: save_results(results, config.output_dir, logger=logger)

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
        show_gpu_memory(config.show_memory, logger=logger)


# ===================================================================
# Utility Functions
# ===================================================================

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


def show_header(config, logger=None):
    """Show experiment header information."""
    header_line = "\n" + "="*60
    exp_line = f"RUN EXPERIMENT: {config.dataloader_type.upper()} - {config.model_type.upper()}"
    footer_line = "="*60

    # Console output with newline
    print(header_line)
    print(exp_line)
    print(footer_line)

    # Log output with leading newline
    if logger:
        logger.info(" ")  # 줄바꿈을 빈 공간으로 기록
        logger.info("="*60)
        logger.info(exp_line)
        logger.info("="*60)


def show_dataloader_info(dataloader, verbose=True, logger=None):
    if verbose:
        info_lines = [
            "",
            f" > Dataset Type:      {dataloader.data_dir}",
            f" > Categories:        {dataloader.categories}"
        ]
        train = dataloader.train_loader().dataset
        normal, anomal = count_labels(train)
        info_lines.append(f" > Train data:        {len(train)} (normal={normal}, anomaly={anomal})")

        valid = None if dataloader.valid_loader() is None else dataloader.valid_loader().dataset
        if valid is not None:
            normal, anomal = count_labels(valid)
            info_lines.append(f" > Valid data:        {len(valid)} (normal={normal}, anomaly={anomal})")

        test = dataloader.test_loader().dataset
        normal, anomal = count_labels(test)
        info_lines.append(f" > Test data:         {len(test)} (normal={normal}, anomaly={anomal})")

        for line in info_lines:
            print(line)
            if logger:
                if line == "": logger.info(" ")
                else: logger.info(line)


def show_modeler_info(modeler, verbose=True, logger=None):
    if verbose:
        info_lines = [
            "",
            f" > Modeler Type:      {type(modeler).__name__}",
            f" > Model Type:        {type(modeler.model).__name__}",
            f" > Total params.:     {sum(p.numel() for p in modeler.model.parameters()):,}",
            f" > Trainable params.: {sum(p.numel() for p in modeler.model.parameters() if p.requires_grad):,}",
            f" > Loss Function:     {type(modeler.loss_fn).__name__}",
            f" > Metrics:           {list(modeler.metrics.keys())}",
            f" > Device:            {modeler.device}"
        ]
    for line in info_lines:
            print(line)
            if logger:
                if line == "": logger.info(" ")
                else: logger.info(line)


def show_trainer_info(trainer, verbose=True, logger=None):
    if verbose:
        info_lines = [
            "",
            f" > Optimizer:         {type(trainer.optimizer).__name__}",
            f" > Learning Rate:     {trainer.optimizer.param_groups[0]['lr']}"
        ]
        if trainer.scheduler is not None:
            info_lines.append(f" > Scheduler:         {type(trainer.scheduler).__name__}")
        else:
            info_lines.append(f" > Scheduler:         None")

        if trainer.stopper is not None:
            info_lines.append(f" > Stopper:           {type(trainer.stopper).__name__}")
            if hasattr(trainer.stopper, 'patience'):
                info_lines.append(f" > Patience:          {trainer.stopper.patience}")
            if hasattr(trainer.stopper, 'min_delta'):
                info_lines.append(f" > Min Delta:         {trainer.stopper.min_delta}")
            if hasattr(trainer.stopper, 'max_epoch'):
                info_lines.append(f" > Max Epochs:        {trainer.stopper.max_epoch}")
        else:
            info_lines.append(f" > Stopper:           None")

        for line in info_lines:
            print(line)
            if logger:
                if line == "": logger.info(" ")
                else: logger.info(line)


def show_results(results, logger=None):
    result_lines = [
        "",
        "-" * 60,
        "EXPERIMENT RESULTS",
        "-" * 60,
        f" > AUROC:             {results['auroc']:.4f}",
        f" > AUPR:              {results['aupr']:.4f}",
        f" > Threshold:         {results['threshold']:.3e}",
        "-" * 60,
        f" > Accuracy:          {results['accuracy']:.4f}",
        f" > Precision:         {results['precision']:.4f}",
        f" > Recall:            {results['recall']:.4f}",
        f" > F1-Score:          {results['f1']:.4f}",
        "",
    ]
    for line in result_lines:
        print(line)
        if logger:
            if line == "": logger.info(" ")
            else: logger.info(line)


def show_gpu_memory(verbose=True, logger=None):
    if verbose:
        memory_lines = [
            "",
            "-" * 60,
            "GPU MEMORY",
            "-" * 60,
            f" > Max allocated:     {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB",
            f" > Max reserved:      {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB",
            "-" * 60,
            f" > Current allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB",
            f" > Current reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        ]
        for line in memory_lines:
            print(line)
            if logger:
                if line == "": logger.info(" ")
                else: logger.info(line)



def save_config(config, output_dir, logger=None):
    """Save experiment configuration to JSON file."""
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = os.path.join(output_dir, f"config_{timestamp}.json")

    try:
        config_dict = {}    # Convert config to dictionary
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if isinstance(value, (int, float, str, bool, list, dict)):
                    config_dict[attr] = value
                elif hasattr(value, '__dict__'):
                    config_dict[attr] = str(value)
                else:
                    config_dict[attr] = str(value)

        config_data = {
            'timestamp': timestamp,
            'configuration': config_dict
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        filename = os.path.basename(config_path)
        success_msg = f" > Config saved: {filename}"
        print(success_msg)
        if logger: 
            logger.info(success_msg)

    except Exception as e:
        error_msg = f" > Config save failed: {str(e)}"
        print(error_msg)
        if logger: 
            logger.error(error_msg)


def save_model(model, output_dir, logger=None):
    """Save model weights to PTH file."""
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(output_dir, f"model_{timestamp}.pth")

    try:
        if hasattr(model, 'save_model'): 
            model.save_model(model_path)
        else: 
            torch.save(model.state_dict(), model_path)

        filename = os.path.basename(model_path)
        success_msg = f" > Model saved: {filename}"
        print(success_msg)
        if logger: 
            logger.info(success_msg)

    except Exception as e:
        error_msg = f" > Model save failed: {str(e)}"
        print(error_msg)
        if logger: 
            logger.error(error_msg)


def save_history(history, output_dir, logger=None):
    """Save training history to pickle file."""
    import pickle
    from datetime import datetime

    if not history:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_path = os.path.join(output_dir, f"history_{timestamp}.pkl")

    try:
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        filename = os.path.basename(history_path)
        success_msg = f" > History saved: {filename}"
        print(success_msg)
        if logger: 
            logger.info(success_msg)

    except Exception as e:
        error_msg = f" > History save failed: {str(e)}"
        print(error_msg)
        if logger: 
            logger.error(error_msg)


def save_results(results, output_dir, logger=None):
    """Save experiment results to JSON file."""
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")

    try:
        # Convert results to JSON serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)

        # Create results data
        experiment_data = {
            'timestamp': timestamp,
            'results': serializable_results
        }

        with open(results_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        filename = os.path.basename(results_path)
        success_msg = f" > Results saved: {filename}"
        print(success_msg)
        if logger: 
            logger.info(success_msg)

    except Exception as e:
        error_msg = f" > Results save failed: {str(e)}"
        print(error_msg)
        if logger: 
            logger.error(error_msg)


def show_results(results, logger=None):
    """Display and save experiment results."""
    result_lines = [
        "",  # empty line
        "-" * 60,
        "EXPERIMENT RESULTS",
        "-" * 60,
        f" > AUROC:             {results['auroc']:.4f}",
        f" > AUPR:              {results['aupr']:.4f}",
        f" > Threshold:         {results['threshold']:.3e}",
        "-" * 60,
        f" > Accuracy:          {results['accuracy']:.4f}",
        f" > Precision:         {results['precision']:.4f}",
        f" > Recall:            {results['recall']:.4f}",
        f" > F1-Score:          {results['f1']:.4f}",
        ""  # empty line at the end
    ]

    # Print to console and log to file (including empty lines)
    for line in result_lines:
        print(line)
        if logger:
            if line == "":  # empty line as space
                logger.info(" ")
            else:
                logger.info(line)


def run(data_type, model_type, categories=[], verbose=False):
    config = build_config(data_type, model_type)
    config.dataloader_params["categories"] = categories
    config.verbose = verbose
    config.show_dataloader=True
    config.show_modeler=True
    config.show_trainer=True
    config.show_memory=True
    config.num_epochs = 10

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

    categories=["bottle"]
    run("mvtec", "vanilla_ae", categories=categories)
    run("mvtec", "unet_ae", categories=categories)
    # run("mvtec", "vanilla_vae", categories=categories)
    # run("mvtec", "unet_vae", categories=categories)

    # run("mvtec", "stfpm", categories=categories)
    # run("mvtec", "fastflow", categories=categories)
    # run("mvtec", "draem", categories=categories)
    # run("mvtec", "efficientad", categories=categories)