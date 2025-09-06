import os
import random
import numpy as np
import torch
import logging

from model_base import set_backbone_dir, check_backbone_files
from config import build_config, save_config
from builders import (build_transform, build_dataloader, build_model, build_loss_fn, build_metrics,
    build_modeler, build_optimizer, build_scheduler, build_stopper, build_trainer)


# ===================================================================
# Simple Run Pipeline
# ===================================================================

def run_experiment(data_type, model_type, categories=[], verbose=False):
    config = build_config(data_type, model_type)
    config.dataloader_params["categories"] = categories
    config.verbose = verbose
    config.num_epochs = 10

    if model_type in ["fastflow"]:
        config.optimizer_params=dict(lr=1e-5, weight_decay=1e-5)

    main_pipeline(config)


# ===================================================================
# Main Pipeline
# ===================================================================


def main_pipeline(config):
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
        device = get_device()
        modeler = build_modeler(config.modeler_type, model=model, loss_fn=loss_fn, metrics=metrics, device=device)
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

def get_logger(output_dir, log_name='experiment.log'):
    """Create logger for training progress tracking."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, log_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


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


# ===================================================================
# Show Functions
# ===================================================================

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
        logger.info(" ")
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


# ===================================================================
# Save Functions
# ===================================================================


def save_model(model, output_dir, logger=None):
    """Save model weights to PTH file."""
    folder_name = os.path.basename(os.path.normpath(output_dir))
    model_path = os.path.join(output_dir, f"model_{folder_name}.pth")

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


def load_model(model, output_dir, logger=None, model_filename=None):
    """Load model weights from PTH file."""
    from glob import glob

    try:
        if model_filename:
            model_path = os.path.join(output_dir, model_filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_filename}")
        else:
            model_pattern = os.path.join(output_dir, "model_*.pth")
            model_files = glob(model_pattern)

            if not model_files:
                raise FileNotFoundError(f"No model files found in: {output_dir}")

            model_path = max(model_files, key=os.path.getctime)

        if hasattr(model, 'load_model'):
            model.load_model(model_path)
        else:
            device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        filename = os.path.basename(model_path)
        success_msg = f" > Model loaded: {filename}"
        print(success_msg)
        if logger:
            logger.info(success_msg)

        return model

    except Exception as e:
        error_msg = f" > Model load failed: {str(e)}"
        print(error_msg)
        if logger:
            logger.error(error_msg)
        return None


def save_history(history, output_dir, logger=None):
    """Save training history to pickle file."""
    import pickle

    folder_name = os.path.basename(os.path.normpath(output_dir))
    history_path = os.path.join(output_dir, f"history_{folder_name}.pkl")

    try:
        history_data = {
            'experiment_name': folder_name,
            'history': history
        }
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)

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

    folder_name = os.path.basename(os.path.normpath(output_dir))
    results_path = os.path.join(output_dir, f"results_{folder_name}.json")

    try:
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)

        experiment_data = {
            'experiment_name': folder_name,
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


if __name__ == "__main__":

    categories=["bottle"]
    # run_experiment("mvtec", "vanilla_ae", categories=categories)
    # run_experiment("mvtec", "unet_ae", categories=categories)
    run_experiment("mvtec", "vanilla_vae", categories=categories)
    run_experiment("mvtec", "unet_vae", categories=categories)

    # run_experiment("mvtec", "stfpm", categories=categories)
    # run_experiment("mvtec", "fastflow", categories=categories)
    # run_experiment("mvtec", "draem", categories=categories)
    # run_experiment("mvtec", "efficientad", categories=categories)