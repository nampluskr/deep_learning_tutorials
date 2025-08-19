"""
Main execution script for MVTec anomaly detection experiments
Orchestrates the complete training and evaluation pipeline
"""

from config import Config, show_config
from mvtec import MVTecDataset
from train import get_transforms, split_dataset, get_dataloader
from models import get_model, save_model, show_model_info
from metrics import get_loss_fn, get_metrics
from train import Trainer, get_optimizer, get_scheduler, set_seed
from evaluate import evaluate_anomaly_detection, show_results


# =============================================================================
# Setup Configurations
# =============================================================================
common_config = dict(
    num_epochs=10,
    # save_log=True,
    # save_model=True,
    fine_tuning=True,
    early_stop=True,
    evaluation=True,
)
config_list = [
    Config(
        model_type='vanilla_ae',
        **common_config,
    ),
    Config(
        model_type='unet_ae',
        **common_config,
    ),
]

# =============================================================================
# Run training pipeline for autoencoder models
# =============================================================================
def run(config):
    # if config.save_log:
    #     save_log(config)

    show_config(config)
    set_seed(seed=config.seed, device=config.device)

    # =====================================================================
    # 1. Data Loading
    # =====================================================================
    print("\n*** Loading data...")

    train_transform, test_transform = get_transforms(img_size=config.img_size)
    train_dataset = MVTecDataset(config.data_dir, config.category, "train", transform=train_transform)
    valid_dataset = MVTecDataset(config.data_dir, config.category, "train", transform=test_transform)
    test_dataset = MVTecDataset(config.data_dir, config.category, "test", transform=test_transform)

    train_dataset, valid_dataset = split_dataset(train_dataset, valid_dataset, 
        valid_ratio=config.valid_ratio, seed=config.seed)

    loader_params = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
    # loader_params = {}
    train_loader = get_dataloader(train_dataset, config.batch_size, "train", **loader_params)
    valid_loader = get_dataloader(valid_dataset, config.batch_size, "valid", **loader_params)
    test_loader = get_dataloader(test_dataset, config.batch_size, "test",  **loader_params)

    # =====================================================================
    # 2. Model Loading
    # =====================================================================
    print("\n*** Loading model...")

    model = get_model(
        model_type=config.model_type,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_dim=config.latent_dim
    ).to(config.device)
    show_model_info(model)

    loss_fn = get_loss_fn(loss_type="combined")
    metrics = get_metrics(metric_names=["psnr", "ssim"])
    optimizer = get_optimizer(model, "adamw")
    scheduler = get_scheduler(optimizer, "default")

    # =====================================================================
    # 3. Model Training with Validation
    # =====================================================================
    print("\n*** Starting training with validation...")

    trainer = Trainer(model, optimizer, loss_fn, metrics=metrics)
    history = trainer.fit(train_loader, num_epochs=config.num_epochs,
        valid_loader=valid_loader, scheduler=scheduler,
        early_stop=config.early_stop)

    # =====================================================================
    # 4. Fine-tuning on Validation Data
    # =====================================================================
    if config.fine_tuning and valid_loader is not None:
        print("\n*** Starting fine-tuning on validation data...")

        history = trainer.fit(valid_loader, num_epochs=5)

    # =====================================================================
    # 5. Evaluate Anomaly Detection Performance on Test Data
    # =====================================================================
    if config.evaluation and test_loader is not None:
        print("\n*** Evaluating anomaly detection performance...")

        results = evaluate_anomaly_detection(model, test_loader,
            method="mse", threshold_method="percentile", percentile=95)
        show_results(results)

    # =====================================================================
    # 6. Save Model
    # =====================================================================
    if config.save_model:
        print("\n*** Saving model and configuration...")

        save_model(model, config.model_path)


# =============================================================================
# Save log, model and configuration
# =============================================================================

# class Logger:
#     """Redirect stdout to both console and file"""
#     def __init__(self, filepath):
#         self.terminal = sys.stdout
#         self.log = open(filepath, "a", encoding="utf-8")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()


# def save_log(config):
#     """Save configuration settings to a log file"""
#     results_dir = os.path.join(os.getcwd(), "results")
#     os.makedirs(results_dir, exist_ok=True)

#     prefix = get_config_prefix(config)
#     save_dir = os.path.join(results_dir, prefix)
#     os.makedirs(save_dir, exist_ok=True)

#     log_filename = prefix + "_log.txt"
#     log_path = os.path.join(save_dir, log_filename)

#     sys.stdout = Logger(log_path)
#     print(f" > Log saved to ./results/.../{log_filename}")


if __name__ == "__main__":

    from datetime import datetime

    for idx, config in enumerate(config_list):
        print(f"\n*** Model [{idx + 1}/{len(config_list)}] training started...")
        run(config)
        timestamp = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"\n*** Model [{idx + 1}/{len(config_list)}] "
              f"training completed at {timestamp}.\n")
