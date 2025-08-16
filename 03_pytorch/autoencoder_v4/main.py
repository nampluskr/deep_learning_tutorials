import torch
from time import time
from dataclasses import replace

from config import Config, print_config
from train import train_model, set_seed, evaluate_anomaly_detection
from autoencoder import get_model
from mvtec import get_transforms, get_dataloaders


# =============================================================================
# Setup Configurations
# =============================================================================
config_list = [
    Config(
        model_type='unet_ae',
        num_epochs=5,
    ),
]

# =============================================================================
# Main training pipeline for autoencoder models
# =============================================================================
def main(config):
    print_config(config)
    set_seed(seed=config.seed, device=config.device)

    # =====================================================================
    # 1. Data Loading
    # =====================================================================
    print("\n*** Loading data...")
    train_transform, test_transform = get_transforms(
        img_size=config.img_size,
        normalize=config.normalize
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        category=config.category,
        batch_size=config.batch_size,
        valid_ratio=config.valid_ratio,
        train_transform=train_transform,
        test_transform=test_transform
    )

    # =====================================================================
    # 2. Model Loading
    # =====================================================================
    print("\n*** Loading model...")
    model = get_model(
        config.model_type,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_dim=config.latent_dim
    )
    model = model.to(config.device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f" > Model: {model.__class__.__name__}")
    print(f" > Parameters: {total_params:,}")
    print(f" > Size: {total_params*4 / 1024**2:.1f} MB")
    print(f" > Latent dim: {config.latent_dim}")

    # =====================================================================
    # 3. Model Training with Validation
    # =====================================================================
    print("\n*** Starting training with validation...")
    start_time = time()
    train_model(model, train_loader, config, valid_loader=valid_loader)
    elapsed_time = time() - start_time
    print(f" > Training completed in {elapsed_time:.1f}s")

    # =====================================================================
    # 4. Fine-tuning on Validation Data
    # =====================================================================
    print("\n*** Starting fine-tuning on validation data...")
    start_time = time()
    fine_tune_config = replace(config, num_epochs=5)
    train_model(model, valid_loader, fine_tune_config)
    elapsed_time = time() - start_time
    print(f" > Fine-tuning completed in {elapsed_time:.1f}s")

    # =====================================================================
    # 5. Evaluate Anomaly Detection Performance on Test Data
    # =====================================================================
    print("\n*** Evaluating anomaly detection performance...")
    start_time = time()
    test_results = evaluate_anomaly_detection(model, test_loader,
        method='mse', percentile=95)

    print(f" > AUROC: {test_results['auroc']:.4f}")
    print(f" > AUPR: {test_results['aupr']:.4f}")
    print(f" > F1 Score: {test_results['f1_score']:.4f}")
    print(f" > Accuracy: {test_results['accuracy']:.4f}")
    print(f" > Threshold: {test_results['threshold']:.6f}")
    print(f" > Normal samples: {test_results['normal_samples']}")
    print(f" > Anomaly samples: {test_results['anomaly_samples']}")
    print(f" > Defect types: {test_results['defect_types']}")

    elapsed_time = time() - start_time
    print(f" > Evaluation completed in {elapsed_time:.1f}s")

    # =====================================================================
    # 6. Save Model
    # =====================================================================
    if config.save_model:
        model_save_path = f"{config.model_type}_{config.category}_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f" > Model saved to {model_save_path}")


if __name__ == "__main__":

    for idx, config in enumerate(config_list):
        print(f"\n*** [{idx + 1}/{len(config_list)}] Training model")
        main(config)
        print(f"\n*** Model {idx + 1} training completed!")