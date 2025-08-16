import torch
from time import time

from config import Config, print_config
from train import train_model, set_seed
from autoencoder import load_model
from mvtec import get_transforms, get_dataloaders


# =============================================================================
# Setup Configurations
# =============================================================================
config_list = [
    Config(
        model_type='unet_ae',
        num_epochs=5,
        # device='cuda',
        # latent_dim=256,
        # out_channels=3,  # Fixed: Use 3 channels for RGB images
    ),
]


def main():
    """Main training pipeline for autoencoder models"""

    for idx, config in enumerate(config_list):
        print(f"\n*** [{idx + 1}/{len(config_list)}] Training model")
        print_config(config)
        set_seed(seed=config.seed, device=config.device)

        # =====================================================================
        # Data Loading
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
        # Model Loading
        # =====================================================================
        print("\n*** Loading model...")
        model = load_model(config.model_type, **config.model_params)
        model = model.to(config.device)
        total_params = sum(p.numel() for p in model.parameters())

        print(f" > Model: {model.__class__.__name__}")
        print(f" > Parameters: {total_params:,}")
        print(f" > Size: {total_params*4 / 1024**2:.1f} MB")
        print(f" > Latent dim: {config.latent_dim}")

        # =====================================================================
        # Model Training with Validation
        # =====================================================================
        print("\n*** Starting training with validation...")
        start_time = time()
        train_model(model, train_loader, config, valid_loader=valid_loader)
        elapsed_time = time() - start_time
        print(f"\n*** Training completed in {elapsed_time:.1f}s")

        # =====================================================================
        # Additional Training without Validation (Fine-tuning)
        # =====================================================================
        # print("\n*** Starting additional training without validation...")
        # start_time = time()

        # # Create a copy of config with fewer epochs for fine-tuning
        # fine_tune_config = Config(**config.__dict__)
        # fine_tune_config.num_epochs = 5

        # # Combine train and validation data for fine-tuning
        # combined_dataset = torch.utils.data.ConcatDataset([
        #     train_loader.dataset,
        #     valid_loader.dataset
        # ])
        # combined_loader = torch.utils.data.DataLoader(
        #     combined_dataset,
        #     batch_size=config.batch_size,
        #     shuffle=True,
        #     drop_last=True,
        #     num_workers=4 if torch.cuda.is_available() else 0,
        #     pin_memory=True if torch.cuda.is_available() else False,
        #     persistent_workers=True if torch.cuda.is_available() else False
        # )

        # train_model(model, combined_loader, fine_tune_config)
        # elapsed_time = time() - start_time
        # print(f" > Additional training completed in {elapsed_time:.1f}s")

        # =====================================================================
        # Save Model
        # =====================================================================
        # if config.save_model:
        #     model_save_path = f"{config.model_type}_{config.category}_model.pth"
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f" > Model saved to {model_save_path}")

        # print(f"\n*** Model {idx + 1} training completed!")


if __name__ == "__main__":

    main()