from data import get_anomaly_detection_dataloaders, get_evaluation_dataloader
from model import get_model, get_loss_fn, get_metric
from train import Trainer, get_optimizer, get_scheduler, get_logger, EarlyStopping
from config import Config
import torch


def run_vanilla_ae_experiment(config):
    """Run vanilla autoencoder experiment"""
    print(f"Running Vanilla Autoencoder experiment...")
    print(f"Categories: {config.categories}")

    # Setup logging
    logger = get_logger('experiments/vanilla_ae_experiment')

    # Get dataloaders
    train_loader, valid_loader, test_loader = get_anomaly_detection_dataloaders(config)

    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Valid dataset size: {len(valid_loader.dataset)}")
    logger.info(f"Test dataset size: {len(test_loader.dataset)}")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model('vanilla_ae',
                     in_channels=config.in_channels,
                     out_channels=config.out_channels,
                     latent_dim=config.latent_dim).to(device)

    # Setup training components
    optimizer = get_optimizer(model, 'adamw', lr=1e-4, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, 'reduce_plateau', patience=5, factor=0.5)
    loss_fn = get_loss_fn('combined', mse_weight=0.7, ssim_weight=0.3)

    # Setup metrics
    metrics = {
        'mse': get_metric('mse'),
        'ssim': get_metric('ssim'),
        'psnr': get_metric('psnr'),
    }

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, restore_best_weights=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler=scheduler,
        logger=logger,
        early_stopping=early_stopping
    )

    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        num_epochs=50,  # Reduced for testing
        valid_loader=valid_loader
    )

    # Save final checkpoint
    trainer.save_checkpoint('experiments/vanilla_ae_experiment/final_model.pth')

    return history, trainer


def run_vae_experiment(config):
    """Run VAE experiment"""
    print(f"Running VAE experiment...")
    print(f"Categories: {config.categories}")

    # Setup logging
    logger = get_logger('experiments/vae_experiment')

    # Get dataloaders
    train_loader, valid_loader, test_loader = get_anomaly_detection_dataloaders(config)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VAE model
    model = get_model('vae',
                     in_channels=config.in_channels,
                     out_channels=config.out_channels,
                     latent_dim=config.latent_dim).to(device)

    optimizer = get_optimizer(model, 'adam', lr=1e-4, weight_decay=1e-6)
    scheduler = get_scheduler(optimizer, 'cosine', T_max=50)

    # VAE-specific loss and metrics
    loss_fn = get_loss_fn('vae', beta=1.0, mse_weight=1.0)
    metrics = {
        'vae': get_metric('vae', beta=1.0),
        'mse': get_metric('mse'),
        'ssim': get_metric('ssim'),
    }

    early_stopping = EarlyStopping(patience=15, min_delta=1e-5, restore_best_weights=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler=scheduler,
        logger=logger,
        early_stopping=early_stopping
    )

    history = trainer.fit(train_loader, num_epochs=50, valid_loader=valid_loader)
    trainer.save_checkpoint('experiments/vae_experiment/final_model.pth')

    return history, trainer


def run_multi_category_experiments():
    """Run experiments on multiple categories"""
    base_config = Config()
    categories_list = [
        ['bottle'], ['cable'], ['capsule'],  # Single categories
        ['bottle', 'cable', 'capsule'],      # Multiple categories
    ]

    results = {}

    for categories in categories_list:
        experiment_name = '_'.join(categories)
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*50}")

        # Update config for current categories
        config = Config(
            data_dir=base_config.data_dir,
            categories=categories,
            img_size=256,
            batch_size=32,
            valid_ratio=0.2,
            seed=42
        )

        try:
            # Run vanilla AE experiment
            history, trainer = run_vanilla_ae_experiment(config)

            # Store results
            final_loss = history['val_loss'][-1] if history['val_loss'] else history['loss'][-1]
            final_ssim = history['val_ssim'][-1] if history['val_ssim'] else history['ssim'][-1]
            final_psnr = history['val_psnr'][-1] if history['val_psnr'] else history['psnr'][-1]

            results[experiment_name] = {
                'categories': categories,
                'final_loss': final_loss,
                'final_ssim': final_ssim,
                'final_psnr': final_psnr,
                'history': history
            }

            print(f"✅ {experiment_name} completed!")
            print(f"   Final loss: {final_loss:.4f}")
            print(f"   Final SSIM: {final_ssim:.4f}")
            print(f"   Final PSNR: {final_psnr:.2f}")

        except Exception as e:
            print(f"❌ {experiment_name} failed: {str(e)}")
            results[experiment_name] = {'error': str(e)}

    return results


def test_interface():
    """Test the new dict-based interface"""
    print("Testing new dict-based interface...")

    config = Config(
        data_dir='/mnt/d/datasets/mvtec',
        categories=['bottle'],
        img_size=64,  # Smaller for quick testing
        batch_size=4,
        valid_ratio=0.2,
        seed=42
    )

    # Test data loading
    print("1. Testing data loading...")
    train_loader, valid_loader, test_loader = get_anomaly_detection_dataloaders(config)

    train_batch = next(iter(train_loader))
    print(f"   Train batch keys: {list(train_batch.keys())}")
    print(f"   Input shape: {train_batch['input'].shape}")
    print(f"   Target shape: {train_batch['target'].shape}")
    print(f"   All labels normal: {torch.all(train_batch['label'] == 0).item()}")

    # Test model forward
    print("2. Testing model forward...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model('vanilla_ae', in_channels=3, latent_dim=128).to(device)

    # Move batch to device
    for key in train_batch:
        if torch.is_tensor(train_batch[key]):
            train_batch[key] = train_batch[key].to(device)

    model_output = model(train_batch)
    print(f"   Model output keys: {list(model_output.keys())}")
    print(f"   Reconstructed shape: {model_output['reconstructed'].shape}")

    # Test loss function
    print("3. Testing loss function...")
    loss_fn = get_loss_fn('mse')
    loss = loss_fn(model_output['reconstructed'], train_batch['target'])
    print(f"   Loss value: {loss.item():.6f}")

    # Test metric function
    print("4. Testing metric function...")
    metric_fn = get_metric('ssim')
    ssim_value = metric_fn(model_output['reconstructed'], train_batch['target'])
    print(f"   SSIM value: {ssim_value.item():.6f}")

    print("✅ Interface test completed successfully!")


# Configuration list for different experiments
config_list = [
    Config(
        data_dir='/mnt/d/datasets/mvtec',
        categories=['bottle', 'cable', 'capsule'],
        img_size=256,
        batch_size=32,
        valid_ratio=0.2,
        seed=42
    ),
]


def run(config):
    """Main run function for single configuration"""
    print(f"Running experiment with config:")
    print(f"  Categories: {config.categories}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")

    # Run vanilla AE experiment
    history, trainer = run_vanilla_ae_experiment(config)

    print("Experiment completed!")
    return history


if __name__ == "__main__":
    # Test the interface first
    print("="*60)
    print("TESTING NEW INTERFACE")
    print("="*60)
    # test_interface()

    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)

    # Run single experiment
    for config in config_list:
        history = run(config)
        break  # Run only first config for testing

    # Uncomment below to run multiple experiments
    # print("\n" + "="*60)
    # print("MULTI-CATEGORY EXPERIMENTS")
    # print("="*60)
    # multi_results = run_multi_category_experiments()

    # # Print summary
    # print("\n" + "="*60)
    # print("EXPERIMENT SUMMARY")
    # print("="*60)
    # for exp_name, result in multi_results.items():
    #     if 'error' in result:
    #         print(f"❌ {exp_name}: {result['error']}")
    #     else:
    #         print(f"✅ {exp_name}: Loss={result['final_loss']:.4f}, SSIM={result['final_ssim']:.4f}")

    print("\nAll experiments completed!")