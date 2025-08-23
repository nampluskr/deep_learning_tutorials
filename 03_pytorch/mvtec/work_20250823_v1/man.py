import torch
import os
import random
import numpy as np
from pathlib import Path

from config import Config
from data import MVTecDataset, get_transforms, split_train_valid, get_dataloader
from model import get_model, get_loss_fn, get_metric
from train import get_optimizer, get_scheduler, EarlyStopping, get_logger, Trainer


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config):
    """Main training function using config"""
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}, Dataset: {config.categories}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    output_dir = Path("experiments") / config.experiment_name
    logger = get_logger(str(output_dir))
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Config: {config}")
    
    # Create transforms
    train_transform, test_transform = get_transforms(config.img_size)
    
    # Create datasets
    train_dataset = MVTecDataset(
        data_dir=config.data_dir,
        categories=config.categories,
        split='train',
        transform=train_transform
    )
    
    valid_dataset = MVTecDataset(
        data_dir=config.data_dir,
        categories=config.categories,
        split='train',
        transform=test_transform
    )
    
    test_dataset = MVTecDataset(
        data_dir=config.data_dir,
        categories=config.categories,
        split='test',
        transform=test_transform
    )
    
    # Split train/validation
    train_subset, valid_subset = split_train_valid(
        train_dataset, valid_dataset, config.valid_ratio, config.seed
    )
    
    # Create dataloaders
    train_loader = get_dataloader(train_subset, config.batch_size, 'train')
    valid_loader = get_dataloader(valid_subset, config.batch_size, 'valid')
    test_loader = get_dataloader(test_dataset, config.batch_size, 'test')
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Valid samples: {len(valid_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = get_model(
        config.model_name,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_dim=config.latent_dim,
        img_size=config.img_size
    ).to(device)
    
    print(f"Model created: {model.model_type}")
    logger.info(f"Model: {model.model_type}")
    
    # Create loss function
    if config.model_name.lower() == 'vae':
        loss_fn = get_loss_fn(config.loss_type, beta=config.beta, mse_weight=config.mse_weight)
    elif config.model_name.lower() == 'fastflow':
        loss_fn = get_loss_fn('fastflow')
    else:
        loss_fn = get_loss_fn(config.loss_type, mse_weight=config.mse_weight, ssim_weight=config.ssim_weight)
    
    # Create metrics
    metrics = {}
    for metric_name in config.metric_names:
        if metric_name == 'vae' and config.model_name.lower() == 'vae':
            metrics[metric_name] = get_metric(metric_name, beta=config.beta, mse_weight=config.mse_weight)
        elif metric_name.startswith('fastflow') and config.model_name.lower() == 'fastflow':
            metrics[metric_name] = get_metric(metric_name)
        else:
            metrics[metric_name] = get_metric(metric_name)
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        optimizer_type=config.optimizer_type,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=config.scheduler_type
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        restore_best_weights=config.restore_best_weights
    ) if config.patience > 0 else None
    
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
    
    # Train model
    print("Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        num_epochs=config.num_epochs,
        valid_loader=valid_loader
    )
    
    # Save model
    model_path = output_dir / f"{config.model_name}_best.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    print(f"Model saved to: {model_path}")
    
    # Evaluate on test set (basic evaluation)
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, loss_fn, metrics, device)
    
    logger.info(f"Test results: {test_results}")
    print(f"Test results: {test_results}")
    
    return {
        'model': model,
        'history': history,
        'test_results': test_results,
        'config': config,
        'output_dir': output_dir
    }


@torch.no_grad()
def evaluate_model(model, dataloader, loss_fn, metrics, device):
    """Basic model evaluation on test set"""
    model.eval()
    total_loss = 0.0
    total_metrics = {name: 0.0 for name in metrics.keys()}
    num_batches = 0
    
    for batch_data in dataloader:
        # Move data to device
        for key, value in batch_data.items():
            if torch.is_tensor(value):
                batch_data[key] = value.to(device)
        
        # Use model's validate_step if available
        if hasattr(model, 'validate_step'):
            results = model.validate_step(batch_data, loss_fn, metrics)
        else:
            # Fallback evaluation
            outputs = model(batch_data)
            if model.model_type == 'fastflow':
                loss = loss_fn(outputs)
                results = {'loss': loss.item()}
                for metric_name, metric_fn in metrics.items():
                    results[metric_name] = metric_fn(outputs).item()
            else:
                if model.model_type == 'vae':
                    loss = loss_fn(outputs['reconstructed'], batch_data['target'], 
                                   outputs['mu'], outputs['logvar'])
                else:
                    loss = loss_fn(outputs['reconstructed'], batch_data['target'])
                results = {'loss': loss.item()}
                for metric_name, metric_fn in metrics.items():
                    if metric_name == 'vae' and model.model_type == 'vae':
                        results[metric_name] = metric_fn(outputs['reconstructed'], batch_data['target'],
                                                       outputs['mu'], outputs['logvar']).item()
                    else:
                        results[metric_name] = metric_fn(outputs['reconstructed'], batch_data['target']).item()
        
        total_loss += results['loss']
        for name in metrics.keys():
            total_metrics[name] += results[name]
        num_batches += 1
    
    # Average results
    avg_results = {'loss': total_loss / num_batches}
    for name in metrics.keys():
        avg_results[name] = total_metrics[name] / num_batches
    
    return avg_results


def run_vanilla_ae_experiment():
    """Run Vanilla AutoEncoder experiment"""
    config = Config(
        experiment_name="vanilla_ae_bottle",
        model_name="vanilla_ae",
        categories=['bottle'],
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        loss_type="combined",
        mse_weight=0.7,
        ssim_weight=0.3,
        metric_names=['mse', 'ssim', 'psnr']
    )
    return run(config)


def run_vae_experiment():
    """Run VAE experiment"""
    config = Config(
        experiment_name="vae_bottle",
        model_name="vae",
        categories=['bottle'],
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        loss_type="vae",
        beta=1.0,
        mse_weight=1.0,
        metric_names=['mse', 'ssim', 'psnr', 'vae']
    )
    return run(config)


def run_fastflow_experiment():
    """Run FastFlow experiment"""
    config = Config(
        experiment_name="fastflow_bottle",
        model_name="fastflow",
        categories=['bottle'],
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        loss_type="fastflow",
        metric_names=['fastflow_log_prob', 'fastflow_anomaly_score']
    )
    return run(config)


def run_multi_category_experiment():
    """Run experiment on multiple categories"""
    categories = ['bottle', 'cable', 'capsule']
    results = {}
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Running experiment on {category}")
        print(f"{'='*50}")
        
        config = Config(
            experiment_name=f"vanilla_ae_{category}",
            model_name="vanilla_ae",
            categories=[category],
            num_epochs=30,
            batch_size=32,
            learning_rate=1e-4,
            loss_type="combined",
            mse_weight=0.7,
            ssim_weight=0.3,
            metric_names=['mse', 'ssim', 'psnr']
        )
        
        results[category] = run(config)
    
    return results


def test_interface():
    """Test model interfaces and basic functionality"""
    print("Testing model interfaces...")
    
    # Test data loading
    train_transform, test_transform = get_transforms(256)
    dataset = MVTecDataset('/mnt/d/datasets/mvtec', ['bottle'], 'train', train_transform)
    dataloader = get_dataloader(dataset, 4, 'train')
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataloader batches: {len(dataloader)}")
    
    # Test batch
    batch = next(iter(dataloader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input shape: {batch['input'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    
    # Test models
    models_to_test = ['vanilla_ae', 'vae', 'fastflow']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        model = get_model(model_name, in_channels=3, out_channels=3, latent_dim=512).to(device)
        
        # Move batch to device
        test_batch = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                test_batch[k] = v.to(device)
            else:
                test_batch[k] = v
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(test_batch)
            print(f"Output keys: {list(outputs.keys())}")
            
            # Test train_step and validate_step
            if hasattr(model, 'train_step'):
                print(f"{model_name} has train_step method")
            if hasattr(model, 'validate_step'):
                print(f"{model_name} has validate_step method")
    
    print("Interface testing completed!")


if __name__ == "__main__":
    # Test interfaces first
    # test_interface()
    
    # Run individual experiments
    print("Running Vanilla AutoEncoder experiment...")
    vanilla_results = run_vanilla_ae_experiment()
    
    print("\nRunning VAE experiment...")
    vae_results = run_vae_experiment()
    
    print("\nRunning FastFlow experiment...")
    fastflow_results = run_fastflow_experiment()
    
    # Run multi-category experiment
    # print("\nRunning multi-category experiment...")
    # multi_results = run_multi_category_experiment()
    
    print("\nAll experiments completed!")