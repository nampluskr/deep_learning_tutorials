"""
Main execution script for MVTec anomaly detection experiments
Orchestrates the complete training and evaluation pipeline
"""

import torch
import os
import sys
from time import time
from dataclasses import replace
import matplotlib.pyplot as plt

from mvtec import get_transforms, get_dataloaders
from models import get_model
from metrics import get_loss_fn, get_metrics
from train import set_seed, train_model, save_model, load_weights
from config import Config, print_config, save_config, create_directories, validate_config
from evaluate import evaluate_anomaly_detection, show_results


class Logger:
    """Simple logging utility for experiments"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = None
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Open log file
        self.log_file = open(log_path, 'w')
        print(f"Logging to: {log_path}")
    
    def log(self, message, print_console=True):
        """Log message to file and optionally print to console"""
        if print_console:
            print(message)
        
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()
    
    def close(self):
        """Close log file"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


def show_history(history):
    """Display training history"""
    print("\n" + "="*50)
    print("TRAINING HISTORY")
    print("="*50)
    
    # Find the best epoch based on validation loss
    if 'val_loss' in history:
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        best_val_loss = min(history['val_loss'])
        print(f"Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
    
    # Show final metrics
    print(f"Final Training Loss: {history['loss'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    
    # Show other metrics if available
    for key, values in history.items():
        if key not in ['loss', 'val_loss'] and not key.startswith('val_'):
            print(f"Final {key.upper()}: {values[-1]:.4f}")
    
    print("="*50 + "\n")


def plot_training_curves(history, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PSNR curves
    if 'psnr' in history:
        axes[0, 1].plot(history['psnr'], label='Train PSNR')
        if 'val_psnr' in history:
            axes[0, 1].plot(history['val_psnr'], label='Val PSNR')
        axes[0, 1].set_title('PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # SSIM curves
    if 'ssim' in history:
        axes[1, 0].plot(history['ssim'], label='Train SSIM')
        if 'val_ssim' in history:
            axes[1, 0].plot(history['val_ssim'], label='Val SSIM')
        axes[1, 0].set_title('SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # MSE curves
    if 'mse' in history:
        axes[1, 1].plot(history['mse'], label='Train MSE')
        if 'val_mse' in history:
            axes[1, 1].plot(history['val_mse'], label='Val MSE')
        axes[1, 1].set_title('MSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def setup_experiment(config):
    """Setup experiment environment"""
    print("Setting up experiment...")
    
    # Validate configuration
    validate_config(config)
    
    # Create necessary directories
    create_directories(config)
    
    # Set device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        config.device = "cpu"
    
    print(f"Using device: {config.device}")
    
    # Save configuration
    if config.save_log:
        config_path = os.path.join(config.config_save_dir, f"{config.experiment_name}_config.json")
        save_config(config, config_path)


def main(config):
    """Main execution function"""
    print("\n" + "="*60)
    print("MVTEC ANOMALY DETECTION EXPERIMENT")
    print("="*60)
    
    # Print configuration
    print_config(config)
    
    # Setup experiment
    setup_experiment(config)
    
    # Initialize logger
    logger = None
    if config.save_log:
        log_path = os.path.join(config.log_save_dir, f"{config.experiment_name}.log")
        logger = Logger(log_path)
        logger.log(f"Starting experiment: {config.experiment_name}")
    
    try:
        # Set seed for reproducibility
        set_seed(seed=config.seed, device=config.device)
        if logger:
            logger.log(f"Random seed set to: {config.seed}")
        
        # =====================================================================
        # 1. Data Loading
        # =====================================================================
        print("\n" + "-"*40)
        print("1. LOADING DATA")
        print("-"*40)
        
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
            test_transform=test_transform,
            num_workers=config.num_workers
        )
        
        print(f"Category: {config.category}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Valid samples: {len(valid_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        if logger:
            logger.log(f"Data loaded - Train: {len(train_loader.dataset)}, "
                      f"Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        # =====================================================================
        # 2. Model Loading
        # =====================================================================
        print("\n" + "-"*40)
        print("2. CREATING MODEL")
        print("-"*40)
        
        model = get_model(
            config.model_type,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            latent_dim=config.latent_dim
        )
        model = model.to(config.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {config.model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if logger:
            logger.log(f"Model created: {config.model_type} with {trainable_params:,} trainable parameters")
        
        # =====================================================================
        # 3. Model Training with Validation
        # =====================================================================
        print("\n" + "-"*40)
        print("3. TRAINING MODEL")
        print("-"*40)
        
        start_time = time()
        history = train_model(model, train_loader, config, valid_loader=valid_loader)
        elapsed_time = time() - start_time
        
        print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        
        if logger:
            logger.log(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Display and save training history
        show_history(history)
        
        if config.save_log:
            plot_save_path = os.path.join(config.log_save_dir, f"{config.experiment_name}_training_curves.png")
            plot_training_curves(history, plot_save_path)
        
        # =====================================================================
        # 4. Fine-tuning on Validation Data (Optional)
        # =====================================================================
        if config.fine_tuning and valid_loader is not None:
            print("\n" + "-"*40)
            print("4. FINE-TUNING MODEL")
            print("-"*40)
            
            fine_tune_config = replace(config, num_epochs=5)
            fine_tune_history = train_model(model, valid_loader, fine_tune_config)
            
            if logger:
                logger.log("Fine-tuning completed")
        
        # =====================================================================
        # 5. Evaluate Anomaly Detection Performance on Test Data
        # =====================================================================
        results = None
        if config.evaluation and test_loader is not None:
            print("\n" + "-"*40)
            print("5. EVALUATING MODEL")
            print("-"*40)
            
            results = evaluate_anomaly_detection(model, test_loader, config)
            show_results(results)
            
            if logger:
                logger.log(f"Evaluation results: {results}")
        
        # =====================================================================
        # 6. Save Model
        # =====================================================================
        if config.save_model:
            print("\n" + "-"*40)
            print("6. SAVING MODEL")
            print("-"*40)
            
            save_model(model, config)
            
            if logger:
                logger.log("Model saved successfully")
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return results
    
    except Exception as e:
        error_msg = f"Experiment failed with error: {str(e)}"
        print(f"\nERROR: {error_msg}")
        
        if logger:
            logger.log(error_msg)
        
        raise e
    
    finally:
        # Close logger
        if logger:
            logger.close()


def run_multiple_experiments(config_list):
    """Run multiple experiments with different configurations"""
    print(f"Running {len(config_list)} experiments...")
    
    all_results = []
    for i, config in enumerate(config_list, 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}/{len(config_list)}")
        print(f"{'='*60}")
        
        try:
            results = main(config)
            all_results.append({
                'config': config,
                'results': results,
                'status': 'success'
            })
        except Exception as e:
            print(f"Experiment {i} failed: {str(e)}")
            all_results.append({
                'config': config,
                'results': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = len(all_results) - successful
    
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\nSuccessful experiments:")
        for i, result in enumerate(all_results):
            if result['status'] == 'success':
                config = result['config']
                results = result['results']
                if results:
                    print(f"  {i+1}. {config.experiment_name} - AUROC: {results.get('auroc', 'N/A'):.3f}")
    
    return all_results


def create_example_configs():
    """Create example configurations for testing"""
    configs = []
    
    # Base configuration
    base_config = Config(
        data_dir="/path/to/mvtec",
        category="bottle",
        num_epochs=5,
        batch_size=16
    )
    
    # Vanilla AE with different learning rates
    for lr in [1e-4, 2e-4, 5e-4]:
        config = replace(base_config, 
                        model_type="vanilla_ae", 
                        learning_rate=lr)
        configs.append(config)
    
    # U-Net AE
    config = replace(base_config, model_type="unet_ae")
    configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Example usage
    
    # Single experiment
    config = Config(
        data_dir="/path/to/mvtec",
        category="bottle",
        model_type="vanilla_ae",
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-4,
        save_model=True,
        save_log=True
    )
    
    print("Running single experiment...")
    results = main(config)
    
    # Multiple experiments (uncomment to run)
    # print("\nRunning multiple experiments...")
    # configs = create_example_configs()
    # all_results = run_multiple_experiments(configs)
