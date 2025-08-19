import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from time import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

# Import our modules
from config import Config, get_preset_config, print_config, create_experiment_configs
from models import get_model, print_model_summary, save_model_checkpoint
from mvtec import get_transforms, get_dataloaders, analyze_dataset
from train import (
    train_model, set_seed, evaluate_anomaly_detection, 
    compute_detailed_metrics, compute_anomaly_scores
)


def setup_experiment(config: Config) -> None:
    """Setup experiment environment and logging"""
    
    # Set random seeds for reproducibility
    set_seed(config.experiment.seed, config.device)
    
    # Setup CUDA settings
    if config.device == 'cuda':
        torch.backends.cudnn.deterministic = config.experiment.deterministic
        torch.backends.cudnn.benchmark = config.experiment.benchmark
        
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create experiment directories
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.experiment.output_dir, 'config.yaml')
    config.save(config_path)
    
    print(f"Experiment setup complete: {config.experiment.run_name}")
    print(f"Output directory: {config.experiment.output_dir}")


def load_data(config: Config):
    """Load and prepare datasets"""
    
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    # Get transforms based on config
    train_transform, test_transform = get_transforms(
        img_size=config.data.img_size,
        normalize=config.data.normalize,
        augmentation_level=config.data.augmentation_level
    )
    
    print(f"Data directory: {config.data.data_dir}")
    print(f"Category: {config.data.category}")
    print(f"Image size: {config.data.img_size}")
    print(f"Augmentation level: {config.data.augmentation_level}")
    print(f"Normalization: {config.data.normalize}")
    
    # Load datasets
    try:
        train_loader, valid_loader, test_loader = get_dataloaders(
            data_dir=config.data.data_dir,
            category=config.data.category,
            batch_size=config.data.batch_size,
            valid_ratio=config.data.valid_ratio,
            train_transform=train_transform,
            test_transform=test_transform,
            img_size=config.data.img_size,
            load_masks=config.data.load_masks,
            cache_images=config.data.cache_images,
            num_workers=config.data.num_workers
        )
        
        print(f"✓ Training batches: {len(train_loader)}")
        print(f"✓ Validation batches: {len(valid_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        
        return train_loader, valid_loader, test_loader
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def create_model(config: Config):
    """Create and initialize model"""
    
    print(f"\n{'='*60}")
    print("CREATING MODEL")
    print(f"{'='*60}")
    
    # Prepare model parameters
    model_params = {
        'in_channels': config.model.in_channels,
        'out_channels': config.model.out_channels,
        'latent_dim': config.model.latent_dim,
        'input_size': config.data.img_size
    }
    
    # Add model-specific parameters
    if 'resnet' in config.model.model_type or 'vgg' in config.model.model_type or 'efficientnet' in config.model.model_type:
        model_params.update({
            'arch': config.model.backbone_arch,
            'pretrained': config.model.pretrained,
            'freeze_backbone': config.model.freeze_backbone
        })
    
    if config.model.model_type == 'beta_vae':
        model_params['beta'] = config.model.beta
    elif config.model.model_type == 'wae':
        model_params['lambda_reg'] = config.model.lambda_reg
    
    # Create model
    try:
        model = get_model(config.model.model_type, **model_params)
        model = model.to(config.device)
        
        # Print model summary
        print_model_summary(model, input_size=(config.model.in_channels, config.data.img_size, config.data.img_size), device=config.device)
        
        # Initialize weights if specified
        if hasattr(model, 'apply') and config.model.init_type != 'default':
            from models.base.utils import init_weights
            model.apply(lambda m: init_weights(m, config.model.init_type, config.model.init_gain))
            print(f"✓ Weights initialized with {config.model.init_type} initialization")
        
        return model
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        raise


def train_and_validate(model, train_loader, valid_loader, config: Config):
    """Training and validation loop"""
    
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    print(f"Model: {config.model.model_type}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Loss type: {config.training.loss_type}")
    print(f"Device: {config.device}")
    
    # Training
    start_time = time()
    try:
        train_model(model, train_loader, config, valid_loader=valid_loader)
        training_time = time() - start_time
        print(f"✓ Training completed in {training_time:.1f}s ({training_time/60:.1f}m)")
        
        return training_time
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise


def fine_tune_model(model, valid_loader, config: Config):
    """Fine-tuning on validation data"""
    
    if config.training.num_epochs < 10:  # Skip fine-tuning for short training
        print("Skipping fine-tuning for short training")
        return 0
    
    print(f"\n{'='*60}")
    print("FINE-TUNING")
    print(f"{'='*60}")
    
    # Create fine-tuning config with reduced epochs
    fine_tune_epochs = max(3, config.training.num_epochs // 4)
    
    # Temporarily modify config for fine-tuning
    original_epochs = config.training.num_epochs
    original_lr = config.training.learning_rate
    
    config.training.num_epochs = fine_tune_epochs
    config.training.learning_rate = original_lr * 0.1  # Reduce learning rate
    
    print(f"Fine-tuning epochs: {fine_tune_epochs}")
    print(f"Fine-tuning learning rate: {config.training.learning_rate}")
    
    start_time = time()
    try:
        train_model(model, valid_loader, config)
        fine_tune_time = time() - start_time
        print(f"✓ Fine-tuning completed in {fine_tune_time:.1f}s")
        
        # Restore original config
        config.training.num_epochs = original_epochs
        config.training.learning_rate = original_lr
        
        return fine_tune_time
        
    except Exception as e:
        print(f"✗ Fine-tuning failed: {e}")
        # Restore original config even on failure
        config.training.num_epochs = original_epochs
        config.training.learning_rate = original_lr
        raise


def evaluate_model(model, test_loader, config: Config):
    """Comprehensive model evaluation"""
    
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    start_time = time()
    
    # Standard evaluation
    try:
        if len(config.anomaly_detection.evaluation_methods) == 1:
            # Single method evaluation
            method = config.anomaly_detection.evaluation_methods[0]
            percentile = config.anomaly_detection.threshold_percentiles[0] if config.anomaly_detection.threshold_percentiles else 95
            
            results = evaluate_anomaly_detection(
                model, test_loader, 
                method=method, 
                percentile=percentile, 
                normalize_input=config.data.normalize
            )
            
            print(f"Method: {method} (percentile: {percentile})")
            print(f"✓ AUROC: {results['auroc']:.4f}")
            print(f"✓ AUPR: {results['aupr']:.4f}")
            print(f"✓ F1 Score: {results['f1_score']:.4f}")
            print(f"✓ Accuracy: {results['accuracy']:.4f}")
            print(f"✓ Threshold: {results['threshold']:.6f}")
            
            all_results = {f"{method}_p{percentile}": results}
            
        else:
            # Multi-method evaluation
            print("Running comprehensive evaluation...")
            all_results = compute_detailed_metrics(
                model, test_loader,
                methods=config.anomaly_detection.evaluation_methods,
                percentiles=config.anomaly_detection.threshold_percentiles,
                normalize_input=config.data.normalize
            )
            
            # Print summary
            print(f"\nEvaluation Summary:")
            print(f"{'Method':<15} {'Percentile':<10} {'AUROC':<8} {'AUPR':<8} {'F1':<8} {'Accuracy':<8}")
            print("-" * 70)
            
            for method_key, results in all_results.items():
                method = results['method']
                percentile = results['percentile']
                print(f"{method:<15} {percentile:<10} {results['auroc']:.4f}   {results['aupr']:.4f}   "
                      f"{results['f1_score']:.4f}   {results['accuracy']:.4f}")
        
        # Additional statistics
        best_result = max(all_results.values(), key=lambda x: x['auroc'])
        print(f"\nBest result: {best_result['method']} (p{best_result['percentile']}) - AUROC: {best_result['auroc']:.4f}")
        
        print(f"✓ Normal samples: {best_result['normal_samples']}")
        print(f"✓ Anomaly samples: {best_result['anomaly_samples']}")
        print(f"✓ Defect types: {best_result['defect_types']}")
        
        eval_time = time() - start_time
        print(f"✓ Evaluation completed in {eval_time:.1f}s")
        
        return all_results, eval_time
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        raise


def save_results(model, config: Config, results: Dict, training_time: float, eval_time: float):
    """Save model and results"""
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    try:
        # Save model checkpoint
        if config.experiment.save_model:
            checkpoint_path = os.path.join(
                config.experiment.checkpoint_dir, 
                f"{config.model.model_type}_{config.data.category}_model.pth"
            )
            
            model_params = {
                'model_type': config.model.model_type,
                'in_channels': config.model.in_channels,
                'out_channels': config.model.out_channels,
                'latent_dim': config.model.latent_dim,
                'input_size': config.data.img_size
            }
            
            additional_info = {
                'config': config.to_dict(),
                'training_time': training_time,
                'eval_time': eval_time,
                'results': results
            }
            
            save_model_checkpoint(
                model, checkpoint_path,
                model_type=config.model.model_type,
                model_params=model_params,
                **additional_info
            )
            
            print(f"✓ Model saved: {checkpoint_path}")
        
        # Save detailed results
        results_path = os.path.join(config.experiment.output_dir, 'results.json')
        detailed_results = {
            'experiment_config': config.to_dict(),
            'evaluation_results': results,
            'performance_metrics': {
                'training_time_seconds': training_time,
                'evaluation_time_seconds': eval_time,
                'total_time_seconds': training_time + eval_time
            },
            'model_info': {
                'model_type': config.model.model_type,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"✓ Results saved: {results_path}")
        
        # Create summary report
        create_summary_report(config, results, training_time, eval_time)
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        raise


def create_summary_report(config: Config, results: Dict, training_time: float, eval_time: float):
    """Create a human-readable summary report"""
    
    report_path = os.path.join(config.experiment.output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ANOMALY DETECTION EXPERIMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Experiment: {config.experiment.run_name}\n")
        f.write(f"Timestamp: {time()}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Model: {config.model.model_type}\n")
        f.write(f"Dataset: {config.data.category}\n")
        f.write(f"Image size: {config.data.img_size}\n")
        f.write(f"Latent dimension: {config.model.latent_dim}\n")
        f.write(f"Batch size: {config.data.batch_size}\n")
        f.write(f"Epochs: {config.training.num_epochs}\n")
        f.write(f"Learning rate: {config.training.learning_rate}\n")
        f.write(f"Loss type: {config.training.loss_type}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"Training time: {training_time:.1f}s ({training_time/60:.1f}m)\n")
        f.write(f"Evaluation time: {eval_time:.1f}s\n")
        f.write(f"Total time: {(training_time + eval_time):.1f}s ({(training_time + eval_time)/60:.1f}m)\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*40 + "\n")
        
        if isinstance(results, dict):
            best_result = max(results.values(), key=lambda x: x.get('auroc', 0))
            f.write(f"Best AUROC: {best_result.get('auroc', 0):.4f} ({best_result.get('method', 'unknown')})\n")
            f.write(f"Best AUPR: {best_result.get('aupr', 0):.4f}\n")
            f.write(f"Best F1: {best_result.get('f1_score', 0):.4f}\n")
            f.write(f"Best Accuracy: {best_result.get('accuracy', 0):.4f}\n")
            f.write(f"Normal samples: {best_result.get('normal_samples', 'unknown')}\n")
            f.write(f"Anomaly samples: {best_result.get('anomaly_samples', 'unknown')}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Summary report: {report_path}")


def visualize_results(model, test_loader, config: Config, results: Dict):
    """Create visualizations of results"""
    
    if not config.experiment.save_plots:
        return
    
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    try:
        # Create plots directory
        plots_dir = os.path.join(config.experiment.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Results comparison plot
        if len(results) > 1:
            create_results_comparison_plot(results, plots_dir, config.experiment.plot_format)
        
        # 2. Sample reconstructions
        if config.experiment.save_reconstructions:
            create_reconstruction_samples(model, test_loader, config, plots_dir)
        
        # 3. Anomaly score distribution
        create_anomaly_score_distribution(model, test_loader, config, plots_dir)
        
        print(f"✓ Visualizations saved in: {plots_dir}")
        
    except Exception as e:
        print(f"⚠ Warning: Visualization failed: {e}")


def create_results_comparison_plot(results: Dict, plots_dir: str, plot_format: str):
    """Create comparison plot of different evaluation methods"""
    
    methods = []
    aurocs = []
    auprs = []
    f1s = []
    
    for key, result in results.items():
        methods.append(f"{result['method']}_p{result['percentile']}")
        aurocs.append(result['auroc'])
        auprs.append(result['aupr'])
        f1s.append(result['f1_score'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # AUROC comparison
    axes[0].bar(range(len(methods)), aurocs, alpha=0.7)
    axes[0].set_title('AUROC Comparison')
    axes[0].set_ylabel('AUROC')
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45)
    
    # AUPR comparison
    axes[1].bar(range(len(methods)), auprs, alpha=0.7, color='orange')
    axes[1].set_title('AUPR Comparison')
    axes[1].set_ylabel('AUPR')
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45)
    
    # F1 comparison
    axes[2].bar(range(len(methods)), f1s, alpha=0.7, color='green')
    axes[2].set_title('F1 Score Comparison')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels(methods, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'results_comparison.{plot_format}'), dpi=300, bbox_inches='tight')
    plt.close()


def create_reconstruction_samples(model, test_loader, config: Config, plots_dir: str):
    """Create sample reconstruction visualizations"""
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get a batch of test samples
    for batch in test_loader:
        images = batch['image'][:config.experiment.num_reconstruction_samples].to(device)
        labels = batch['label'][:config.experiment.num_reconstruction_samples]
        
        with torch.no_grad():
            if 'vae' in config.model.model_type.lower():
                if config.model.model_type == 'wae':
                    reconstructed, _ = model(images)
                else:
                    reconstructed, _, _, _ = model(images)
            else:
                reconstructed, _, _ = model(images)
        
        # Prepare images for visualization
        if config.data.normalize:
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            images = images * std + mean
            reconstructed = torch.clamp(reconstructed, 0, 1)  # Sigmoid output
        
        # Convert to numpy
        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Create visualization
        n_samples = min(config.experiment.num_reconstruction_samples, len(images))
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
        
        for i in range(n_samples):
            # Original image
            img = np.transpose(images[i], (1, 2, 0))
            axes[0, i].imshow(np.clip(img, 0, 1))
            axes[0, i].set_title(f'Original\n{"Normal" if labels[i] == 0 else "Anomaly"}')
            axes[0, i].axis('off')
            
            # Reconstructed image
            recon = np.transpose(reconstructed[i], (1, 2, 0))
            axes[1, i].imshow(np.clip(recon, 0, 1))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
            
            # Difference
            diff = np.abs(img - recon)
            axes[2, i].imshow(np.clip(diff, 0, 1))
            axes[2, i].set_title('Difference')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'reconstruction_samples.{config.experiment.plot_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        break


def create_anomaly_score_distribution(model, test_loader, config: Config, plots_dir: str):
    """Create anomaly score distribution plot"""
    
    # Compute anomaly scores
    method = config.anomaly_detection.evaluation_methods[0] if config.anomaly_detection.evaluation_methods else 'mse'
    scores, labels, _ = compute_anomaly_scores(model, test_loader, method, config.data.normalize)
    
    # Separate normal and anomaly scores
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
    if len(anomaly_scores) > 0:
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True)
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Anomaly Score Distribution ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'score_distribution.{config.experiment.plot_format}'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def run_single_experiment(config: Config) -> Dict[str, Any]:
    """Run a single experiment with given configuration"""
    
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT: {config.experiment.run_name}")
    print(f"{'='*80}")
    
    # Print configuration
    print_config(config, sections=['data', 'model', 'training'])
    
    total_start_time = time()
    
    try:
        # 1. Setup experiment
        setup_experiment(config)
        
        # 2. Load data
        train_loader, valid_loader, test_loader = load_data(config)
        
        # 3. Create model
        model = create_model(config)
        
        # 4. Training
        training_time = train_and_validate(model, train_loader, valid_loader, config)
        
        # 5. Fine-tuning (optional)
        fine_tune_time = fine_tune_model(model, valid_loader, config)
        total_training_time = training_time + fine_tune_time
        
        # 6. Evaluation
        results, eval_time = evaluate_model(model, test_loader, config)
        
        # 7. Save results
        save_results(model, config, results, total_training_time, eval_time)
        
        # 8. Visualizations
        visualize_results(model, test_loader, config, results)
        
        # Final summary
        total_time = time() - total_start_time
        best_result = max(results.values(), key=lambda x: x.get('auroc', 0)) if results else {}
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best AUROC: {best_result.get('auroc', 0):.4f}")
        print(f"Results saved in: {config.experiment.output_dir}")
        
        return {
            'config': config,
            'results': results,
            'best_auroc': best_result.get('auroc', 0),
            'total_time': total_time,
            'success': True
        }
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("EXPERIMENT FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        
        return {
            'config': config,
            'error': str(e),
            'success': False
        }


def run_multiple_experiments(configs: List[Config]) -> List[Dict[str, Any]]:
    """Run multiple experiments"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING {len(configs)} EXPERIMENTS")
    print(f"{'='*80}")
    
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n>>> EXPERIMENT {i+1}/{len(configs)} <<<")
        
        result = run_single_experiment(config)
        all_results.append(result)
        
        if result['success']:
            print(f"✓ Experiment {i+1} completed - AUROC: {result['best_auroc']:.4f}")
        else:
            print(f"✗ Experiment {i+1} failed: {result['error']}")
    
    # Summary of all experiments
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"Successful: {len(successful)}/{len(configs)}")
    print(f"Failed: {len(failed)}/{len(configs)}")
    
    if successful:
        best_overall = max(successful, key=lambda x: x['best_auroc'])
        print(f"Best overall AUROC: {best_overall['best_auroc']:.4f}")
        print(f"Best config: {best_overall['config'].model.model_type} on {best_overall['config'].data.category}")
    
    return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("MVTEC ANOMALY DETECTION PIPELINE")
    print("=" * 80)
    
    # ==========================================================================
    # EXPERIMENT CONFIGURATIONS
    # ==========================================================================
    
    # Choose experiment type
    experiment_type = 'single'  # 'single', 'preset_comparison', 'parameter_sweep'
    
    if experiment_type == 'single':
        # Single experiment with preset config
        config = get_preset_config('oled_optimized')
        
        # Customize if needed
        config.data.category = 'bottle'
        config.data.data_dir = '/mnt/d/datasets/mvtec'  # Update this path
        config.training.num_epochs = 15
        config.experiment.save_plots = True
        config.experiment.save_reconstructions = True
        
        result = run_single_experiment(config)
        
    elif experiment_type == 'preset_comparison':
        # Compare different preset configurations
        preset_names = ['vanilla_baseline', 'unet_advanced', 'vae_disentangled']
        configs = []
        
        for preset in preset_names:
            config = get_preset_config(preset)
            config.data.category = 'bottle'
            config.data.data_dir = '/mnt/d/datasets/mvtec'
            config.training.num_epochs = 10  # Shorter for comparison
            config.experiment.run_name = f"{preset}_{config.data.category}"
            configs.append(config)
        
        results = run_multiple_experiments(configs)
        
    elif experiment_type == 'parameter_sweep':
        # Parameter sweep experiment
        base_config = get_preset_config('vanilla_baseline')
        base_config.data.data_dir = '/mnt/d/datasets/mvtec'
        
        param_grid = {
            'model.model_type': ['vanilla_ae', 'unet_ae'],
            'model.latent_dim': [256, 512],
            'training.learning_rate': [1e-3, 1e-4]
        }
        
        configs = create_experiment_configs(base_config, param_grid)
        
        # Set unique run names
        for i, config in enumerate(configs):
            config.experiment.run_name = f"sweep_experiment_{i:02d}"
            config.training.num_epochs = 8  # Shorter for sweep
        
        results = run_multiple_experiments(configs)
    
    print("\nPipeline execution completed!")


if __name__ == "__main__":
    main()