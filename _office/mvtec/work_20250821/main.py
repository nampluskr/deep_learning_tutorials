import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import json
import pandas as pd
import logging
from pathlib import Path

# Import custom modules
from data import create_mvtec_loaders, get_dataset_info
from model import VanillaEncoder, VanillaDecoder, VanillaAutoencoder, VAEEncoder, VAEDecoder, VAE
from padim_model import PaDiM, PaDiMTrainingWrapper
from fastflow_model import FastFlow
from stfpm_model import STFPM
from train import (Trainer, create_vanilla_ae_loss_config, create_vae_loss_config, create_standard_metrics,
                  create_padim_loss_config, create_fastflow_loss_config, create_stfpm_loss_config)
from evaluate import (compute_anomaly_scores, evaluate_anomaly_detection, create_score_configs, 
                     evaluate_model, plot_training_curves, plot_performance_comparison, 
                     visualize_reconstructions)


# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================

# Data configuration
DATA_DIR = '/home/namu/myspace/NAMU/datasets/mvtec'
CATEGORIES = ['bottle',]
IMG_SIZE = 256
VALID_RATIO = 0.2

# Training configuration
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LATENT_DIM = 512

# Model configuration
RECONSTRUCTION_LOSS = 'mse'  # Options: 'mse', 'l1', 'bce'
KL_WEIGHT = 0.1  # KL divergence weight for VAE
# Pretrained weights paths (set to None to use random initialization)
WEIGHTS_PATHS = {
    'resnet18': None,  # Path to ResNet18 weights file (e.g., './weights/resnet18.pth')
    'resnet34': None,  # Path to ResNet34 weights file
    'resnet50': None,  # Path to ResNet50 weights file
}

# Hardware configuration
NUM_WORKERS = 4
DEVICE = 'auto'  # Options: 'auto', 'cpu', 'cuda'

# Output configuration
OUTPUT_DIR = './results'
SAVE_MODELS = True
SHOW_PLOTS = False

# Experiment configuration
SEED = 42
# MODELS = ['vanilla_ae', 'vae', 'padim', 'fastflow', 'stfpm']
MODELS = ['vanilla_ae', 'vae']
# MODELS = ['vae']
# MODELS = ['padim']
# MODELS = ['fastflow']
# MODELS = ['stfpm']

# ============================================================================


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'experiment.log')
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler with timestamp and level
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate console output
    logger.propagate = False
    
    return logger


def get_model_config(model_name, latent_dim, reconstruction_loss, kl_weight):
    """Get model configuration for specified model type"""
    model_configs = {
        'vanilla_ae': {
            'encoder_class': VanillaEncoder,
            'decoder_class': VanillaDecoder,
            'model_class': VanillaAutoencoder,
            'loss_config_fn': lambda: create_vanilla_ae_loss_config(reconstruction_loss, 1.0),
            'display_name': 'Vanilla Autoencoder',
            'training_type': 'standard'
        },
        'vae': {
            'encoder_class': VAEEncoder,
            'decoder_class': VAEDecoder,
            'model_class': VAE,
            'loss_config_fn': lambda: create_vae_loss_config(reconstruction_loss, 1.0, kl_weight),
            'display_name': 'Variational Autoencoder',
            'training_type': 'standard'
        },
        'padim': {
            'model_class': PaDiM,
            'loss_config_fn': lambda: create_padim_loss_config(),
            'display_name': 'PaDiM',
            'training_type': 'statistical',
            'backbone': 'resnet18',
            'layers': ['layer2', 'layer3'],
            'weights_key': 'resnet18'
        },
        'fastflow': {
            'model_class': FastFlow,
            'loss_config_fn': lambda: create_fastflow_loss_config(1.0),
            'display_name': 'FastFlow',
            'training_type': 'flow',
            'backbone': 'resnet18',
            'layers': ['layer2', 'layer3'],
            'weights_key': 'resnet18'
        },
        'stfpm': {
            'model_class': STFPM,
            'loss_config_fn': lambda: create_stfpm_loss_config([1.0, 1.0, 1.0]),
            'display_name': 'STFPM',
            'training_type': 'teacher_student',
            'backbone': 'resnet18',
            'layers': ['layer1', 'layer2', 'layer3'],
            'weights_key': 'resnet18'
        }
    }
    
    return model_configs.get(model_name, None)


def train_and_evaluate_model(model_name, model_config, device, logger, train_loader, valid_loader, 
                            test_loader, score_configs, metrics, config):
    """Train and evaluate a single model"""
    logger.info(f"=== Training {model_config['display_name']} ===")
    
    training_type = model_config.get('training_type', 'standard')
    
    if training_type == 'standard':
        # Standard autoencoder training
        encoder = model_config['encoder_class'](in_channels=3, latent_dim=config['latent_dim'])
        decoder = model_config['decoder_class'](out_channels=3, latent_dim=config['latent_dim'])
        model = model_config['model_class'](encoder, decoder).to(device)
        
        # Create trainer
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        loss_config = model_config['loss_config_fn']()
        trainer = Trainer(model, optimizer, loss_config, metrics, logger=logger)
        
        # Train model
        history = trainer.fit(train_loader, config['num_epochs'], valid_loader)
        
    elif training_type == 'statistical':
        # PaDiM training (statistical fitting)
        backbone = model_config.get('backbone', 'resnet18')
        layers = model_config.get('layers', ['layer2', 'layer3'])
        weights_key = model_config.get('weights_key', 'resnet18')
        weights_path = config.get('weights_paths', {}).get(weights_key, None)
        
        model = model_config['model_class'](backbone=backbone, layers=layers, weights_path=weights_path).to(device)
        
        # Special training for PaDiM
        padim_trainer = PaDiMTrainingWrapper(model)
        history = padim_trainer.train(train_loader)
        
    elif training_type == 'flow':
        # FastFlow training
        backbone = model_config.get('backbone', 'resnet18')
        layers = model_config.get('layers', ['layer2', 'layer3'])
        weights_key = model_config.get('weights_key', 'resnet18')
        weights_path = config.get('weights_paths', {}).get(weights_key, None)
        
        model = model_config['model_class'](backbone=backbone, layers=layers, weights_path=weights_path).to(device)
        
        # Create trainer
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        loss_config = model_config['loss_config_fn']()
        trainer = Trainer(model, optimizer, loss_config, metrics, logger=logger)
        
        # Train model
        history = trainer.fit(train_loader, config['num_epochs'], valid_loader)
        
    elif training_type == 'teacher_student':
        # STFPM training
        backbone = model_config.get('backbone', 'resnet18')
        layers = model_config.get('layers', ['layer1', 'layer2', 'layer3'])
        weights_key = model_config.get('weights_key', 'resnet18')
        weights_path = config.get('weights_paths', {}).get(weights_key, None)
        
        model = model_config['model_class'](backbone=backbone, layers=layers, weights_path=weights_path).to(device)
        
        # Create trainer
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        loss_config = model_config['loss_config_fn']()
        trainer = Trainer(model, optimizer, loss_config, metrics, logger=logger)
        
        # Train model
        history = trainer.fit(train_loader, config['num_epochs'], valid_loader)
        
    else:
        raise ValueError(f"Unsupported training type: {training_type}")
    
    # Evaluate model
    results, scores, test_labels = evaluate_model(model, test_loader, score_configs, logger)
    
    # Visualize reconstructions (only for reconstruction-based models)
    output_dir = Path(config['output_dir'])
    display_name = model_config['display_name'].replace(' ', '_')
    if training_type in ['standard']:  # Only for models that have reconstructions
        visualize_reconstructions(model, test_loader, output_dir, display_name, 
                                num_samples=8, show_plots=config['show_plots'])
    
    # Save model
    if config['save_models']:
        torch.save(model.state_dict(), output_dir / f'{model_name}_model.pth')
        logger.info(f"{model_config['display_name']} model saved")
    
    return results, history


def save_results(model_results, model_histories, config):
    """Save experimental results to files"""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    summary_data = []
    for model_name, results in model_results.items():
        for score_name, metrics in results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Score Type': score_name,
                'AUROC': metrics.get('auroc', 0),
                'AUPR': metrics.get('aupr', 0),
                'F1': metrics.get('f1', 0),
                'Accuracy': metrics.get('accuracy', 0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    
    # Save training histories
    for model_name, history in model_histories.items():
        with open(output_dir / f'{model_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save detailed results
    detailed_results = {
        'model_results': model_results,
        'config': config
    }
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    """Save experimental results to files"""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    summary_data = []
    for model_name, results in model_results.items():
        for score_name, metrics in results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Score Type': score_name,
                'AUROC': metrics.get('auroc', 0),
                'AUPR': metrics.get('aupr', 0),
                'F1': metrics.get('f1', 0),
                'Accuracy': metrics.get('accuracy', 0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    
    # Save training histories
    for model_name, history in model_histories.items():
        with open(output_dir / f'{model_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save detailed results
    detailed_results = {
        'model_results': model_results,
        'config': config
    }
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)


def main():
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Setup device
    if DEVICE == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    
    # Setup output directory and logging
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info(f"Starting MVTec anomaly detection evaluation")
    logger.info(f"Using device: {device}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Categories: {CATEGORIES}")
    logger.info(f"Models: {MODELS}")
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_mvtec_loaders(
        DATA_DIR, CATEGORIES, IMG_SIZE, BATCH_SIZE,
        VALID_RATIO, NUM_WORKERS, SEED
    )
    
    # Print dataset information
    train_info = get_dataset_info(train_loader)
    test_info = get_dataset_info(test_loader)
    logger.info(f"Train dataset: {train_info}")
    logger.info(f"Test dataset: {test_info}")
    
    # Create configurations
    score_configs = create_score_configs()
    metrics = create_standard_metrics()
    
    # Configuration dictionary for saving
    config = {
        'data_dir': DATA_DIR,
        'categories': CATEGORIES,
        'img_size': IMG_SIZE,
        'valid_ratio': VALID_RATIO,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'latent_dim': LATENT_DIM,
        'reconstruction_loss': RECONSTRUCTION_LOSS,
        'kl_weight': KL_WEIGHT,
        'weights_paths': WEIGHTS_PATHS,
        'num_workers': NUM_WORKERS,
        'device': str(device),
        'output_dir': OUTPUT_DIR,
        'save_models': SAVE_MODELS,
        'show_plots': SHOW_PLOTS,
        'seed': SEED,
        'models': MODELS
    }
    
    # Initialize result storage
    model_results = {}
    model_histories = {}
    
    # Train and evaluate models using for loop
    for model_name in MODELS:
        model_config = get_model_config(model_name, LATENT_DIM, RECONSTRUCTION_LOSS, KL_WEIGHT)
        
        if model_config is None:
            logger.warning(f"Unknown model type: {model_name}. Skipping...")
            continue
        
        # Train and evaluate model
        results, history = train_and_evaluate_model(
            model_name, model_config, device, logger, train_loader, valid_loader,
            test_loader, score_configs, metrics, config
        )
        
        # Store results
        model_results[model_name] = results
        model_histories[model_name] = history
    
    # Create visualizations and save results
    if model_results:
        # Plot training curves
        plot_training_curves(model_histories, output_dir, SHOW_PLOTS)
        
        # Plot performance comparison
        plot_performance_comparison(model_results, score_configs, output_dir, SHOW_PLOTS)
        
        # Save all results
        save_results(model_results, model_histories, config)
        
        # Print summary table
        summary_data = []
        for model_name, results in model_results.items():
            for score_name, metrics in results.items():
                summary_data.append({
                    'Model': model_name.upper(),
                    'Score Type': score_name,
                    'AUROC': metrics.get('auroc', 0),
                    'AUPR': metrics.get('aupr', 0),
                    'F1': metrics.get('f1', 0),
                    'Accuracy': metrics.get('accuracy', 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info("\n=== Performance Summary ===")
        logger.info(f"\n{summary_df.round(4).to_string(index=False)}")
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
