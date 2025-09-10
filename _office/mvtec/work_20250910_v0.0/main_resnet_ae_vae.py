import torchvision.models as models
from collections import OrderedDict

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from dataloader import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD, denormalize_imagenet
from models import ResNetAE, ResNetVAE, VAELoss, create_resnet_models


def test_resnet_models():
    """Complete test for ResNet-based AE and VAE models"""

    # Configuration
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
    CATEGORY = "grid"  # OLED-related category
    BATCH_SIZE = 8  # Smaller batch for ResNet models (more memory intensive)
    IMG_SIZE = 256
    LATENT_DIM = 512
    NUM_EPOCHS = 20  # Reduced for testing
    LEARNING_RATE = 1e-4  # Lower LR for pretrained features
    VAE_BETA = 0.005  # Smaller beta for ResNet VAE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 RESNET BACKBONE ANOMALY DETECTION TEST")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}/{CATEGORY}")
    print(f"Configuration: batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}")
    print("="*70)

    # Load data
    try:
        train_loader, test_loader = get_dataloaders(
            DATA_DIR, CATEGORY, BATCH_SIZE, IMG_SIZE, num_workers=4
        )
        print(f"✓ Data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Create ResNet models
    print("\n📦 Creating ResNet models...")
    models_dict = create_resnet_models(DEVICE)

    # Print model information
    print(f"\n📊 Model Information:")
    print("-" * 50)
    for name, model in models_dict.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<15}: Total={total_params:>9,}, Trainable={trainable_params:>9,}")

    # Test configurations for different models
    test_configs = {
        'ResNet18_AE': {
            'model_type': 'AE',
            'lr': LEARNING_RATE,
            'scheduler_patience': 10
        },
        'ResNet18_VAE': {
            'model_type': 'VAE',
            'lr': LEARNING_RATE * 0.5,
            'scheduler_patience': 15
        },
        'ResNet50_AE': {
            'model_type': 'AE',
            'lr': LEARNING_RATE * 0.8,
            'scheduler_patience': 10
        },
        'ResNet50_VAE': {
            'model_type': 'VAE',
            'lr': LEARNING_RATE * 0.3,
            'scheduler_patience': 20
        }
    }

    # Results storage
    all_results = {}
    all_training_losses = {}

    # Test each model
    for model_name in ['ResNet18_AE', 'ResNet18_VAE', 'ResNet50_AE', 'ResNet50_VAE']:
        if model_name not in models_dict:
            continue

        print(f"\n🔥 TRAINING {model_name}")
        print("="*50)

        model = models_dict[model_name]
        config = test_configs[model_name]

        # Setup optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=1e-5
        )

        if config['model_type'] == 'VAE':
            loss_fn = VAELoss(beta=VAE_BETA, capacity=25.0, gamma=1000.0)
        else:
            loss_fn = nn.MSELoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            patience=config['scheduler_patience'],
            factor=0.5, verbose=True
        )

        # Train model
        try:
            trained_model, losses = train_resnet_model(
                model, optimizer, loss_fn, train_loader, DEVICE,
                num_epochs=NUM_EPOCHS, scheduler=scheduler,
                model_type=config['model_type'], model_name=model_name
            )

            all_training_losses[model_name] = losses
            print(f"✓ {model_name} training completed")

        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            continue

        # Evaluate model
        print(f"\n📊 Evaluating {model_name}...")
        try:
            if config['model_type'] == 'VAE':
                # Debug VAE latent space
                latent_stats = debug_resnet_vae_latent_stats(trained_model, test_loader, DEVICE)

            scoring_results, best_method = compare_resnet_scoring_methods(
                trained_model, test_loader, DEVICE, config['model_type']
            )

            all_results[model_name] = (scoring_results, best_method)
            print(f"✓ {model_name} evaluation completed")

        except Exception as e:
            print(f"❌ {model_name} evaluation failed: {e}")
            continue

    # Compare all models
    if all_results:
        print(f"\n🏆 FINAL COMPARISON")
        print("="*70)
        compare_all_resnet_models(all_results, CATEGORY)

        # Plot training losses
        if all_training_losses:
            plot_resnet_training_losses(all_training_losses, CATEGORY)

        # Generate comparison plots
        generate_resnet_comparison_plots(all_results, CATEGORY)

        # Final summary
        print_resnet_final_summary(all_results, CATEGORY)

    print(f"\n✅ ResNet model testing completed!")


def train_resnet_model(model, optimizer, loss_fn, train_loader, device,
                      num_epochs=50, scheduler=None, model_type='AE', model_name='ResNet'):
    """Train ResNet-based model with monitoring"""

    model.train()
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kld_losses = []

    print(f"Starting {model_name} training...")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

        for batch_idx, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)

            optimizer.zero_grad()

            try:
                if model_type == 'VAE':
                    recon, mu, logvar, features = model(imgs)
                    loss, recon_loss, kld_loss = loss_fn(recon, imgs, mu, logvar)
                    epoch_recon_loss += recon_loss.item()
                    epoch_kld_loss += kld_loss.item()

                    pbar.set_postfix({
                        'Total': f'{loss.item():.6f}',
                        'Recon': f'{recon_loss.item():.6f}',
                        'KLD': f'{kld_loss.item():.6f}'
                    })

                else:  # AE
                    recon, latent, features = model(imgs)
                    loss = loss_fn(recon, imgs)
                    pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ GPU memory error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        if num_batches == 0:
            print(f"❌ No successful batches in epoch {epoch}")
            break

        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)

        if model_type == 'VAE':
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kld_loss = epoch_kld_loss / num_batches
            epoch_recon_losses.append(avg_recon_loss)
            epoch_kld_losses.append(avg_kld_loss)

            print(f"Epoch [{epoch}/{num_epochs}] Total: {avg_loss:.6f}, "
                  f"Recon: {avg_recon_loss:.6f}, KLD: {avg_kld_loss:.6f}")
        else:
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.6f}")

        if scheduler:
            scheduler.step(avg_loss)

        # FIXED: Monitor input/output ranges to verify proper normalization
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_imgs = imgs[:2]
                if model_type == 'VAE':
                    sample_recon, *_ = model(sample_imgs)
                else:
                    sample_recon, *_ = model(sample_imgs)

                print(f"  Input range: [{sample_imgs.min():.3f}, {sample_imgs.max():.3f}]")
                print(f"  Recon range: [{sample_recon.min():.3f}, {sample_recon.max():.3f}]")

        # Memory cleanup
        torch.cuda.empty_cache()

    if model_type == 'VAE':
        return model, (epoch_losses, epoch_recon_losses, epoch_kld_losses)
    else:
        return model, epoch_losses


def debug_resnet_vae_latent_stats(model, test_loader, device):
    """Debug ResNet VAE latent space"""

    model.eval()
    all_mu = []
    all_logvar = []
    all_labels = []

    print("Analyzing ResNet VAE latent space...")

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Collecting latent stats", leave=False):
            try:
                imgs = imgs.to(device)
                mu, logvar, _ = model.encode(imgs)

                all_mu.append(mu.cpu())
                all_logvar.append(logvar.cpu())
                all_labels.extend(labels.cpu().numpy())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    if not all_mu:
        print("⚠️ No latent statistics collected")
        return {}

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_labels = np.array(all_labels)

    # Compute statistics
    effective_dims = (all_logvar.exp().mean(0) > 0.01).sum()

    print(f"\nResNet VAE LATENT SPACE ANALYSIS:")
    print("="*40)
    print(f"Latent dimension: {all_mu.shape[1]}")
    print(f"Effective dimensions: {effective_dims}/{all_mu.shape[1]}")
    print(f"Mu range: [{all_mu.min():.4f}, {all_mu.max():.4f}]")
    print(f"LogVar range: [{all_logvar.min():.4f}, {all_logvar.max():.4f}]")

    if effective_dims < all_mu.shape[1] * 0.1:
        print("⚠️ WARNING: Possible posterior collapse!")
    elif effective_dims < all_mu.shape[1] * 0.5:
        print("△ CAUTION: Low effective dimensions")
    else:
        print("✓ GOOD: Healthy latent space")

    return {
        'mu': all_mu,
        'logvar': all_logvar,
        'labels': all_labels,
        'effective_dims': effective_dims
    }


def compare_resnet_scoring_methods(model, test_loader, device, model_type='AE'):
    """Compare scoring methods for ResNet models"""

    methods = ['max_pooling', 'percentile_95', 'top_k_mean']
    results = {}

    print(f"\n🔍 Testing scoring methods for ResNet {model_type}...")

    for method in methods:
        try:
            scores, labels = predict_resnet_anomaly_scores(
                model, test_loader, device, method, model_type
            )

            if len(scores) > 0 and len(np.unique(labels)) >= 2:
                auc = roc_auc_score(labels, scores)
                fpr, tpr, thresholds = roc_curve(labels, scores)
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                threshold = thresholds[best_idx]

                results[method] = {
                    'scores': scores,
                    'labels': labels,
                    'auc': auc,
                    'threshold': threshold
                }

                print(f"  {method}: AUC={auc:.4f}")
            else:
                print(f"  {method}: Failed (insufficient data)")

        except Exception as e:
            print(f"  {method}: Error - {e}")

    if results:
        best_method = max(results.keys(), key=lambda k: results[k]['auc'])
        print(f"🏆 Best method: {best_method} (AUC: {results[best_method]['auc']:.4f})")
        return results, best_method
    else:
        print("❌ No successful scoring methods")
        return {}, None


@torch.no_grad()
def predict_resnet_anomaly_scores(model, test_loader, device, score_method='max_pooling', model_type='AE'):
    """Predict anomaly scores for ResNet models"""

    model.eval()
    all_scores = []
    all_labels = []

    for imgs, labels in tqdm(test_loader, desc=f"Computing {score_method} scores", leave=False):
        try:
            imgs = imgs.to(device)

            if model_type == 'VAE':
                recon = model(imgs)  # In eval mode, returns only reconstruction
            else:
                recon = model(imgs)

            # Compute reconstruction error
            error_map = (imgs - recon) ** 2
            error_map = error_map.mean(dim=1)  # Average across channels

            # Apply scoring method
            if score_method == 'max_pooling':
                scores = torch.amax(error_map.view(error_map.size(0), -1), dim=1)
            elif score_method == 'percentile_95':
                flattened = error_map.view(error_map.size(0), -1)
                scores = torch.quantile(flattened, q=0.95, dim=1)
            elif score_method == 'top_k_mean':
                flattened = error_map.view(error_map.size(0), -1)
                k = max(1, int(0.01 * flattened.size(1)))
                top_k_values, _ = torch.topk(flattened, k=k, dim=1)
                scores = torch.mean(top_k_values, dim=1)
            else:
                scores = error_map.view(error_map.size(0), -1).mean(dim=1)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return np.array(all_scores), np.array(all_labels)


def compare_all_resnet_models(results_dict, category):
    """Compare all ResNet model results"""

    print("🏆 RESNET MODEL COMPARISON")
    print("="*60)

    model_summary = {}
    for model_name, (scoring_results, best_method) in results_dict.items():
        if best_method and best_method in scoring_results:
            best_auc = scoring_results[best_method]['auc']
            model_summary[model_name] = {
                'auc': best_auc,
                'method': best_method
            }

    # Sort by AUC
    sorted_models = sorted(model_summary.items(), key=lambda x: x[1]['auc'], reverse=True)

    print(f"Dataset: {category}")
    print(f"{'Rank':<4} {'Model':<15} {'AUC':<8} {'Best Method':<15}")
    print("-" * 50)

    for rank, (model_name, result) in enumerate(sorted_models, 1):
        print(f"{rank:<4} {model_name:<15} {result['auc']:<8.4f} {result['method']:<15}")

    if sorted_models:
        winner = sorted_models[0]
        print(f"\n🥇 WINNER: {winner[0]} (AUC: {winner[1]['auc']:.4f})")

        # Performance categories
        best_auc = winner[1]['auc']
        if best_auc > 0.95:
            print("🎉 OUTSTANDING performance!")
        elif best_auc > 0.90:
            print("🌟 EXCELLENT performance!")
        elif best_auc > 0.85:
            print("✅ VERY GOOD performance!")
        elif best_auc > 0.80:
            print("👍 GOOD performance!")
        else:
            print("⚠️ Needs improvement")


def plot_resnet_training_losses(losses_dict, category):
    """Plot training losses for ResNet models"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (model_name, losses) in enumerate(losses_dict.items()):
        if idx >= 4:
            break

        ax = axes[idx]

        if isinstance(losses, tuple):  # VAE losses
            total_losses, recon_losses, kld_losses = losses
            ax.plot(total_losses, label='Total', color='blue', linewidth=2)
            ax.plot(recon_losses, label='Recon', color='green', linewidth=1)
            ax.plot(kld_losses, label='KLD', color='red', linewidth=1)
            ax.legend()
        else:  # AE losses
            ax.plot(losses, color='blue', linewidth=2)

        ax.set_title(f"{model_name} Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(losses_dict), 4):
        axes[idx].set_visible(False)

    plt.suptitle(f"ResNet Models Training Losses - {category.title()}")
    plt.tight_layout()
    plt.show()


def generate_resnet_comparison_plots(results_dict, category):
    """Generate comparison plots for ResNet models"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ROC curves
    for model_name, (scoring_results, best_method) in results_dict.items():
        if best_method and best_method in scoring_results:
            scores = scoring_results[best_method]['scores']
            labels = scoring_results[best_method]['labels']

            if len(np.unique(labels)) >= 2:
                fpr, tpr, _ = roc_curve(labels, scores)
                auc = roc_auc_score(labels, scores)
                ax1.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curves - {category.title()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUC comparison bar chart
    model_names = []
    aucs = []

    for model_name, (scoring_results, best_method) in results_dict.items():
        if best_method and best_method in scoring_results:
            model_names.append(model_name)
            aucs.append(scoring_results[best_method]['auc'])

    colors = ['skyblue' if 'AE' in name else 'lightcoral' for name in model_names]
    bars = ax2.bar(range(len(model_names)), aucs, color=colors)

    ax2.set_xlabel('Models')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title(f'Model Performance Comparison - {category.title()}')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def print_resnet_final_summary(results_dict, category):
    """Print final summary of ResNet model performance"""

    print(f"\n📋 FINAL SUMMARY - {category.upper()}")
    print("="*70)

    # Count models by type
    ae_models = [name for name in results_dict.keys() if 'AE' in name and 'VAE' not in name]
    vae_models = [name for name in results_dict.keys() if 'VAE' in name]

    print(f"Tested Models: {len(results_dict)} total")
    print(f"  - Autoencoder models: {len(ae_models)}")
    print(f"  - VAE models: {len(vae_models)}")

    # Best performance by type
    best_ae_auc = 0
    best_vae_auc = 0
    best_ae_name = "None"
    best_vae_name = "None"

    for model_name, (scoring_results, best_method) in results_dict.items():
        if best_method and best_method in scoring_results:
            auc = scoring_results[best_method]['auc']

            if 'VAE' in model_name:
                if auc > best_vae_auc:
                    best_vae_auc = auc
                    best_vae_name = model_name
            else:
                if auc > best_ae_auc:
                    best_ae_auc = auc
                    best_ae_name = model_name

    print(f"\nBest Performance by Type:")
    print(f"  🔵 Best AE:  {best_ae_name} (AUC: {best_ae_auc:.4f})")
    print(f"  🔴 Best VAE: {best_vae_name} (AUC: {best_vae_auc:.4f})")

    # Overall best
    all_aucs = []
    for model_name, (scoring_results, best_method) in results_dict.items():
        if best_method and best_method in scoring_results:
            all_aucs.append((model_name, scoring_results[best_method]['auc']))

    if all_aucs:
        overall_best = max(all_aucs, key=lambda x: x[1])
        print(f"\n🏆 Overall Best: {overall_best[0]} (AUC: {overall_best[1]:.4f})")

        # Readiness assessment
        best_auc = overall_best[1]
        if best_auc > 0.90:
            print("\n✅ PRODUCTION READY!")
            print("   - Excellent anomaly detection performance")
            print("   - ResNet backbone provides robust features")
            print("   - Ready for OLED quality inspection deployment")
        elif best_auc > 0.85:
            print("\n🔄 OPTIMIZATION PHASE")
            print("   - Good baseline performance achieved")
            print("   - Consider hyperparameter tuning")
            print("   - Test with more OLED-specific data")
        else:
            print("\n⚠️ DEVELOPMENT PHASE")
            print("   - Needs architectural improvements")
            print("   - Consider ensemble methods")
            print("   - Collect more training data")

    print("="*70)

# Run the complete test
if __name__ == "__main__":
    test_resnet_models()
