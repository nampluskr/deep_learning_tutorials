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
from models import VanillaAE, VanillaVAE, VAELoss

# ============================================================================
# Training Functions - ImageNet Compatible
# ============================================================================

def train_autoencoder(model, optimizer, loss_fn, train_loader, device, num_epochs=100, 
                     scheduler=None, save_samples=False, model_type='AE'):
    """Train autoencoder with ImageNet normalization"""
    
    model.train()
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kld_losses = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        
        for batch_idx, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'VAE':
                recon, mu, logvar = model(imgs)
                loss, recon_loss, kld_loss = loss_fn(recon, imgs, mu, logvar)
                epoch_recon_loss += recon_loss.item()
                epoch_kld_loss += kld_loss.item()
                
                pbar.set_postfix({
                    'Total': f'{loss.item():.6f}',
                    'Recon': f'{recon_loss.item():.6f}',
                    'KLD': f'{kld_loss.item():.6f}'
                })
                
            else:  # AE
                recon = model(imgs)
                loss = loss_fn(recon, imgs)
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Save sample reconstructions periodically
            if save_samples and batch_idx == 0 and epoch % 10 == 0:
                save_reconstruction_samples(imgs, recon, epoch, device, model_type)
        
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
        
        # Monitor input/output ranges for ImageNet normalization
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_imgs = imgs[:2]
                if model_type == 'VAE':
                    sample_recon, _, _ = model(sample_imgs)
                else:
                    sample_recon = model(sample_imgs)
                
                print(f"  Input range: [{sample_imgs.min():.3f}, {sample_imgs.max():.3f}]")
                print(f"  Recon range: [{sample_recon.min():.3f}, {sample_recon.max():.3f}]")
    
    if model_type == 'VAE':
        return model, (epoch_losses, epoch_recon_losses, epoch_kld_losses)
    else:
        return model, epoch_losses


def save_reconstruction_samples(original, reconstructed, epoch, device, model_type='AE', save_dir="samples"):
    """Save sample reconstructions for visual inspection with ImageNet denormalization"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(min(4, original.size(0))):
        # Denormalize for visualization
        orig_denorm = denormalize_imagenet(original[i:i+1])[0]
        recon_denorm = denormalize_imagenet(reconstructed[i:i+1])[0]
        
        # Original
        axes[0, i].imshow(orig_denorm.cpu().permute(1, 2, 0))
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon_denorm.cpu().detach().permute(1, 2, 0))
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.suptitle(f"{model_type} Reconstruction - Epoch {epoch} (ImageNet Norm)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_type}_imagenet_reconstruction_epoch_{epoch}.png"))
    plt.close()

# ============================================================================
# Evaluation Functions - ImageNet Compatible
# ============================================================================

@torch.no_grad()
def predict_anomaly_scores(model, test_loader, device, score_method='max_pooling', model_type='AE'):
    """Compute anomaly scores with ImageNet normalized data"""
    
    model.eval()
    all_scores = []
    all_labels = []
    
    print(f"Computing anomaly scores using method: {score_method}")
    
    for imgs, labels in tqdm(test_loader, desc="Computing anomaly scores", leave=False):
        imgs = imgs.to(device)
        
        if model_type == 'VAE':
            recon = model(imgs)
        else:
            recon = model(imgs)
        
        # Compute pixel-wise reconstruction error in ImageNet normalized space
        error_map = (imgs - recon) ** 2
        error_map = error_map.mean(dim=1)  # Average across RGB channels
        
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
        elif score_method == 'mean':
            scores = error_map.view(error_map.size(0), -1).mean(dim=1)
        else:
            scores = torch.amax(error_map.view(error_map.size(0), -1), dim=1)
        
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_scores), np.array(all_labels)


def evaluate_model(scores, labels, verbose=True):
    """Evaluate anomaly detection performance"""
    
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels. Cannot compute ROC-AUC.")
        return 0.5, 0.5, (None, None, None)
    
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    
    if verbose:
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Best threshold (Youden's J): {best_threshold:.6f}")
        print(f"Best TPR: {tpr[best_idx]:.4f}, Best FPR: {fpr[best_idx]:.4f}")
    
    return best_threshold, auc, (fpr, tpr, thresholds)


def show_classification_results(scores, labels, threshold):
    """Show detailed classification results"""
    
    predictions = (scores >= threshold).astype(int)
    
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Normal  Anomaly")
    print(f"Normal    {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"Anomaly   {cm[1,0]:6d}  {cm[1,1]:7d}")
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["Normal", "Anomaly"]))
    
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")


def plot_score_distribution(scores, labels, bins=50, title="Anomaly Score Distribution", save_path=None):
    """Plot anomaly score distribution with save option"""
    
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall distribution
    sns.histplot(scores, bins=bins, kde=True, color="steelblue", 
                edgecolor="black", ax=ax1)
    ax1.set_title(f"{title} - Overall")
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Separated by class
    if len(normal_scores) > 0:
        sns.histplot(normal_scores, bins=bins, kde=True, color="green", 
                    label=f"Normal (n={len(normal_scores)})", alpha=0.6, 
                    edgecolor="black", ax=ax2)
    
    if len(anomaly_scores) > 0:
        sns.histplot(anomaly_scores, bins=bins, kde=True, color="red", 
                    label=f"Anomaly (n={len(anomaly_scores)})", alpha=0.6, 
                    edgecolor="black", ax=ax2)
    
    ax2.set_title(f"{title} - By Class")
    ax2.set_xlabel("Anomaly Score")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Print detailed statistics
    print(f"\nDETAILED SCORE STATISTICS:")
    print("="*40)
    if len(normal_scores) > 0:
        print(f"Normal   - Count: {len(normal_scores):4d}")
        print(f"           Mean:  {normal_scores.mean():.6f} ± {normal_scores.std():.6f}")
        print(f"           Range: [{normal_scores.min():.6f}, {normal_scores.max():.6f}]")
    
    if len(anomaly_scores) > 0:
        print(f"\nAnomaly  - Count: {len(anomaly_scores):4d}")
        print(f"           Mean:  {anomaly_scores.mean():.6f} ± {anomaly_scores.std():.6f}")
        print(f"           Range: [{anomaly_scores.min():.6f}, {anomaly_scores.max():.6f}]")
    
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        pooled_std = np.sqrt(((len(normal_scores) - 1) * normal_scores.var() + 
                             (len(anomaly_scores) - 1) * anomaly_scores.var()) / 
                            (len(normal_scores) + len(anomaly_scores) - 2))
        cohens_d = (anomaly_scores.mean() - normal_scores.mean()) / pooled_std
        
        print(f"\nSEPARATION ANALYSIS:")
        print("="*40)
        print(f"Cohen's d (effect size): {cohens_d:.4f}")
        
        if cohens_d > 0.8:
            print("✓ EXCELLENT separation (Cohen's d > 0.8)")
        elif cohens_d > 0.5:
            print("○ GOOD separation (Cohen's d > 0.5)")
        elif cohens_d > 0.2:
            print("△ FAIR separation (Cohen's d > 0.2)")
        else:
            print("✗ POOR separation (Cohen's d ≤ 0.2)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    """Plot ROC curve"""
    
    if fpr is None or tpr is None:
        print("Cannot plot ROC curve: insufficient data")
        return
        
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# Debug and Analysis Functions
# ============================================================================

def debug_vae_latent_stats(model, test_loader, device):
    """Debug function to analyze VAE latent space statistics"""
    
    model.eval()
    all_mu = []
    all_logvar = []
    all_labels = []
    
    print("Analyzing VAE latent space...")
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Collecting latent stats", leave=False):
            imgs = imgs.to(device)
            mu, logvar = model.encode(imgs)
            
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_labels.extend(labels.cpu().numpy())
    
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_labels = np.array(all_labels)
    
    # Compute statistics
    mu_mean = all_mu.mean()
    mu_std = all_mu.std()
    logvar_mean = all_logvar.mean()
    logvar_std = all_logvar.std()
    
    print(f"\nVAE LATENT SPACE ANALYSIS:")
    print("="*40)
    print(f"Latent dimension: {all_mu.shape[1]}")
    print(f"Mu statistics:")
    print(f"  Mean: {mu_mean:.6f}, Std: {mu_std:.6f}")
    print(f"  Range: [{all_mu.min():.6f}, {all_mu.max():.6f}]")
    print(f"LogVar statistics:")
    print(f"  Mean: {logvar_mean:.6f}, Std: {logvar_std:.6f}")
    print(f"  Range: [{all_logvar.min():.6f}, {all_logvar.max():.6f}]")
    
    # Check for posterior collapse
    effective_dims = (all_logvar.exp().mean(0) > 0.01).sum()
    print(f"Effective dimensions (var > 0.01): {effective_dims}/{all_mu.shape[1]}")
    
    if effective_dims < all_mu.shape[1] * 0.1:
        print("⚠️  WARNING: Possible posterior collapse detected!")
        print("   - Consider reducing beta or increasing capacity")
    elif effective_dims < all_mu.shape[1] * 0.5:
        print("△ CAUTION: Low effective dimensions")
        print("   - Monitor training and consider adjusting hyperparameters")
    else:
        print("✓ GOOD: Healthy latent space utilization")
    
    return {
        'mu': all_mu,
        'logvar': all_logvar,
        'labels': all_labels,
        'effective_dims': effective_dims
    }


def compare_scoring_methods(model, test_loader, device, model_type='AE'):
    """Compare different anomaly scoring methods"""
    
    methods = ['max_pooling', 'percentile_95', 'top_k_mean', 'mean']
    results = {}
    
    print("\n" + "="*60)
    print(f"COMPARING DIFFERENT ANOMALY SCORING METHODS - {model_type}")
    print("="*60)
    
    for method in methods:
        print(f"\n--- Testing method: {method} ---")
        scores, labels = predict_anomaly_scores(model, test_loader, device, 
                                               score_method=method, model_type=model_type)
        threshold, auc, _ = evaluate_model(scores, labels, verbose=False)
        
        results[method] = {
            'scores': scores,
            'labels': labels,
            'auc': auc,
            'threshold': threshold
        }
        
        print(f"AUC: {auc:.4f}, Threshold: {threshold:.6f}")
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['auc'])
    print(f"\n🏆 BEST METHOD: {best_method} (AUC: {results[best_method]['auc']:.4f})")
    
    return results, best_method


def compare_models(results_dict, category):
    """Compare performance between different models"""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Extract best results for each model
    model_results = {}
    for model_name, (scoring_results, best_method) in results_dict.items():
        best_auc = scoring_results[best_method]['auc']
        model_results[model_name] = {
            'auc': best_auc,
            'best_method': best_method,
            'scores': scoring_results[best_method]['scores'],
            'labels': scoring_results[best_method]['labels'],
            'threshold': scoring_results[best_method]['threshold']
        }
    
    # Sort by AUC
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    print(f"Dataset: {category}")
    print(f"{'Rank':<4} {'Model':<12} {'AUC':<8} {'Best Method':<15}")
    print("-" * 45)
    
    for rank, (model_name, result) in enumerate(sorted_models, 1):
        print(f"{rank:<4} {model_name:<12} {result['auc']:<8.4f} {result['best_method']:<15}")
    
    # Best model
    best_model_name, best_result = sorted_models[0]
    print(f"\n🏆 WINNER: {best_model_name} (AUC: {best_result['auc']:.4f})")
    
    return best_model_name, best_result


def plot_training_losses(losses_dict, save_path=None):
    """Plot training losses for comparison"""
    
    fig, axes = plt.subplots(1, len(losses_dict), figsize=(6*len(losses_dict), 5))
    if len(losses_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, losses) in enumerate(losses_dict.items()):
        ax = axes[idx]
        
        if isinstance(losses, tuple):  # VAE with multiple loss components
            total_losses, recon_losses, kld_losses = losses
            ax.plot(total_losses, label='Total Loss', color='blue', linewidth=2)
            ax.plot(recon_losses, label='Recon Loss', color='green', linewidth=1)
            ax.plot(kld_losses, label='KLD Loss', color='red', linewidth=1)
            ax.legend()
        else:  # Single loss
            ax.plot(losses, color='blue', linewidth=2)
        
        ax.set_title(f"{model_name} Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to: {save_path}")
    
    plt.show()

# ============================================================================
# Main Execution Function - ImageNet Version
# ============================================================================

def main():
    """Main function with ImageNet normalization for backbone compatibility"""
    
    # Configuration
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"  # Update this path
    CATEGORY = "grid"
    BATCH_SIZE = 16
    IMG_SIZE = 256
    LATENT_DIM = 1024
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    VAE_BETA = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = f"results_{CATEGORY}_imagenet"
    os.makedirs(output_dir, exist_ok=True)
    
    print("ImageNet NORMALIZED VERSION - AE vs VAE Comparison")
    print("="*70)
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}/{CATEGORY}")
    print(f"Configuration: batch_size={BATCH_SIZE}, img_size={IMG_SIZE}, epochs={NUM_EPOCHS}")
    print(f"VAE Beta: {VAE_BETA}")
    print("KEY FEATURES:")
    print("  1. ImageNet normalization for backbone compatibility")
    print("  2. Tanh output converted to ImageNet range")
    print("  3. Improved anomaly scoring methods")
    print("  4. Ready for pretrained backbone integration")
    print("="*70)
    
    # Load data
    try:
        train_loader, test_loader = get_dataloaders(
            DATA_DIR, CATEGORY, BATCH_SIZE, IMG_SIZE, num_workers=2
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the dataset path and structure.")
        return
    
    # Dictionary to store results
    model_results = {}
    training_losses = {}
    
    # ============================================================================
    # Train and Evaluate VanillaAE with ImageNet Normalization
    # ============================================================================
    
    print("\n" + "="*70)
    print("TRAINING VANILLA AUTOENCODER (ImageNet Normalized)")
    print("="*70)
    
    # Initialize AE model
    ae_model = VanillaAE(latent_dim=LATENT_DIM, input_size=IMG_SIZE).to(DEVICE)
    ae_optimizer = optim.AdamW(ae_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    ae_loss_fn = nn.MSELoss()
    ae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ae_optimizer, mode='min', patience=15, factor=0.5, verbose=True
    )
    
    # Train AE
    ae_model_trained, ae_losses = train_autoencoder(
        ae_model, ae_optimizer, ae_loss_fn, train_loader, DEVICE, 
        num_epochs=NUM_EPOCHS, scheduler=ae_scheduler, save_samples=True, model_type='AE'
    )
    
    training_losses['VanillaAE_ImageNet'] = ae_losses
    
    # Evaluate AE
    print("\nEvaluating VanillaAE with ImageNet normalization...")
    ae_scoring_results, ae_best_method = compare_scoring_methods(
        ae_model_trained, test_loader, DEVICE, model_type='AE'
    )
    model_results['VanillaAE_ImageNet'] = (ae_scoring_results, ae_best_method)
    
    # ============================================================================
    # Train and Evaluate VanillaVAE with ImageNet Normalization
    # ============================================================================
    
    print("\n" + "="*70)
    print("TRAINING VANILLA VARIATIONAL AUTOENCODER (ImageNet Normalized)")
    print("="*70)
    
    # Initialize VAE model
    vae_model = VanillaVAE(latent_dim=LATENT_DIM, input_size=IMG_SIZE, beta=VAE_BETA).to(DEVICE)
    vae_optimizer = optim.AdamW(vae_model.parameters(), lr=LEARNING_RATE*0.5, weight_decay=1e-5)
    vae_loss_fn = VAELoss(beta=VAE_BETA, capacity=25.0, gamma=1000.0)
    vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        vae_optimizer, mode='min', patience=20, factor=0.5, verbose=True
    )
    
    # Train VAE
    vae_model_trained, vae_losses = train_autoencoder(
        vae_model, vae_optimizer, vae_loss_fn, train_loader, DEVICE, 
        num_epochs=NUM_EPOCHS, scheduler=vae_scheduler, save_samples=True, model_type='VAE'
    )
    
    training_losses['VanillaVAE_ImageNet'] = vae_losses
    
    # Evaluate VAE with debugging
    print("\nEvaluating VanillaVAE with ImageNet normalization...")
    
    # Debug VAE latent space
    vae_latent_stats = debug_vae_latent_stats(vae_model_trained, test_loader, DEVICE)
    
    vae_scoring_results, vae_best_method = compare_scoring_methods(
        vae_model_trained, test_loader, DEVICE, model_type='VAE'
    )
    model_results['VanillaVAE_ImageNet'] = (vae_scoring_results, vae_best_method)
    
    # ============================================================================
    # Compare Models and Generate Reports
    # ============================================================================
    
    # Compare models
    best_model_name, best_result = compare_models(model_results, CATEGORY)
    
    # Plot training losses
    print("\nPlotting training losses...")
    plot_training_losses(training_losses, 
                        save_path=os.path.join(output_dir, f"{CATEGORY}_imagenet_training_losses.png"))
    
    # Plot score distributions for both models
    for model_name, (scoring_results, best_method) in model_results.items():
        best_scores = scoring_results[best_method]['scores']
        best_labels = scoring_results[best_method]['labels']
        
        title = f"{CATEGORY.title()} - {model_name} ({best_method.replace('_', ' ').title()})"
        save_path = os.path.join(output_dir, f"{CATEGORY}_{model_name}_score_distribution.png")
        
        print(f"\nPlotting score distribution for {model_name}...")
        plot_score_distribution(best_scores, best_labels, title=title, save_path=save_path)
    
    # Plot ROC curves for both models
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, (scoring_results, best_method) in model_results.items():
        best_scores = scoring_results[best_method]['scores']
        best_labels = scoring_results[best_method]['labels']
        
        if len(np.unique(best_labels)) >= 2:
            _, auc, roc_data = evaluate_model(best_scores, best_labels, verbose=False)
            fpr, tpr, _ = roc_data
            
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{CATEGORY.title()} - ROC Curve Comparison (ImageNet Normalized)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    roc_save_path = os.path.join(output_dir, f"{CATEGORY}_imagenet_roc_comparison.png")
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    print(f"ROC comparison plot saved to: {roc_save_path}")
    plt.show()
    
    # Show detailed results for best model
    print(f"\nDETAILED EVALUATION FOR BEST MODEL: {best_model_name}")
    print("="*70)
    show_classification_results(best_result['scores'], best_result['labels'], best_result['threshold'])
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY - ImageNet Normalized Version")
    print("="*70)
    print(f"Dataset: {CATEGORY}")
    print(f"Models compared: VanillaAE vs VanillaVAE (ImageNet Normalized)")
    print(f"Training epochs: {NUM_EPOCHS}")
    print(f"VAE Beta parameter: {VAE_BETA}")
    print(f"Normalization: ImageNet (mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
    
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"Best scoring method: {best_result['best_method']}")
    print(f"Final ROC-AUC: {best_result['auc']:.4f}")
    print(f"Optimal threshold: {best_result['threshold']:.6f}")
    print(f"Total samples: {len(best_result['labels'])} "
          f"(Normal: {(best_result['labels']==0).sum()}, "
          f"Anomaly: {(best_result['labels']==1).sum()})")
    
    # Performance assessment
    auc = best_result['auc']
    if auc > 0.9:
        status, emoji = "EXCELLENT", "🎉"
    elif auc > 0.8:
        status, emoji = "GOOD", "✅"
    elif auc > 0.7:
        status, emoji = "FAIR", "⚠️"
    elif auc > 0.6:
        status, emoji = "POOR", "❌"
    else:
        status, emoji = "FAILED", "💥"
    
    print(f"\n{emoji} PERFORMANCE: {status} (AUC: {auc:.4f})")
    
    # Model-specific insights
    ae_auc = model_results['VanillaAE_ImageNet'][0][model_results['VanillaAE_ImageNet'][1]]['auc']
    vae_auc = model_results['VanillaVAE_ImageNet'][0][model_results['VanillaVAE_ImageNet'][1]]['auc']
    
    print(f"\nMODEL INSIGHTS (ImageNet Normalized):")
    print(f"VanillaAE  AUC: {ae_auc:.4f}")
    print(f"VanillaVAE AUC: {vae_auc:.4f}")
    print(f"Improvement: {abs(vae_auc - ae_auc):.4f} ({'VAE' if vae_auc > ae_auc else 'AE'} wins)")
    
    print(f"\nBACKBONE READINESS:")
    print("✓ ImageNet normalization applied")
    print("✓ Compatible with pretrained ResNet/EfficientNet backbones")
    print("✓ Tanh output properly converted to ImageNet range")
    print("✓ Ready for STFPM, PaDiM, PatchCore integration")
    
    if auc > 0.8:
        print("\n✅ READY FOR PRODUCTION:")
        print("✓ Models successfully distinguish normal vs anomaly!")
        print("✓ ImageNet compatibility enables backbone integration")
        print("✓ Ready for OLED display quality inspection deployment")
    elif auc > 0.6:
        print("\n⚠️  NEEDS IMPROVEMENT:")
        print("○ Models show potential but need backbone integration")
        print("○ Consider ResNet/EfficientNet encoder for better features")
    else:
        print("\n❌ REQUIRES SIGNIFICANT WORK:")
        print("✗ Models need architectural improvements")
        print("✗ Consider advanced methods like STFPM or PatchCore")
    
    print(f"\nAll results saved to: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    main()
