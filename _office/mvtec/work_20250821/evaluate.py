import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score


@torch.no_grad()
def compute_anomaly_scores(model, test_loader, score_configs):
    """Compute anomaly scores for test dataset"""
    device = next(model.parameters()).device
    model.eval()
    
    all_scores = {name: [] for name in score_configs.keys()}
    all_labels = []
    
    for data in test_loader:
        images = data['image'].to(device)
        labels = data['label']
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images)
        
        # Compute different types of anomaly scores
        for score_name, config in score_configs.items():
            score_type = config['type']
            
            if score_type == 'reconstruction':
                if 'reconstructed' in outputs and 'input' in outputs:
                    errors = (outputs['input'] - outputs['reconstructed']) ** 2
                    if config.get('reduction', 'mean') == 'mean':
                        scores = torch.mean(errors, dim=[1, 2, 3])
                    else:
                        scores = torch.sum(errors, dim=[1, 2, 3])
                else:
                    scores = torch.zeros(batch_size, device=device)
                    
            elif score_type == 'ssim_based':
                if 'reconstructed' in outputs and 'input' in outputs:
                    from pytorch_msssim import ssim
                    # Compute SSIM per image
                    scores = []
                    for i in range(batch_size):
                        img_ssim = ssim(outputs['input'][i:i+1], outputs['reconstructed'][i:i+1], data_range=1.0)
                        scores.append(1 - img_ssim)
                    scores = torch.stack(scores).squeeze()
                    if scores.dim() == 0:  # Handle single item case
                        scores = scores.unsqueeze(0)
                else:
                    scores = torch.zeros(batch_size, device=device)
                    
            elif score_type == 'l1_reconstruction':
                if 'reconstructed' in outputs and 'input' in outputs:
                    errors = torch.abs(outputs['input'] - outputs['reconstructed'])
                    scores = torch.mean(errors, dim=[1, 2, 3])
                else:
                    scores = torch.zeros(batch_size, device=device)
                    
            elif score_type == 'latent_magnitude':
                if 'latent' in outputs:
                    scores = torch.mean(torch.abs(outputs['latent']), dim=1)
                elif 'z' in outputs:
                    scores = torch.mean(torch.abs(outputs['z']), dim=1)
                else:
                    scores = torch.zeros(batch_size, device=device)
                    
            elif score_type == 'padim_mahalanobis':
                # PaDiM specific scoring
                if 'anomaly_scores' in outputs:
                    # Pre-computed scores from forward pass
                    scores = outputs['anomaly_scores']
                elif hasattr(model, 'compute_anomaly_scores') and 'patch_embeddings' in outputs:
                    try:
                        scores = model.compute_anomaly_scores(outputs['patch_embeddings'], outputs['shape_info'])
                    except Exception as e:
                        print(f"Warning: PaDiM scoring failed: {e}")
                        scores = torch.zeros(batch_size, device=device)
                elif hasattr(model, 'mean_vectors') and model.mean_vectors is not None:
                    # PaDiM is fitted, compute scores from current forward pass
                    try:
                        patch_embeddings, shape_info = model._extract_patch_embeddings(
                            model.feature_extractor(images))
                        scores = model.compute_anomaly_scores(patch_embeddings, shape_info)
                    except Exception as e:
                        print(f"Warning: PaDiM scoring failed: {e}")
                        scores = torch.zeros(batch_size, device=device)
                else:
                    print("Warning: PaDiM model not fitted or no patch embeddings available")
                    scores = torch.zeros(batch_size, device=device)
                    
            elif score_type == 'fastflow_likelihood':
                # FastFlow specific scoring
                if hasattr(model, 'compute_anomaly_scores'):
                    scores = model.compute_anomaly_scores(outputs)
                else:
                    scores = torch.zeros(batch_size, device=device)
                    
            elif score_type == 'stfpm_feature_diff':
                # STFPM specific scoring
                if hasattr(model, 'compute_anomaly_scores'):
                    scores, _ = model.compute_anomaly_scores(outputs)
                else:
                    scores = torch.zeros(batch_size, device=device)
                    
            else:
                scores = torch.zeros(batch_size, device=device)
            
            # Ensure scores is 1D tensor
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            elif scores.dim() > 1:
                scores = scores.flatten()
            
            # Convert to numpy and extend
            scores_np = scores.cpu().numpy()
            if scores_np.ndim == 0:  # Handle 0-d array
                scores_np = np.array([scores_np.item()])
            
            all_scores[score_name].extend(scores_np.tolist())
        
        # Handle labels
        labels_np = labels.numpy()
        if labels_np.ndim == 0:  # Handle 0-d array
            labels_np = np.array([labels_np.item()])
        
        all_labels.extend(labels_np.tolist())
    
    return all_scores, all_labels


def evaluate_anomaly_detection(scores, labels, score_name=""):
    """Evaluate anomaly detection performance"""
    try:
        # Convert to numpy arrays
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Compute AUROC and AUPR
        auroc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute metrics at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        results = {
            'auroc': auroc,
            'aupr': aupr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'optimal_threshold': optimal_threshold
        }
        
        if score_name:
            print(f"\n=== {score_name} Results ===")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPR: {aupr:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {score_name}: {e}")
        return {}


def create_score_configs():
    """Create standard anomaly score configurations"""
    return {
        'mse_recon': {'type': 'reconstruction', 'reduction': 'mean'},
        'l1_recon': {'type': 'l1_reconstruction'},
        'ssim_score': {'type': 'ssim_based'},
        'latent_mag': {'type': 'latent_magnitude'},
        'padim_score': {'type': 'padim_mahalanobis'},
        'fastflow_score': {'type': 'fastflow_likelihood'},
        'stfpm_score': {'type': 'stfpm_feature_diff'}
    }


def evaluate_model(model, test_loader, score_configs, logger):
    """Evaluate model on test set with multiple anomaly scores"""
    logger.info(f"Evaluating {model.model_type} model on test set")
    
    # Filter score configs based on model type
    model_type = getattr(model, 'model_type', 'unknown')
    filtered_score_configs = {}
    
    if model_type in ['vanilla_ae', 'vae']:
        # Reconstruction-based models
        filtered_score_configs = {k: v for k, v in score_configs.items() 
                                if v['type'] in ['reconstruction', 'l1_reconstruction', 'ssim_based', 'latent_magnitude']}
    elif model_type == 'padim':
        # PaDiM specific scores
        filtered_score_configs = {k: v for k, v in score_configs.items() 
                                if v['type'] in ['padim_mahalanobis']}
        # Ensure at least one score config exists for PaDiM
        if not filtered_score_configs:
            filtered_score_configs = {'padim_score': {'type': 'padim_mahalanobis'}}
    elif model_type == 'fastflow':
        # FastFlow specific scores
        filtered_score_configs = {k: v for k, v in score_configs.items() 
                                if v['type'] in ['fastflow_likelihood']}
        # Ensure at least one score config exists for FastFlow
        if not filtered_score_configs:
            filtered_score_configs = {'fastflow_score': {'type': 'fastflow_likelihood'}}
    elif model_type == 'stfpm':
        # STFPM specific scores
        filtered_score_configs = {k: v for k, v in score_configs.items() 
                                if v['type'] in ['stfpm_feature_diff']}
        # Ensure at least one score config exists for STFPM
        if not filtered_score_configs:
            filtered_score_configs = {'stfpm_score': {'type': 'stfpm_feature_diff'}}
    else:
        # Use all available scores
        filtered_score_configs = score_configs
    
    # Compute anomaly scores
    scores, test_labels = compute_anomaly_scores(model, test_loader, filtered_score_configs)
    
    logger.info(f"Test dataset: {sum(1 for l in test_labels if l == 0)} normal, {sum(1 for l in test_labels if l == 1)} anomaly samples")
    
    # Evaluate each score type
    results = {}
    for score_name in filtered_score_configs.keys():
        results[score_name] = evaluate_anomaly_detection(
            scores[score_name], test_labels, f"{model.model_type.upper()} - {score_name}")
    
    return results, scores, test_labels


def plot_training_curves(model_histories, output_dir, show_plots=False):
    """Plot training curves for all models"""
    if len(model_histories) == 0:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Total loss
    for i, (model_name, history) in enumerate(model_histories.items()):
        color = colors[i % len(colors)]
        if 'total' in history:
            axes[0].plot(history['total'], label=f'{model_name.upper()} Train', color=color)
            if 'val_total' in history:
                axes[0].plot(history['val_total'], label=f'{model_name.upper()} Val', 
                           color=color, linestyle='--')
    
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # SSIM
    for i, (model_name, history) in enumerate(model_histories.items()):
        color = colors[i % len(colors)]
        if 'ssim' in history:
            axes[1].plot(history['ssim'], label=f'{model_name.upper()} Train', color=color)
            if 'val_ssim' in history:
                axes[1].plot(history['val_ssim'], label=f'{model_name.upper()} Val', 
                           color=color, linestyle='--')
    
    axes[1].set_title('SSIM')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SSIM')
    axes[1].legend()
    axes[1].grid(True)
    
    # PSNR
    for i, (model_name, history) in enumerate(model_histories.items()):
        color = colors[i % len(colors)]
        if 'psnr' in history:
            axes[2].plot(history['psnr'], label=f'{model_name.upper()} Train', color=color)
            if 'val_psnr' in history:
                axes[2].plot(history['val_psnr'], label=f'{model_name.upper()} Val', 
                           color=color, linestyle='--')
    
    axes[2].set_title('PSNR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PSNR')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def plot_performance_comparison(model_results, score_configs, output_dir, show_plots=False):
    """Plot performance comparison across models and score types"""
    if len(model_results) == 0:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get all unique score names from all models
    all_score_names = set()
    for model_results_dict in model_results.values():
        all_score_names.update(model_results_dict.keys())
    
    score_names = sorted(list(all_score_names))
    model_names = list(model_results.keys())
    
    # AUROC comparison
    x = np.arange(len(score_names))
    width = 0.35 if len(model_names) == 2 else 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        aurocs = []
        for score_name in score_names:
            if score_name in model_results[model_name]:
                aurocs.append(model_results[model_name][score_name].get('auroc', 0))
            else:
                aurocs.append(0)  # Missing score for this model
        
        offset = (i - len(model_names)/2 + 0.5) * width
        axes[0].bar(x + offset, aurocs, width, label=model_name.upper(), alpha=0.7)
    
    axes[0].set_title('AUROC Comparison')
    axes[0].set_xlabel('Score Type')
    axes[0].set_ylabel('AUROC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(score_names, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1-Score comparison
    for i, model_name in enumerate(model_names):
        f1s = []
        for score_name in score_names:
            if score_name in model_results[model_name]:
                f1s.append(model_results[model_name][score_name].get('f1', 0))
            else:
                f1s.append(0)  # Missing score for this model
        
        offset = (i - len(model_names)/2 + 0.5) * width
        axes[1].bar(x + offset, f1s, width, label=model_name.upper(), alpha=0.7)
    
    axes[1].set_title('F1-Score Comparison')
    axes[1].set_xlabel('Score Type')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(score_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def visualize_reconstructions(model, test_loader, output_dir, model_name, num_samples=8, show_plots=False):
    """Visualize reconstructions"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data in test_loader:
            images = data['image'][:num_samples].to(device)
            labels = data['label'][:num_samples]
            
            outputs = model(images)
            
            fig, axes = plt.subplots(3, num_samples, figsize=(16, 8))
            for i in range(num_samples):
                # Original
                orig_img = images[i].cpu().permute(1, 2, 0)
                axes[0, i].imshow(orig_img)
                axes[0, i].set_title(f"Original\nLabel: {labels[i].item()}")
                axes[0, i].axis('off')
                
                # Reconstructed
                recon_img = outputs['reconstructed'][i].cpu().permute(1, 2, 0)
                axes[1, i].imshow(recon_img)
                axes[1, i].set_title("Reconstructed")
                axes[1, i].axis('off')
                
                # Difference
                diff_img = torch.abs(images[i] - outputs['reconstructed'][i]).cpu().permute(1, 2, 0)
                axes[2, i].imshow(diff_img, cmap='hot')
                axes[2, i].set_title("Difference")
                axes[2, i].axis('off')
            
            plt.suptitle(f'{model_name} Reconstructions')
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name.lower()}_reconstructions.png', dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close()
            break
