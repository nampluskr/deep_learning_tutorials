import os
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders import get_dataloaders
from model_ae_v2 import VanillaAEV1, ResNetAEV1, VanillaAEV2, VanillaAEV3, UNetAE
# from model_ae import VanillaAE, UNetAE
from utils import show_history, show_roc_curve, show_distribution, show_evaluation, show_statistics


def get_config():
    config = SimpleNamespace(
        # data_dir="/mnt/d/datasets/mvtec",   # WSL
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category="grid",
        model_type="none",
        batch_size=8,
        img_size=512,
        latent_dim=1024,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    config.output_dir = f"results_{config.category}"
    os.makedirs(config.output_dir, exist_ok=True)
    return config


def denormalize_imagenet(tensor):
    """ imagenet range [-2.6, 2.6] to [0, 1] """
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def save_reconstruction_samples(original, reconstructed, epoch, model_type='ae', save_dir="samples"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))

    for i in range(min(4, original.size(0))):
        # Denormalize for visualization
        original_denorm = denormalize_imagenet(original[i:i+1])[0]
        reconstructed_denorm = denormalize_imagenet(reconstructed[i:i+1])[0]
        anomaly_map = torch.mean((original_denorm - reconstructed_denorm)**2, dim=0)

        # Original
        axes[0, i].imshow(original_denorm.cpu().permute(1, 2, 0))
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        # Reconstructed
        axes[1, i].imshow(reconstructed_denorm.cpu().detach().permute(1, 2, 0))
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')

        # Anomaly Map
        axes[2, i].imshow(anomaly_map.cpu().detach(), cmap="jet", vmin=0.0, vmax=1.0)
        axes[2, i].set_title(f"Anomaly Map {i+1}")
        axes[2, i].axis('off')

    plt.suptitle(f"{model_type} Reconstruction - Epoch {epoch} (ImageNet Norm)")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_type}_epoch_{epoch}.png"))
    plt.close()


def compute_anomaly_scores(anomaly_map, method='max_pooling'):
    if method == 'max_pooling':
        scores = torch.amax(anomaly_map.view(anomaly_map.size(0), -1), dim=1)
    elif method == 'percentile_95':
        flattened = anomaly_map.view(anomaly_map.size(0), -1)
        scores = torch.quantile(flattened, q=0.95, dim=1)
    elif method == 'topk_mean':
        flattened = anomaly_map.view(anomaly_map.size(0), -1)
        k = max(1, int(0.01 * flattened.size(1)))
        topk_values, _ = torch.topk(flattened, k=k, dim=1)
        scores = torch.mean(topk_values, dim=1)
    elif method == 'mean':
        scores = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
    else:
        scores = torch.amax(anomaly_map.view(anomaly_map.size(0), -1), dim=1)
    return scores

def compute_anomaly_map(original, reconstructed):
    anomaly_map = torch.mean((original - reconstructed)**2, dim=1)
    return anomaly_map

@torch.no_grad()
def trainer_predict(model, test_loader, device, method='max_pooling'):
    model.eval()
    all_scores = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        reconstructed = model(images)

        # anomaly_map = predictions['anomaly_map']
        # scores = predictions['pred_score']

        anomaly_map = compute_anomaly_map(images, reconstructed)
        scores = compute_anomaly_scores(anomaly_map)

        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return np.array(all_scores), np.array(all_labels)


def tanh_to_imagenet(tanh_output):
    """ tanh output in [-1, 1] to imagenet range in [-2.3, 2.3] """
    scaled = (tanh_output + 1.0) / 2.0   # in [0, 1]
    device = tanh_output.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (scaled - mean) / std


def trainer_fit(model, optimizer, loss_fn, train_loader, device, num_epochs=10,
                model_type="ae", save_dir="samples"):
    model.train()
    history = {"loss": []}

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=False, ascii=True) as pbar:
            for images, _ in pbar:
                images = images.to(device)

                optimizer.zero_grad()
                reconstructed, *_ = model(images)
                # reconstructed = tanh_to_imagenet(reconstructed)
                loss = loss_fn(reconstructed, images)
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["loss"].append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] loss: {avg_loss:.2e}")
            save_reconstruction_samples(images, reconstructed, epoch, model_type=model_type, save_dir=save_dir)

            with torch.no_grad():
                sample_images = images[:2]
                sample_reconstructed, *_ = model(sample_images)
                # sample_reconstructed = tanh_to_imagenet(sample_reconstructed)
                print(f"  Input:  [{sample_images.min():.3f}, {sample_images.max():.3f}]")
                print(f"  Output: [{sample_reconstructed.min():.3f}, {sample_reconstructed.max():.3f}]")
    return history


def run(model, config):
    train_loader, test_loader = get_dataloaders(
    config.data_dir, config.category, config.batch_size, config.img_size)

    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    history = trainer_fit(model, optimizer, loss_fn, train_loader,
        device=config.device, num_epochs=config.num_epochs,
        model_type=config.model_type, save_dir=config.output_dir)

    scores, labels = trainer_predict(model, test_loader, config.device, method='max_pooling')
    show_evaluation(scores, labels)
    show_statistics(scores, labels)


from model_stfpm import STFPMModel, STFPMLoss
def run_stfpm(model, config):
    train_loader, test_loader = get_dataloaders(
    config.data_dir, config.category, config.batch_size, config.img_size)

    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = STFPMLoss()

    history = trainer_fit(model, optimizer, loss_fn, train_loader,
        device=config.device, num_epochs=config.num_epochs,
        model_type=config.model_type, save_dir=config.output_dir)

    scores, labels = trainer_predict(model, test_loader, config.device, method='max_pooling')
    show_evaluation(scores, labels)
    show_statistics(scores, labels)


if __name__ == "__main__":

    config = get_config()

    if 1:
        # # ========================================
        # config.model_type = "resnet18-ae"
        # resnet18_ae = ResNetAEV1(
        #     backbone='resnet18',
        #     pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet18-f37072fd.pth',
        #     latent_dim=config.latent_dim,
        #     img_size=config.img_size,
        #     use_skip_connections=True)

        # run(resnet18_ae, config)

        # # ========================================
        # config.model_type = "resnet34-ae"
        # resnet34_ae = ResNetAEV1(
        #     backbone='resnet18',
        #     pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet34-b627a593.pth',
        #     latent_dim=config.latent_dim,
        #     img_size=config.img_size,
        #     use_skip_connections=True)

        # run(resnet34_ae, config)

        # # ========================================
        # config.model_type = "resnet50-ae"
        # resnet50_ae = ResNetAEV1(
        #     backbone='resnet50',
        #     pretrained_path='/home/namu/myspace/NAMU/project_2025/backbones/resnet50-0676ba61.pth',
        #     latent_dim=config.latent_dim,
        #     img_size=config.img_size,
        #     use_skip_connections=True)

        # run(resnet50_ae, config)

        # ========================================
        config.model_type = "vanilla-aev3"
        model = VanillaAEV3(
            latent_dim=config.latent_dim,
            img_size=config.img_size)

        run(model, config)

        # config.model_type = "unet-ae"
        # model = UNetAE(base=64)

        # run(model, config)


    if 0:
        config.model_type = "vanilla-ae"
        # model = VanillaAE(
        #     latent_dim=config.latent_dim,
        #     img_size=config.img_size,
        #     backbone='resnet18',
        #     layers=['layer1', 'layer2', 'layer3'],
        # )
        model = VanillaAE(
            latent_dim=config.latent_dim,
            img_size=config.img_size,
        )

        run(model, config)

    if 0:
        config.model_type = "stfpm"
        model = STFPMModel(layers=['layer1', 'layer2', 'layer3'])
        run_stfpm(model, config)
