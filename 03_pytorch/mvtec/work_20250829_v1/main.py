# main.py
# Entry point for anomaly detection framework with unified dataloaders & anomaly map evaluation

import argparse
import torch
import matplotlib.pyplot as plt
import os

# Dataset loaders
from datasets.dataset_mvtec import get_dataloaders as get_mvtec_dataloaders
from datasets.dataset_visa import get_dataloaders as get_visa_dataloaders
from datasets.dataset_btad import get_dataloaders as get_btad_dataloaders
from datasets.dataset_oled import get_dataloaders as get_oled_dataloaders

# Models & Modelers
from models.model_fastflow import FastflowModel, FastflowLoss
from models.model_padim import PadimModel, PadimLoss
from models.model_patchcore import PatchcoreModel, PatchcoreLoss
from models.model_stfpm import STFPMModel, STFPMLoss
from models.model_autoencoder import VanillaAE, UNetAE, AutoencoderLoss
from models.model_vae import VanillaVAE, UNetVAE, VAELoss

from modelers.modeler_fastflow import FastflowModeler
from modelers.modeler_padim import PadimModeler
from modelers.modeler_patchcore import PatchcoreModeler
from modelers.modeler_stfpm import STFPMModeler
from modelers.modeler_autoencoder import AutoencoderModeler
from modelers.modeler_vae import VAEModeler

# Trainer
from trainer import Trainer


def get_dataloaders(dataset_name, data_dir, category, img_size, batch_size, valid_ratio):
    """Dataset selector"""
    dataset_name = dataset_name.lower()
    if dataset_name == "mvtec":
        return get_mvtec_dataloaders(data_dir, category, img_size, batch_size, valid_ratio)
    elif dataset_name == "visa":
        return get_visa_dataloaders(data_dir, category, img_size, batch_size, valid_ratio)
    elif dataset_name == "btad":
        return get_btad_dataloaders(data_dir, category, img_size, batch_size, valid_ratio)
    elif dataset_name == "oled":
        return get_oled_dataloaders(data_dir, categories=["normal", "defect"], img_size=img_size, batch_size=batch_size, valid_ratio=valid_ratio)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_modeler(model_name, device="cuda"):
    """Model selector"""
    model_name = model_name.lower()
    if model_name == "fastflow":
        model = FastflowModel(backbone="resnet18")
        return FastflowModeler(model, FastflowLoss(), device=device)
    elif model_name == "padim":
        model = PadimModel(backbone="resnet18")
        return PadimModeler(model, PadimLoss(), device=device)
    elif model_name == "patchcore":
        model = PatchcoreModel(backbone="resnet18")
        return PatchcoreModeler(model, PatchcoreLoss(), device=device)
    elif model_name == "stfpm":
        model = STFPMModel(backbone="resnet18")
        return STFPMModeler(model, STFPMLoss(), device=device)
    elif model_name == "vanillaae":
        model = VanillaAE(in_channels=3, img_size=256, latent_dim=128)
        return AutoencoderModeler(model, AutoencoderLoss(), device=device)
    elif model_name == "unetae":
        model = UNetAE(in_channels=3, img_size=256)
        return AutoencoderModeler(model, AutoencoderLoss(), device=device)
    elif model_name == "vanillavae":
        model = VanillaVAE(in_channels=3, img_size=256, latent_dim=128)
        return VAEModeler(model, VAELoss(), device=device)
    elif model_name == "unetvae":
        model = UNetVAE(in_channels=3, img_size=256, latent_dim=128)
        return VAEModeler(model, VAELoss(), device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_anomaly_maps(modeler, dataloader, save_dir="results", num_samples=5, device="cuda"):
    """Visualize anomaly maps for a few test samples"""
    os.makedirs(save_dir, exist_ok=True)
    modeler.model.eval()
    count = 0
    for batch in dataloader:
        images, labels = batch["image"].to(device), batch["label"]
        outputs = modeler.predict_step({"image": images})

        anomaly_maps = outputs.get("anomaly_map")
        if anomaly_maps is None:
            continue

        for i in range(images.size(0)):
            if count >= num_samples:
                return
            img = images[i].permute(1, 2, 0).cpu().numpy()
            amap = anomaly_maps[i, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(img)
            axes[0].set_title("Input")
            axes[0].axis("off")
            axes[1].imshow(amap, cmap="jet")
            axes[1].set_title("Anomaly Map")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{count}.png"))
            plt.close(fig)
            count += 1


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Framework")
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset: mvtec|visa|btad|oled")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--category", type=str, default="bottle", help="Category for dataset (e.g. bottle, pcb1, 01)")
    parser.add_argument("--model", type=str, default="fastflow", help="Model name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    # Load data
    train_loader, valid_loader, test_loader = get_dataloaders(
        args.dataset, args.data_dir, args.category,
        img_size=args.img_size,
        batch_size=args.batch_size,
        valid_ratio=0.2
    )

    # Load model
    modeler = get_modeler(args.model, device=args.device)

    # Optimizer (only for trainable models)
    optimizer = torch.optim.Adam(modeler.model.parameters(), lr=1e-3) if any(p.requires_grad for p in modeler.model.parameters()) else None

    trainer = Trainer(modeler, optimizer=optimizer, scheduler=None, device=args.device)

    # Train
    if optimizer is not None:
        trainer.fit(train_loader, valid_loader, num_epochs=args.epochs)

    # Evaluate
    results = trainer.evaluate(test_loader)
    print("Evaluation Results:", results)

    # Visualize anomaly maps
    visualize_anomaly_maps(modeler, test_loader, save_dir=args.save_dir, num_samples=5, device=args.device)


if __name__ == "__main__":
    main()
