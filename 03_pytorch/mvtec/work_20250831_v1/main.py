from datasets.dataset_factory import get_transform
from datasets.dataset_mvtec import MVTecDataloader
from models.model_ae import VanillaAE, AECombinedLoss
from modelers.modeler_ae import AEModeler
from metrics.metrics_default import get_metric


def run_ae():
    data = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["carpet", "leather", "tile"],
        train_batch_size=16,
        test_batch_size=8,
        train_transform = get_transform("train"),
        test_transform = get_transform("test"),
        valid_ratio=0.2,
        seed=42,
        num_workers= 8,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = data.train_loader()
    valid_loader = data.valid_loader()
    test_loader = data.test_loader()

    modeler = AEModeler(
        model=VanillaAE(),
        loss_fn=AECombinedLoss(),
        metrics={
            "psnr": get_metric("psnr"),
            "ssim": get_metric("ssim"),
        },
    )
    
    print(f">> OK!")


if __name__ == "__main__":

    run_ae()