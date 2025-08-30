from dataset_factory import get_transform
from dataset_mvtec import MVTecDataloader
from model_ae import VanillaAE, UNetAE, AECombinedLoss
from modeler_ae import AEModeler

from model_padim import PadimModel
from modeler_padim import PadimModeler

from metrics import get_metric
from trainer import Trainer


def run_autoencoder():
    train_loader, valid_loader, test_loader = get_dataloaders()
    modeler = AEModeler(
        model=UNetAE(),
        loss_fn=AECombinedLoss(),
        metrics={"psnr": get_metric("psnr"),
                 "ssim": get_metric("ssim")
        },
    )
    trainer = Trainer(modeler)
    trainer.fit(train_loader, num_epochs=10, valid_loader=valid_loader)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")

def run_padim():
    train_loader, valid_loader, test_loader = get_dataloaders()
    modeler = PadimModeler(
        model=PadimModel(),
    )
    trainer = Trainer(modeler)
    trainer.fit(train_loader, num_epochs=10)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")


def run_padim():
    train_loader, valid_loader, test_loader = get_dataloaders()
    modeler = PadimModeler(
        model=PadimModel(),
    )
    trainer = Trainer(modeler)
    trainer.fit(train_loader, num_epochs=10)
    scores, labels = trainer.predict(test_loader)
    show_results(scores, labels, method="roc")


def get_dataloaders():
    mvtec = MVTecDataloader(
        data_dir='/mnt/d/datasets/mvtec',
        categories=["bottle"],
        train_batch_size=32,
        test_batch_size=16,
        train_transform = get_transform("train"),
        test_transform = get_transform("test"),
        valid_ratio=0.3,
        seed=42,
        num_workers= 8,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = mvtec.train_loader()
    valid_loader = mvtec.valid_loader()
    test_loader = mvtec.test_loader()

    print(f" > Train dataset: {len(train_loader.dataset)}")
    print(f" > Valid dataset: {len(valid_loader.dataset)}")
    print(f" > Test dataset:  {len(test_loader.dataset)}")
    return train_loader, valid_loader, test_loader


def show_results(scores, labels, method="roc"):
    auroc_metric = get_metric("auroc")
    aupr_metric = get_metric("aupr")
    threshold_metric = get_metric("threshold", method=method)

    auroc = auroc_metric(labels, scores)
    aupr = aupr_metric(labels, scores)
    optimal_threshold = threshold_metric(labels, scores)

    print()
    print(f" > AUROC:     {auroc:.4f}")
    print(f" > AUPR:      {aupr:.4f}")
    print(f" > Threshold: {optimal_threshold:.4f}")


if __name__ == "__main__":

    # run_autoencoder()
    run_padim()