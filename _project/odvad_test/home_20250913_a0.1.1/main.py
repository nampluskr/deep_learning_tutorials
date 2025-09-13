import os
from utils import show_modeler_info, show_trainer_info, show_results_old, show_results_new
from dataloader import get_dataloaders
from trainer import GradientTrainer

from model_ae import VanillaAE, AECombinedLoss, SSIMMetric
from model_stfpm import STFPMModel, STFPMLoss, FeatureSimilarityMetric

from modelers.modeler_ae import AEModeler
from modelers.modeler_stfpm import STFPMModeler



def run_vanilla_ae():
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)

    train_loader, test_loader = get_dataloaders(root='/mnt/d/datasets/mvtec', category="grid",
                                                batch_size=4, img_size=256)
    modeler = AEModeler(
        model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512),
        loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean'),
        metrics = {"ssim": SSIMMetric(data_range=1.0)},
    )
    show_modeler_info(modeler)

    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)
    show_trainer_info(trainer)

    trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader)
    scores, labels = trainer.predict(test_loader)
    show_results_old(scores, labels)
    show_results_new(scores, labels)



def run_stfpm():
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)

    train_loader, test_loader = get_dataloaders(root='/mnt/d/datasets/mvtec', category="grid",
                                                batch_size=4, img_size=256)
    modeler = STFPMModeler(
        model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18"),
        loss_fn = STFPMLoss(),
        metrics = {"feature_sim": FeatureSimilarityMetric()},
    )
    show_modeler_info(modeler)

    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)
    show_trainer_info(trainer)

    trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader)
    scores, labels = trainer.predict(test_loader)
    show_results_old(scores, labels)
    show_results_new(scores, labels)


if __name__ == "__main__":

    run_vanilla_ae()
    run_stfpm()
