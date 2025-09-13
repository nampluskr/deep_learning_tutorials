import os
from utils import show_modeler_info, show_model_info, show_trainer_info, show_results_old, show_results_new
from dataloader import get_dataloaders
from trainer_old import GradientTrainer

from model_ae import VanillaAE, AECombinedLoss, SSIMMetric
from model_stfpm import STFPMModel, STFPMLoss, FeatureSimilarityMetric

from modelers.modeler_ae import AEModeler
from modelers.modeler_stfpm import STFPMModeler

import os
import random
import numpy as np
import torch
import torch.optim as optim


def run_autoencoder():
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)

    train_loader, test_loader = get_dataloaders(root='/mnt/d/datasets/mvtec', category="grid",
                                                batch_size=4, img_size=256)

    model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512)
    show_model_info(model)

    loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean')
    metrics = {"ssim": SSIMMetric(data_range=1.0)}
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    modeler = AEModeler(model, loss_fn, metrics)

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

    model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18")
    show_model_info(model)

    loss_fn = STFPMLoss()
    metrics = {"feature_sim": FeatureSimilarityMetric()}
    optimizer = optim.AdamW(model.student_model.parameters(), lr=1e-3, weight_decay=1e-2)

    modeler = STFPMModeler(model, loss_fn, metrics)
    show_modeler_info(modeler)

    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)
    show_trainer_info(trainer)

    trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader)
    scores, labels = trainer.predict(test_loader)
    show_results_old(scores, labels)
    show_results_new(scores, labels)


def run_autoencoder_old():
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: AUTOENCODER\n" + "="*50)

    train_loader, test_loader = get_dataloaders(root='/mnt/d/datasets/mvtec', category="grid",
                                                batch_size=4, img_size=256)

    model = VanillaAE(in_channels=3, out_channels=3, latent_dim=512)
    loss_fn = AECombinedLoss(mse_weight=0.7, ssim_weight=0.3, reduction='mean')
    metrics = {"ssim": SSIMMetric(data_range=1.0)}
    modeler = AEModeler(model, loss_fn, metrics)
    show_modeler_info(modeler)

    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)
    show_trainer_info(trainer)

    trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader)
    scores, labels = trainer.predict(test_loader)
    show_results_old(scores, labels)
    show_results_new(scores, labels)



def run_stfpm_old():
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)

    train_loader, test_loader = get_dataloaders(root='/mnt/d/datasets/mvtec', category="grid",
                                                batch_size=4, img_size=256)

    model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18")
    loss_fn = STFPMLoss()
    metrics = {"feature_sim": FeatureSimilarityMetric()}
    modeler = STFPMModeler(model, loss_fn, metrics)
    show_modeler_info(modeler)

    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)
    show_trainer_info(trainer)

    trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader)
    scores, labels = trainer.predict(test_loader)
    show_results_old(scores, labels)
    show_results_new(scores, labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":

    set_seed(seed=42)

    # run_autoencoder_old()
    run_autoencoder()

    # run_stfpm_old()
    run_stfpm()
