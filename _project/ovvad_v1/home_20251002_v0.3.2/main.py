import os
import random
import numpy as np
from types import SimpleNamespace
import torch
import torch.optim as optim

from dataloader import get_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(trainer):
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    print()
    print(f" > Total params.:         {total_params:,}")
    print(f" > Trainable params.:     {trainable_params:,}")

    if trainer.optimizer is not None:
        optim_params = sum(p.numel() for group in trainer.optimizer.param_groups for p in group['params'])
        print(f" > Optimizer params.:     {optim_params:,}")


def get_config(model_type, dataset, category, num_epochs=20):
    config = SimpleNamespace(
        dataset=dataset,
        # data_dir=os.path.join("/home/namu/myspace/NAMU/datasets", dataset),
        data_dir=os.path.join("/mnt/d/datasets", dataset),
        category=category,
        img_size=256,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        # persistent_workers=False, # NAMU
        seed=42,

        model_type=model_type,
        num_epochs=num_epochs,
        img_name=f"img_{dataset}_{category}_{model_type}_epochs-{num_epochs}",
        weight_name=f"model_{dataset}_{category}_{model_type}_epochs-{num_epochs}.pth",
        output_dir=os.path.join("/mnt/d/outputs", dataset, category, model_type),
    )
    config.weight_path=os.path.join(config.output_dir, config.weight_name)
    config.imagenet_normalize = False if model_type.startswith("efficientad") else True
    return config


def run_experiment(trainer, config):
    print("\n" + "="*50)
    print(f"RUN EXPERIMENT: {config.dataset} / {config.category} / {config.model_type}")
    print("="*50 + "\n")

    train_loader, test_loader = get_dataloaders(config)
    count_parameters(trainer)
    # trainer.fit(train_loader, num_epochs=config.num_epochs)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader,
        weight_path=config.weight_path)

    trainer.load_model(weight_path=config.weight_path)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_normal=True, num_max=10, imagenet_normalize=config.imagenet_normalize)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_anomaly=True, num_max=10, imagenet_normalize=config.imagenet_normalize)


def run_autoencoder(dataset, category, num_epochs=10):
    from model_autoencoder import AutoEncoder, AutoEncoderTrainer

    config = get_config("autoencoder", dataset, category, num_epochs)
    config.imagenet_normalize = False
    config.batch_size = 16
    set_seed(seed=config.seed)
    model=AutoEncoder(latent_dim=256, img_size=config.img_size)
    trainer = AutoEncoderTrainer(model)
    run_experiment(trainer, config)


def run_stfpm(dataset, category, num_epochs=10):
    from model_stfpm import STFPM, STFPMTrainer

    config = get_config("stfpm-resnet50", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size=16
    set_seed(seed=config.seed)
    trainer = STFPMTrainer(STFPM(backbone="resnet50", layers=["layer1", "layer2", "layer3"]))
    run_experiment(trainer, config)


def run_efficientad_small(dataset, category, num_epochs=3):
    from model_efficientad import EfficientAD, EfficientADTrainer

    config = get_config("efficientad-small", dataset, category, num_epochs)
    config.imagenet_normalize = False
    config.batch_size=1
    set_seed(seed=config.seed)
    trainer = EfficientADTrainer(EfficientAD(model_size="small"))
    run_experiment(trainer, config)


def run_efficientad_medium(dataset, category, num_epochs=3):
    from model_efficientad import EfficientAD, EfficientADTrainer

    config = get_config("efficientad-medium", dataset, category, num_epochs)
    config.imagenet_normalize = False
    config.batch_size=1
    set_seed(seed=config.seed)
    trainer = EfficientADTrainer(EfficientAD(model_size="medium"))
    run_experiment(trainer, config)


def run_patchcore(dataset, category):
    from model_patchcore import PatchCore, PatchCoreTrainer

    config = get_config("patchcore", dataset, category, num_epochs=1)
    config.imagenet_normalize = True
    config.batch_size=8
    set_seed(seed=config.seed)
    trainer = PatchCoreTrainer(PatchCore(layers=["layer2", "layer3"], backbone="wide_resnet50_2"))
    run_experiment(trainer, config)


def run_cflow(dataset, category, num_epochs=10):
    from model_cflow import CFlow, CFlowTrainer

    config = get_config("cflow-resnet50", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size = 4
    set_seed(seed=config.seed)
    model = CFlow(backbone="resnet50", layers=["layer1", "layer2", "layer3"])
    trainer = CFlowTrainer(model)
    run_experiment(trainer, config)


def run_fastflow(dataset, category, num_epochs=10):
    from model_fastflow import FastFlow, FastFlowTrainer

    config = get_config("fastflow-resnet50", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size = 8
    set_seed(seed=config.seed)
    model = FastFlow(input_size=(256, 256), backbone="wide_resnet50_2")
    trainer = FastFlowTrainer(model)
    run_experiment(trainer, config)


if __name__ == "__main__":

    dataset, category = "mvtec", "tile"

    # run_autoencoder(dataset, category, num_epochs=50)
    # run_stfpm(dataset, category, num_epochs=50)
    # run_efficientad_small(dataset, category, num_epochs=10)
    # run_efficientad_medium(dataset, category, num_epochs=10)
    # run_patchcore(dataset, category)
    # run_cflow(dataset, category, num_epochs=10)
    run_fastflow(dataset, category, num_epochs=10)


