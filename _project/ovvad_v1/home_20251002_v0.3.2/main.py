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


#############################################################
## 1. Memory-based (3): PaDim(2020), PatchCore(2022), DFKDE(2022)
#############################################################

def run_patchcore(dataset, category):
    from model_patchcore import PatchCore, PatchCoreTrainer

    config = get_config("patchcore", dataset, category, num_epochs=1)
    config.imagenet_normalize = True
    config.batch_size=8
    set_seed(seed=config.seed)
    trainer = PatchCoreTrainer(PatchCore(layers=["layer2", "layer3"], backbone="wide_resnet50_2"))
    run_experiment(trainer, config)

def run_padim(dataset, category):
    from model_padim import PaDim, PaDimTrainer

    config = get_config("padim", dataset, category, num_epochs=1)
    config.imagenet_normalize = True
    config.batch_size=8
    set_seed(seed=config.seed)
    trainer = PaDimTrainer(PaDim(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"],
            pre_trained=True, n_features=None))
    run_experiment(trainer, config)

def run_dfkde(dataset, category):
    from model_dfkde import DFKDE, DFKDETrainer, FeatureScalingMethod

    config = get_config("dfkde", dataset, category, num_epochs=1)
    config.imagenet_normalize = True
    config.batch_size=8
    set_seed(seed=config.seed)
    trainer = DFKDETrainer(DFKDE(layers=["layer4"],
            backbone="wide_resnet50_2",
            pre_trained=True,
            n_pca_components=16,
            feature_scaling_method=FeatureScalingMethod.SCALE,
            max_training_points=40000))
    run_experiment(trainer, config)

#############################################################
## 2. Nomalizing Flow-based (4): CFlow(2021), FastFlow(2021), CSFlow(2021), UFlow(2022)
#############################################################

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

def run_csflow(dataset, category, num_epochs=10):
    from model_csflow import CSFlow, CSFlowTrainer

    config = get_config("csflow", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size = 8
    set_seed(seed=config.seed)
    model = CSFlow(input_size=(256, 256),
            cross_conv_hidden_channels=1024, n_coupling_blocks=4, clamp=3, num_channels=3)
    trainer = CSFlowTrainer(model)
    run_experiment(trainer, config)


def run_uflow(dataset, category, num_epochs=10):
    from model_uflow import UFlow, UFlowTrainer

    config = get_config("uflow", dataset, category, num_epochs)
    config.img_size = 448
    config.imagenet_normalize = True
    config.batch_size = 8
    set_seed(seed=config.seed)
    model = UFlow(input_size=(448, 448),
            backbone="wide_resnet50_2",
            flow_steps=4,
            affine_clamp=2.0,
            affine_subnet_channels_ratio=1.0,
            permute_soft=False,
        )
    trainer = UFlowTrainer(model)
    run_experiment(trainer, config)


#############################################################
# 3. Knowledge Distillation (4): STFPM(2021), FRE(2023), Reverse Distillation(2022), EfficientAD(2024)
#############################################################

def run_stfpm(dataset, category, num_epochs=10):
    from model_stfpm import STFPM, STFPMTrainer

    config = get_config("stfpm-resnet50", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size=16
    set_seed(seed=config.seed)
    trainer = STFPMTrainer(STFPM(backbone="resnet50", layers=["layer1", "layer2", "layer3"]))
    run_experiment(trainer, config)

def run_fre(dataset, category, num_epochs=10):
    from model_fre import FRE, FRETrainer

    config = get_config("fre-resnet50", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size=16
    set_seed(seed=config.seed)
    trainer = FRETrainer(FRE(
            backbone="resnet50",
            pre_trained=True,
            layer="layer3",
            pooling_kernel_size=2,
            input_dim=65536,
            latent_dim=220,
        ))
    run_experiment(trainer, config)

def run_reverse_distillation(dataset, category, num_epochs=10):
    from model_reverse_distillation import ReverseDistillation, ReverseDistillationTrainer, AnomalyMapGenerationMode

    config = get_config("reverse-distillation", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size=16
    set_seed(seed=config.seed)
    trainer = ReverseDistillationTrainer(ReverseDistillation(
            backbone="wide_resnet50_2",
            pre_trained=True,
            layers=["layer1", "layer2", "layer3"],
            input_size=(256, 256),
            anomaly_map_mode=AnomalyMapGenerationMode.ADD,
        ))
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


#############################################################
## 4. Reconstruction-based (4): AutoEncoder(Baseline), GANomaly(2018), DRAEM(2021), DSR(2022)
#############################################################

def run_autoencoder(dataset, category, num_epochs=10):
    from model_autoencoder import AutoEncoder, AutoEncoderTrainer

    config = get_config("autoencoder", dataset, category, num_epochs)
    config.imagenet_normalize = False
    config.batch_size = 16
    set_seed(seed=config.seed)
    model=AutoEncoder(latent_dim=256, img_size=config.img_size)
    trainer = AutoEncoderTrainer(model)
    run_experiment(trainer, config)

def run_ganomaly(dataset, category, num_epochs=100):
    from model_ganomaly import GANomaly, GANomalyTrainer

    config = get_config("ganomaly", dataset, category, num_epochs)
    config.imagenet_normalize = True
    config.batch_size = 8
    set_seed(seed=config.seed)
    model = GANomaly(
        input_size=(config.img_size, config.img_size),
        num_input_channels=3,
        n_features=64,
        latent_vec_size=100,
        extra_layers=0,
        add_final_conv_layer=True
    )
    trainer = GANomalyTrainer(model=model)
    run_experiment(trainer, config)

def run_draem(dataset, category, num_epochs=10):
    from model_draem import DRAEM, DRAEMTrainer

    config = get_config("draem", dataset, category, num_epochs)
    config.imagenet_normalize = False
    config.batch_size = 4
    set_seed(seed=config.seed)
    model = DRAEM(sspcab=True)
    trainer = DRAEMTrainer(model=model)
    run_experiment(trainer, config)


def run_dsr(dataset, category, num_epochs=10):
    from model_dsr import DSR, DSRTrainer

    config = get_config("dsr", dataset, category, num_epochs)
    config.imagenet_normalize = False
    config.batch_size = 8
    set_seed(seed=config.seed)
    model = DSR(
        latent_anomaly_strength=0.2,
        embedding_dim=128,
        num_embeddings=4096,
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64
    )
    trainer = DSRTrainer(model=model)
    run_experiment(trainer, config)

if __name__ == "__main__":

    dataset, category = "mvtec", "grid"

    #############################################################
    ## 1. Memory-based (3): PaDim(2020), PatchCore(2022), DFKDE(2022)
    #############################################################

    # run_patchcore(dataset, category)
    # run_padim(dataset, category)
    # run_dfkde(dataset, category)

    #############################################################
    ## 2. Nomalizing Flow-based (4): CFlow(2021), FastFlow(2021), CSFlow(2021), UFlow(2022)
    #############################################################

    # run_cflow(dataset, category, num_epochs=10)
    # run_fastflow(dataset, category, num_epochs=10)
    # run_csflow(dataset, category, num_epochs=10)
    # run_uflow(dataset, category, num_epochs=10)

    #############################################################
    # 3. Knowledge Distillation (4): STFPM(2021), FRE(2023), Reverse Distillation(2022), EfficientAD(2024)
    #############################################################

    # run_stfpm(dataset, category, num_epochs=20)
    # run_fre(dataset, category, num_epochs=20)
    # run_reverse_distillation(dataset, category, num_epochs=10)
    # run_efficientad_small(dataset, category, num_epochs=10)
    # run_efficientad_medium(dataset, category, num_epochs=10)

    #############################################################
    # 4. Reconstruction-based (4): AutoEncoder(Baseline), GANomaly(2018), DRAEM(2021), DSR(2022)
    #############################################################

    # run_autoencoder(dataset, category, num_epochs=50)
    # run_ganomaly(dataset, category, num_epochs=20)
    # run_draem(dataset, category, num_epochs=20)
    run_dsr(dataset, category, num_epochs=10)
