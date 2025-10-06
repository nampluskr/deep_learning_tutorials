import os
import torch



def set_seed(seed=42):
    import random
    import numpy as np

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
    from types import SimpleNamespace
    config = SimpleNamespace(
        dataset=dataset,
        # dataset_dir=os.path.join("/home/namu/myspace/NAMU/datasets", dataset),
        dataset_dir=os.path.join("/mnt/d/datasets", dataset),
        backbone_dir="/mnt/d/backbones",
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
    config.normalize = False if model_type.startswith("efficientad") else True
    return config

def get_model_config(model_type):
    configs = {
        # Memory-based
        "padim": dict(num_epochs=1, batch_size=8, normalize=True, img_size=256),
        "patchcore": dict(num_epochs=1, batch_size=8, normalize=True, img_size=256),
        
        # Normalizing Flow
        "cflow-resnet18": dict(num_epochs=10, batch_size=4, normalize=True, img_size=256),
        "cflow-resnet50": dict(num_epochs=10, batch_size=4, normalize=True, img_size=256),
        "fastflow-resnet50": dict(num_epochs=10, batch_size=8, normalize=True, img_size=256),
        "fastflow-cait": dict(num_epochs=10, batch_size=4, normalize=True, img_size=448),
        "fastflow-deit": dict(num_epochs=10, batch_size=8, normalize=True, img_size=384),
        "csflow": dict(num_epochs=10, batch_size=8, normalize=True, img_size=256),
        "uflow-resnet50": dict(num_epochs=10, batch_size=8, normalize=True, img_size=448),
        "uflow-mcait": dict(num_epochs=10, batch_size=4, normalize=True, img_size=448),
        
        # Knowledge Distillation
        "stfpm": dict(num_epochs=20, batch_size=16, normalize=True, img_size=256),
        "fre": dict(num_epochs=20, batch_size=16, normalize=True, img_size=256),
        "efficientad-small": dict(num_epochs=70, batch_size=1, normalize=False, img_size=256),
        "efficientad-medium": dict(num_epochs=70, batch_size=1, normalize=False, img_size=256),
        "reverse-distillation": dict(num_epochs=50, batch_size=8, normalize=True, img_size=256),
        
        # Reconstruction
        "autoencoder": dict(num_epochs=50, batch_size=16, normalize=False, img_size=256),
        "draem": dict(num_epochs=100, batch_size=4, normalize=False, img_size=256),
    }
    return configs.get(model_type, dict(num_epochs=10, batch_size=8, normalize=True, img_size=256))

def get_trainer(model_type, img_size=256, backbone_dir=None, dataset_dir=None):
    trainers = {
        # 1. Memory-based
        "padim": ("models.model_padim.PadimTrainer",
            dict(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"]),
            dict(num_epochs=1, batch_size=8, normalize=True, img_size=256),
        ),
        "patchcore": ("models.model_patchcore.PatchcoreTrainer",
            dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"]),
            dict(num_epochs=1, batch_size=8, normalize=True, img_size=256),
        ),

        # 2. Normalizing Flow
        "cflow-resnet18": ("models.model_cflow.CflowTrainer",
            dict(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        ),
        "cflow-resnet50": ("models.model_cflow.CflowTrainer",
            dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"])
        ),
        "csflow": ("models.model_csflow.CsFlowTrainer",
            dict(input_size=(img_size, img_size), num_channels=3)
        ),
        "fastflow-resnet50": ("models.model_fastflow.FastflowTrainer",
            dict(backbone="wide_resnet50_2", input_size=(img_size, img_size))
        ),
        "fastflow-cait": ("models.model_fastflow.FastflowTrainer",
            dict(backbone="cait_m48_448", input_size=(img_size, img_size))
        ),
        "fastflow-deit": ("models.model_fastflow.FastflowTrainer",
            dict(backbone="deit_base_distilled_patch16_384", input_size=(img_size, img_size))),
        "uflow-resnet50": ("models.model_uflow.UflowTrainer",
            dict(backbone="wide_resnet50_2", input_size=(img_size, img_size))
        ),
        "uflow-mcait": ("models.model_uflow.UflowTrainer",
            dict(backbone="mcait", input_size=(img_size, img_size))
        ),

        # 3. Knowledge Distillation
        "stfpm": ("models.model_stfpm.STFPMTrainer",
            dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"])
        ),
        "fre": ("models.model_fre.FRETrainer",
            dict(backbone="resnet50", layer="layer3")
        ),
        "efficientad-small": ("models.model_efficientad.EfficientAdTrainer",
            dict(model_size="small")
        ),
        "efficientad-medium": ("models.model_efficientad.EfficientAdTrainer",
            dict(model_size="medium")
        ),
        "reverse-distillation": ("models.model_reverse_distillation.ReverseDistillationTrainer",
            dict(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], input_size=(img_size, img_size))
        ),
        
        # 4. Rconstruction-based
        "autoencoder": ("models.model_autoencoder.AutoencoderTrainer",
            dict(latent_dim=128, img_size=img_size)
        ),
        "draem": ("models.model_draem.DraemTrainer",
            dict(sspcab=True, dtd_dir=os.path.join(dataset_dir, "dtd"))
        ),
    }
    if model_type not in trainers:
        raise ValueError(f"Unknown model_type: {model_type}")

    module_path, kwargs = trainers[model_type]
    module_path, class_name = module_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    TrainerClass = getattr(module, class_name)
    kwargs['backbone_dir'] = backbone_dir or "/mnt/d/backbones"
    return TrainerClass(**kwargs)


def run(model_type, dataset, category, num_epochs, batch_size=16, normalize=True):
    from dataloader import get_dataloaders

    print("\n" + "="*50)
    print(f"RUN EXPERIMENT: {dataset} / {category} / {model_type}")
    print("="*50 + "\n")

    config = get_config(model_type, dataset, category, num_epochs)
    set_seed(seed=config.seed)
    if model_type in ["fastflow-cait", "uflow-mcait"]:
        config.img_size = 448
    elif model_type == "fastflow-deit":
        config.img_size = 384
    else:
        config.img_size = 256
    config.normalize = normalize
    config.batch_size = batch_size

    train_loader, test_loader = get_dataloaders(config)
    trainer = get_trainer(config.model_type, img_size=config.img_size, 
                          backbone_dir=config.backbone_dir, dataset_dir=config.dataset_dir)
    count_parameters(trainer)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader,
        weight_path=config.weight_path)

    trainer.load_model(weight_path=config.weight_path)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_normal=True, num_max=10, normalize=config.normalize)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_anomaly=True, num_max=10, normalize=config.normalize)


if __name__ == "__main__":
    dataset, category = "mvtec", "tile"

    #############################################################
    ## 1. Memory-based: PaDim(2020), PatchCore(2022), DFKDE(2022)
    #############################################################

    # run("padim", dataset, category, num_epochs=1)
    # run("patchcore", dataset, category, num_epochs=1)

    #############################################################
    ## 2. Nomalizing Flow-based: CFlow(2021), FastFlow(2021), CSFlow(2021), UFlow(2022)
    #############################################################

    # run("cflow-resnet18", dataset, category, num_epochs=3)
    # run("cflow-resnet50", dataset, category, num_epochs=3)
    # run("fastflow-resnet50", dataset, category, num_epochs=10)
    # run("fastflow-cait", dataset, category, num_epochs=10)
    # run("fastflow-deit", dataset, category, num_epochs=10)
    # run("csflow", dataset, category, num_epochs=10)
    # run("uflow-resnet50", dataset, category, num_epochs=10)
    # run("uflow-mcait", dataset, category, num_epochs=10)

    #############################################################
    # 3. Knowledge Distillation: STFPM(2021), FRE(2023), Reverse Distillation(2022), EfficientAD(2024)
    #############################################################

    # run("stfpm", dataset, category, num_epochs=50)
    # run("fre", dataset, category, num_epochs=50)
    # run("efficientad-small", dataset, category, num_epochs=10)
    # run("efficientad-medium", dataset, category, num_epochs=10)
    # run("reverse-distillation", dataset, category, num_epochs=50)

    #############################################################
    # 4. Reconstruction-based: GANomaly(2018), DRAEM(2021), DSR(2022)
    #############################################################

    run("autoencoder", dataset, category, num_epochs=50)
    # run("draem", dataset, category, num_epochs=10)




