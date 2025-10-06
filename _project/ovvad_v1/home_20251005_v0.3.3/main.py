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
        # data_dir=os.path.join("/home/namu/myspace/NAMU/datasets", dataset),
        data_dir=os.path.join("/mnt/d/datasets", dataset),
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
    config.imagenet_normalize = False if model_type.startswith("efficientad") else True
    return config

def get_trainer(model_type, img_size=256, backbone_dir=None):
    trainers = {
        "stfpm": ("models.model_stfpm.STFPMTrainer",
            dict(backbone="resnet50",
                 layers=["layer1", "layer2", "layer3"])
        ),
        "efficientad-small": ("models.model_efficientad.EfficientAdTrainer",
            dict(model_size="small")
        ),
        "efficientad-medium": ("models.model_efficientad.EfficientAdTrainer",
            dict(model_size="medium")
        ),
        "reverse-distillation": ("models.model_reverse_distillation.ReverseDistillationTrainer",
            dict(backbone="wide_resnet50_2",
                 layers=["layer1", "layer2", "layer3"], 
                 input_size=(img_size, img_size))
        ),
        "cflow-resnet18": ("models.model_cflow.CflowTrainer",
            dict(backbone="resnet18", 
                 layers=["layer1", "layer2", "layer3"])
        ),
        "cflow-resnet50": ("models.model_cflow.CflowTrainer",
            dict(backbone="resnet50", 
                 layers=["layer1", "layer2", "layer3"])
        ),
        "csflow": ("models.model_csflow.CsFlowTrainer",
            dict(input_size=(img_size, img_size), 
                 num_channels=3)
        ),
        "fastflow-resnet50": ("models.model_fastflow.FastflowTrainer",
            dict(backbone="wide_resnet50_2",
                 input_size=(img_size, img_size))
        ),
        "fastflow-cait": ("models.model_fastflow.FastflowTrainer",
            dict(backbone="cait_m48_448",
                 input_size=(img_size, img_size))
        ),
        "fastflow-deit": ("models.model_fastflow.FastflowTrainer",
            dict(backbone="deit_base_distilled_patch16_384",
                 input_size=(img_size, img_size))
        ),
        "uflow-resnet50": ("models.model_uflow.UflowTrainer",
            dict(backbone="wide_resnet50_2",
                 input_size=(img_size, img_size))
        ),
        "uflow-mcait": ("models.model_uflow.UflowTrainer",
            dict(backbone="mcait",
                 input_size=(img_size, img_size))
        ),
        "patchcore": ("models.model_patchcore.PatchcoreTrainer",
            dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"])
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


def run(model_type, dataset, category, num_epochs, batch_size=16, imagenet_normalize=True):
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
    config.imagenet_normalize = imagenet_normalize
    config.batch_size = batch_size

    train_loader, test_loader = get_dataloaders(config)
    trainer = get_trainer(config.model_type, img_size=config.img_size, backbone_dir=config.backbone_dir)
    count_parameters(trainer)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader,
        weight_path=config.weight_path)

    trainer.load_model(weight_path=config.weight_path)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_normal=True, num_max=10, imagenet_normalize=config.imagenet_normalize)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_anomaly=True, num_max=10, imagenet_normalize=config.imagenet_normalize)


if __name__ == "__main__":
    dataset, category = "mvtec", "tile"

    # run("stfpm", dataset, category, num_epochs=50, batch_size=16, imagenet_normalize=True)
    # run("efficientad-small", dataset, category, num_epochs=10, batch_size=1, imagenet_normalize=False)
    # run("efficientad-medium", dataset, category, num_epochs=10, batch_size=1, imagenet_normalize=False)
    # run("reverse-distillation", dataset, category, num_epochs=50, batch_size=8, imagenet_normalize=True)
    # run("cflow-resnet18", dataset, category, num_epochs=3, batch_size=4, imagenet_normalize=True)
    # run("cflow-resnet50", dataset, category, num_epochs=3, batch_size=4, imagenet_normalize=True)
    # run("csflow", dataset, category, num_epochs=10, batch_size=8, imagenet_normalize=True)
    # run("fastflow-resnet50", dataset, category, num_epochs=10, batch_size=8, imagenet_normalize=True)
    # run("fastflow-cait", dataset, category, num_epochs=10, batch_size=4, imagenet_normalize=True)
    # run("fastflow-deit", dataset, category, num_epochs=10, batch_size=8, imagenet_normalize=True)
    # run("uflow-resnet50", dataset, category, num_epochs=10, batch_size=8, imagenet_normalize=True)
    # run("uflow-mcait", dataset, category, num_epochs=10, batch_size=8, imagenet_normalize=True)
    # run("patchcore", dataset, category, num_epochs=1, batch_size=8, imagenet_normalize=True)