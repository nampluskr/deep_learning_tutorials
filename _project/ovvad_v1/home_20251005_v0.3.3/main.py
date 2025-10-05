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

def get_trainer(model_type, img_size=256):
    trainers = {
        "stfpm": "models.model_stfpm.STFPMTrainer",
        # "padim": "models.model_padim.PaDimTrainer",
        # "patchcore": "models.model_patchcore.PatchCoreTrainer",
    }
    if model_type not in trainers:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    module_path, class_name = trainers[model_type].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    TrainerClass = getattr(module, class_name)
    return TrainerClass()

def run_experiment(model_type, dataset, category, num_epochs, batch_size=16, imagenet_normalize=True):
    from dataloader import get_dataloaders

    print("\n" + "="*50)
    print(f"RUN EXPERIMENT: {dataset} / {category} / {model_type}")
    print("="*50 + "\n")

    config = get_config(model_type, dataset, category, num_epochs)
    config.imagenet_normalize = imagenet_normalize
    config.batch_size = batch_size

    train_loader, test_loader = get_dataloaders(config)
    trainer = get_trainer(config.model_type)
    count_parameters(trainer)
    trainer.fit(train_loader, num_epochs=config.num_epochs, valid_loader=test_loader,
        weight_path=config.weight_path)

    trainer.load_model(weight_path=config.weight_path)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_normal=True, num_max=10, imagenet_normalize=config.imagenet_normalize)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
        skip_anomaly=True, num_max=10, imagenet_normalize=config.imagenet_normalize)


if __name__ == "__main__":
    dataset, category = "mvtec", "grid"

    run_experiment("stfpm", dataset, category, num_epochs=50, batch_size=16, imagenet_normalize=True)