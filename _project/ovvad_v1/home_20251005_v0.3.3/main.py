# main.py
import os
import torch
from registry import ModelRegistry


def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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


def get_config(model_type, dataset, category, num_epochs):
    from types import SimpleNamespace

    config = SimpleNamespace(
        dataset=dataset,
        dataset_dir=os.path.join("/mnt/d/datasets", dataset),
        backbone_dir="/mnt/d/backbones",
        category=category,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        seed=42,
        model_type=model_type,
        num_epochs=num_epochs,
        weight_name=f"model_{dataset}_{category}_{model_type}_epochs-{num_epochs}.pth",
        output_dir=os.path.join("/mnt/d/outputs", dataset, category, model_type),
    )
    config.weight_path = os.path.join(config.output_dir, config.weight_name)
    return config


def get_trainer_from_registry(model_type, backbone_dir, dataset_dir, img_size):
    config = ModelRegistry.get(model_type)

    module_path = config["trainer_path"]
    module_path, class_name = module_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    TrainerClass = getattr(module, class_name)

    trainer_kwargs = config["trainer_kwargs"].copy()
    trainer_kwargs['backbone_dir'] = backbone_dir

    if 'input_size' in trainer_kwargs:
        trainer_kwargs['input_size'] = (img_size, img_size)
    if 'img_size' in trainer_kwargs:
        trainer_kwargs['img_size'] = img_size

    if 'dtd_dir' in trainer_kwargs:
        trainer_kwargs['dtd_dir'] = os.path.join(dataset_dir, "dtd")

    return TrainerClass(**trainer_kwargs)


def run(model_type, dataset, category, num_epochs=None, batch_size=None, normalize=None):
    from dataloader import get_dataloaders

    model_config = ModelRegistry.get(model_type)
    train_cfg = model_config["train_config"]

    num_epochs = num_epochs or train_cfg["epochs"]
    batch_size = batch_size or train_cfg["batch_size"]
    normalize = normalize if normalize is not None else train_cfg["normalize"]
    img_size = train_cfg["img_size"]

    print("\n" + "="*70)
    print(f"MODEL: {model_type} | DATA: {dataset}/{category}")
    print(f"EPOCHS: {num_epochs} | BATCH: {batch_size} | NORMALIZE: {normalize} | SIZE: {img_size}")
    print("="*70 + "\n")

    # Config 생성
    config = get_config(model_type, dataset, category, num_epochs)
    set_seed(seed=config.seed)

    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset,
        dataset_dir=config.dataset_dir,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        normalize=normalize
    )
    trainer = get_trainer_from_registry(
        model_type,
        config.backbone_dir,
        config.dataset_dir,
        img_size
    )
    count_parameters(trainer)
    trainer.fit(train_loader, num_epochs=num_epochs,
                valid_loader=test_loader, weight_path=config.weight_path)

    trainer.load_model(weight_path=config.weight_path)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
                 skip_normal=True, num_max=10, normalize=normalize)
    trainer.test(test_loader, output_dir=config.output_dir, img_prefix=config.model_type,
                 skip_anomaly=True, num_max=10, normalize=normalize)


def benchmark(dataset, category, model_list=None):
    if model_list is None:
        model_list = ModelRegistry.list_models()

    print(f"\n{'='*70}")
    print(f"BENCHMARKING {len(model_list)} MODELS")
    print(f"Dataset: {dataset}/{category}")
    print(f"{'='*70}\n")

    for model_type in model_list:
        try:
            run(model_type, dataset, category)
        except Exception as e:
            print(f"\n[ERROR] {model_type}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    dataset, category = "mvtec", "tile"

    print("Available models:", ModelRegistry.list_models())
    print("\nModels by category:")
    for cat, models in ModelRegistry.list_by_category().items():
        print(f"  {cat}: {', '.join(models)}")

    # 단일 모델 실행
    run("stfpm", dataset, category)

    # 특정 모델들 벤치마크
    # benchmark(dataset, category, ["padim", "patchcore", "stfpm"])

    # 전체 벤치마크
    # benchmark(dataset, category)