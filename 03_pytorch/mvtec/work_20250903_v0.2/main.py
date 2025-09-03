import os
import torch

from utils import show_dataloader_info, show_modeler_info, show_trainer_info, show_results
from config import build_config, show_config, merge_configs
from types import SimpleNamespace

from registry import build_dataloader, build_transform
from registry import build_model, build_loss_fn, build_metrics, build_modeler
from registry import build_optimizer, build_scheduler, build_stopper, build_trainer

DATA_DIR = '/mnt/d/datasets/mvtec'

BASE_CONFIGS = SimpleNamespace(
    backbone_dir="/mnt/d/github/deep_learning_tutorials/03_pytorch/backbones",
    current_dir=os.getcwd(),
    output_dir="./experiments",
    num_epochs=10,
    seed=42,
    verbose=True,
)

DATA_CONFIGS = {
    "mvtec": SimpleNamespace(
        dataloader_type="mvtec",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/mvtec",
            categories=["bottle", "cable", "capsule"],
            train_batch_size=16, test_batch_size=8,
            test_ratio=0.2, valid_ratio=0.0,
            num_workers=8, pin_memory=True, persistent_workers=False,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
    "btad": SimpleNamespace(
        dataloader_type="btad",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets/btad",
            categories=["01", "02", "03"],
            train_batch_size=8, test_batch_size=8,
            test_ratio=0.2, valid_ratio=0.0,
            num_workers=8, pin_memory=True, persistent_workers=False,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
    "visa": SimpleNamespace(
        dataloader_type="visa",
        dataloader_params=dict(
            data_dir="/mnt/d/datasets//visa",
            categories=["candle", "capsules", "cashew", "chewinggum", "fryum",
                        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
            train_batch_size=4, test_batch_size=4,
            test_ratio=0.2, valid_ratio=0.0,
            num_workers=8, pin_memory=True, persistent_workers=False,
        ),
        train_transform_type="train",
        train_transform_params=dict(img_size=256),
        test_transform_type="test",
        test_transform_params=dict(img_size=256),
    ),
}

MODEL_CONFIGS = {
    "vanilla_ae": SimpleNamespace(
        modeler_type="ae",
        model_type="vanilla_ae",
        model_params=dict(),
        loss_type="combined",
        loss_params=dict(),
        metric_list=[("psnr", dict()), ("ssim", dict())],   # (metric_type, matric_params)
    ),
    "unet_ae": SimpleNamespace(
        modeler_type="ae",
        model_type="unet_ae",
        model_params=dict(),
        loss_type="combined",
        loss_params=dict(),
        metric_list=[("psnr", dict()), ("ssim", dict())],   # (metric_type, matric_params)
    ),
    "stfpm": SimpleNamespace(
        modeler_type="stfpm",
        model_type="stfpm",
        model_params=dict(backbone="resnet50", layers=["layer1", "layer2", "layer3"]),
        loss_type="stfpm",
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],   # (metric_type, matric_params)
    ),
    "padim": SimpleNamespace(
        modeler_type="padim",
        model_type="padim",
        model_params=dict(backbone="resnet18", layers=["layer2", "layer3"]),
        loss_type="none",
        loss_params=dict(),
        metric_list=[("feature_sim", dict(similarity_fn='cosine'))],   # (metric_type, matric_params)
    ),
}

TRAIN_CONFIS = {
    "gradient": SimpleNamespace(
        trainer_type="gradient",
        optimizer_type="adamw",
        optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
        scheduler_type="step",
        scheduler_params=dict(),
        stopper_type="early",
        stopper_params=dict(patience=5, min_delta=1e-4),
    ),
    "memory": SimpleNamespace(
        trainer_type="memory",
        trainer_params=dict(),
        optimizer_type="none",
        optimizer_params=dict(),
        scheduler_type="none",
        scheduler_params=dict(),
        stopper_type="none",
        stopper_params=dict(),
    ),
}

def show_gpu_memory(stage):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"[{stage}]")
    print(f" > Allocated: {allocated / 1024**2:.2f} MB ({allocated / 1024:.2f} KB)")
    print(f" > Reserved:  {reserved / 1024**2:.2f} MB ({reserved / 1024:.2f} KB)")


def run_experiment(config):
    try:
        torch.cuda.reset_peak_memory_stats()

        ## 1. Data loaders
        train_transform = build_transform(config.train_transform_type, **config.train_transform_params)
        test_transform = build_transform(config.test_transform_type, **config.test_transform_params)
        data = build_dataloader(config.dataloader_type, **config.dataloader_params,
            train_transform=train_transform, test_transform=test_transform)

        ## 2. Modeler
        model = build_model(config.model_type, **config.model_params)
        loss_fn = build_loss_fn(config.loss_type, **config.loss_params)
        metrics = build_metrics(config.metric_list)
        modeler = build_modeler(config.modeler_type, model=model, loss_fn=loss_fn, metrics=metrics)

        ## 3. Trainer
        optimizer = build_optimizer(model, config.optimizer_type, **config.optimizer_params)
        scheduler = build_scheduler(optimizer, config.scheduler_type, **config.scheduler_params)
        stopper = build_stopper(config.stopper_type, **config.stopper_params)
        trainer = build_trainer(modeler, config.trainer_type, scheduler=scheduler, stopper=stopper)
        trainer.optimizer = optimizer

        show_gpu_memory("After model creaton")

        if config.verbose:
            # show_config(config)
            show_dataloader_info(data)
            show_modeler_info(modeler)
            # show_trainer_info(trainer)

        ## 4. Training & Evaluation
        history = trainer.fit(data.train_loader(), config.num_epochs, valid_loader=data.valid_loader())
        scores, labels = trainer.predict(data.test_loader())
        show_results(scores, labels)

    finally:
        if 'trainer' in locals():         del trainer
        if 'stopper' in locals():         del stopper
        if 'scheduler' in locals():       del scheduler
        if 'optimizer' in locals():       del optimizer
        if 'modeler' in locals():         del modeler
        if 'metrics' in locals():         del metrics
        if 'loss_fn' in locals():         del loss_fn
        if 'model' in locals():           del model
        if 'data' in locals():            del data
        if 'test_transform' in locals():  del test_transform
        if 'train_transfomr' in locals(): del train_transform

        import gc
        gc.collect()

        torch.cuda.empty_cache()
        if config.verbose:
            print(f"[Memory Summary] Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            print(f"[Memory Summary] Max reserved:  {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
            show_gpu_memory("After cleanup")


def build_config(data_type, model_type, train_type):
    config = SimpleNamespace()
    config = merge_configs(config, BASE_CONFIGS)
    config = merge_configs(config, DATA_CONFIGS[data_type])
    config = merge_configs(config, MODEL_CONFIGS[model_type])
    config = merge_configs(config, TRAIN_CONFIS[train_type])

    if model_type in ["patchcore", "padim"]:    # memory-based models
        config.num_epochs = 1
        config.dataloader_params["train_batch_size"] = 4
        config.dataloader_params["test_batch_size"] = 2
        config.dataloader_params["valid_ratio"] = 0.0
        config.dataloader_params["train_shuffle"] = False
        config.dataloader_params["test_shuffle"] = False
        config.dataloader_params["train_drop_last"] = False
        config.dataloader_params["test_drop_last"] = False

    return config


if __name__ == "__main__":

    # config = build_config("mvtec", "vanilla_ae", "gradient")
    # # config.dataloader_params["categories"] = ["bottle"]
    # run_experiment(config)

    # config = build_config("mvtec", "unet_ae", "gradient")
    # # config.dataloader_params["categories"] = ["bottle"]
    # run_experiment(config)

    # config = build_config("mvtec", "stfpm", "gradient")
    # # config.dataloader_params["categories"] = ["bottle"]
    # run_experiment(config)

    config = build_config("btad", "vanilla_ae", "gradient")
    config.dataloader_params["categories"] = ["01"]
    run_experiment(config)
