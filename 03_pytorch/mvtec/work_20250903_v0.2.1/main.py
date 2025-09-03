import torch

from utils import show_dataloader_info, show_modeler_info, show_trainer_info, show_gpu_memory, show_results
from config import build_config, show_config

from registry import build_dataloader, build_transform
from registry import build_model, build_loss_fn, build_metrics, build_modeler
from registry import build_optimizer, build_scheduler, build_stopper, build_trainer


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
        modeler = build_modeler(model, config.modeler_type, loss_fn=loss_fn, metrics=metrics)

        ## 3. Trainer
        optimizer = build_optimizer(model, config.optimizer_type, **config.optimizer_params)
        scheduler = build_scheduler(optimizer, config.scheduler_type, **config.scheduler_params)
        stopper = build_stopper(config.stopper_type, **config.stopper_params)
        logger = None
        trainer = build_trainer(modeler, optimizer, config.trainer_type, scheduler=scheduler, stopper=stopper, logger=logger)

        show_gpu_memory("After model creaton")

        if config.verbose:
            # show_config(config)
            show_dataloader_info(data)
            show_modeler_info(modeler)
            show_trainer_info(trainer)

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


if __name__ == "__main__":

    # config = build_config("mvtec", "vanilla_ae", "gradient")
    # config.dataloader_params["categories"] = ["bottle"]
    # run_experiment(config)

    config = build_config("mvtec", "stfpm", "gradient")
    config.dataloader_params["categories"] = ["bottle"]
    config.optimizer_params["lr"] = 1e-3
    config.num_epochs = 10
    run_experiment(config)

    # config = build_config("visa", "stfpm", "gradient")
    # config.dataloader_params["categories"] = ["bottle"]
    # run_experiment(config)

    # config = build_config("btad", "vanilla_ae", "gradient")
    # config.dataloader_params["categories"] = ["01"]
    # run_experiment(config)
