import torch
from config import Config
from data import MVTecDataset, get_transforms, get_dataloader, split_train_valid
from modeler import Modeler, get_model, get_loss_fn, get_metric
from trainer import Trainer, get_optimizer, get_scheduler, get_logger
from stopper import get_stopper


def run(config):

    logger = get_logger("./experments")

    # ###############################################################
    # 1. Load Dataloders
    # ###############################################################

    train_transform, test_transform = get_transforms(img_size=config.img_size)

    train_dataset = MVTecDataset(config.data_dir, config.categories, 'train', train_transform)
    valid_dataset = MVTecDataset(config.data_dir, config.categories, 'train', test_transform)
    test_dataset = MVTecDataset(config.data_dir, config.categories, 'test', test_transform)

    train_dataset, valid_dataset = split_train_valid(
        train_dataset, valid_dataset,
        valid_ratio=config.valid_ratio,
        seed=config.seed
    )
    train_loader = get_dataloader(train_dataset, config.batch_size, 'train')
    valid_loader = get_dataloader(valid_dataset, 16, 'valid')
    test_loader = get_dataloader(test_dataset, 16, 'test')

    # ###############################################################
    # 2. Load Model / loss_fn / metrics
    # ###############################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model('vanilla_ae', latent_dim=512, img_size=256)
    loss_fn = get_loss_fn('combined')
    metrics = {"ssim": get_metric('ssim'), "psnr": get_metric('psnr')}

    # Fast Flow
    # model = get_model("fastflow", backbone="resnet34", layers=["layer2","layer3"])
    # loss_fn = get_loss_fn("fastflow")
    # metrics = {
    #     "log_prob": get_metric("fastflow_log_prob"),
    #     "anomaly_score": get_metric("fastflow_anomaly_score")
    # }

    # PatchCore
    # model = get_model("patchcore", backbone="resnet18", layers=["layer2","layer3"])
    # model.to(device)
    # model.build_memory_bank(train_loader, device)
    # loss_fn = get_loss_fn("patchcore")
    # metrics = {"anomaly_score": get_metric("patchcore_anomaly_score")}
    
    # STFPM
    # model = get_model("stfpm", backbone="resnet18", layers=["layer1","layer2","layer3"])
    # loss_fn = get_loss_fn("stfpm")
    # metrics = {"anomaly_score": get_metric("stfpm_anomaly_score")}

    modeler = Modeler(model, loss_fn, metrics, device)

    # ###############################################################
    # 3. Train Model on Train Dataset
    # ###############################################################

    optimizer = get_optimizer(model, "adamw", lr=1e-4, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, "plateau")
    stopper = get_stopper("stop", max_epoch=5)

    trainer = Trainer(modeler, optimizer, scheduler, logger, stopper)
    history = trainer.fit(train_loader, num_epochs=100, valid_loader=valid_loader)


if __name__ == "__main__":

    config = Config()
    run(config)
