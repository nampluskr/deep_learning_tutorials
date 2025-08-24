from config import Config
from data import MVTecDataset, get_transforms, get_dataloader, split_train_valid
from modeler import Modeler, get_model
from trainer import Trainer, get_optimizer, get_scheduler, get_logger
from trainer import set_seed, get_device
from stopper import get_stopper


def run(config):

    set_seed(config.seed)
    config.device = get_device()
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

    # model = get_model('vanilla_ae', latent_dim=512, img_size=256)
    model = get_model('vae', latent_dim=512, img_size=256)
    modeler = Modeler(model, config.device)

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
