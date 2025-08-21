from data import MVTecDataset, get_transforms, split_train_valid, get_dataloader
from config import Config


config_list = [
    Config(
        data_dir='/mnt/d/datasets/mvtec',
        categories=['bottle', 'cable', 'capsule'],
        img_size=256,
        batch_size=32,
        valid_ratio=0.2,
        seed=42
    ),
]

def run(config):
    # Load datasets
    train_transform, test_transform = get_transforms(img_size=config.img_size)
    train_dataset = MVTecDataset(config.data_dir, config.categories, 'train', transform=train_transform)
    valid_dataset = MVTecDataset(config.data_dir, config.categories, 'train', transform=test_transform)
    test_dataset = MVTecDataset(config.data_dir, config.categories, 'test', transform=test_transform)

    # Load data loaders
    train_dataset, valid_dataset = split_train_valid(train_dataset, valid_dataset,
        valid_ratio=config.valid_ratio, seed=config.seed)
    train_loader = get_dataloader(train_dataset, config.batch_size, 'train')
    valid_loader = get_dataloader(valid_dataset, 16, 'valid')
    test_loader = get_dataloader(test_dataset, 16, 'test')

    print(f'train dataset size: {len(train_loader.dataset)}')
    print(f'valid dataset size: {len(valid_loader.dataset)}')
    print(f'test dataset size:  {len(test_loader.dataset)}')

    # model = get_model()
    # optimizer = get_optimizer(model, config.learning_rate, config.weight_decay, config.optimizer_type)
    # scheduler = get_scheduler(optimizer, config.scheduler_type)
    # loss_fn = get_loss_fn(config.loss_type)
    # metrics = get_metrics(config.metric_names)

    # trainer = Trainer(model, optimizer, scheduler, loss_fn, metrics=metrics)
    # history = trainer.fit(train_loader, num_epochs=config.num_epochs,
                        # valid_loader=valid_loader, shceduler=scheduler)


if __name__ == "__main__":

    for config in config_list:
        run(config)