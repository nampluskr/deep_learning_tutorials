from config import Config, DataConfig, ModelConfig, TrainConfig, print_config
from data import MVTecDataset, get_transforms, get_dataloader, split_train_valid
from modeler import Modeler, get_model
from trainer import Trainer, get_optimizer, get_scheduler, get_logger
from trainer import set_seed, get_device
from stopper import get_stopper
from evaluator import Evaluator


config1 = Config(
    output_dir="./experiments",
    data=DataConfig(),
    model=ModelConfig(model_type="vanilla_ae"),
    train=TrainConfig(num_epochs=3),
)
config2 = Config(
    output_dir="./experiments",
    data=DataConfig(),
    model=ModelConfig(model_type="vae"),
    train=TrainConfig(num_epochs=3),
)
config_list = [config1, config2]


def run(config):
    print_config(config.model)
    set_seed(config.seed)
    config.device = get_device()
    logger = get_logger(config.output_dir)

    # ###############################################################
    # 1. Load Dataloders
    # ###############################################################

    train_transform, test_transform = get_transforms(img_size=config.data.img_size)

    train_dataset = MVTecDataset(config.data.data_dir, config.data.categories,
        'train', train_transform)
    valid_dataset = MVTecDataset(config.data.data_dir, config.data.categories,
        'train', test_transform)
    test_dataset = MVTecDataset(config.data.data_dir, config.data.categories,
        'test', test_transform)

    train_dataset, valid_dataset = split_train_valid(
        train_dataset, valid_dataset,
        valid_ratio=config.data.valid_ratio,
        seed=config.seed
    )
    train_loader = get_dataloader(train_dataset, config.data.batch_size, 'train')
    valid_loader = get_dataloader(valid_dataset, 16, 'valid')
    test_loader = get_dataloader(test_dataset, 16, 'test')

    # ###############################################################
    # 2. Load Model / loss_fn / metrics
    # ###############################################################

    # model = get_model('vanilla_ae', latent_dim=512, img_size=256)
    model = get_model(config.model.model_type, **config.model.model_params)
    modeler = Modeler(model, config.device)

    # ###############################################################
    # 3. Train Model on Train Dataset with Validataion on Valid Dataset
    # ###############################################################

    optimizer = get_optimizer(model, config.train.optimizer_type,
        **config.train.optimizer_params)
    scheduler = get_scheduler(optimizer, config.train.scheduler_type,
        **config.train.scheduler_params)
    stopper = get_stopper(config.train.stopper_type,
        **config.train.stopper_params)

    trainer = Trainer(modeler, optimizer, scheduler, logger, stopper)
    history = trainer.fit(train_loader, num_epochs=config.train.num_epochs,
        valid_loader=valid_loader)
    print(history)
    
    # ###############################################################
    # 4. Evaluate Model on Test Dataset
    # ###############################################################

    evaluator = Evaluator(modeler)
    results = evaluator.predict(test_loader)
    print(results)

if __name__ == "__main__":

    for config in config_list:
        run(config)
