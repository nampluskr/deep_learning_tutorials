from config import Config, DataConfig, ModelConfig, TrainConfig, print_config, save_config
from data import MVTecDataset, get_transforms, get_dataloader, split_train_valid
from modeler import Modeler, get_model, save_weights
from trainer import Trainer, get_optimizer, get_scheduler, get_logger
from trainer import set_seed, get_device
from stopper import get_stopper
from evaluator import Evaluator, evaluate_anomaly_detection, compute_threshold


data_config = DataConfig(
    data_dir='/home/namu/myspace/NAMU/datasets/mvtec',
    # categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
    #               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    #               'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
    categories=["grid", "carpet", "tile"],
    # categories=["bottle"],
    batch_size=16,
    img_size=512,
)
train_config = TrainConfig(
    num_epochs=100,
    optimizer_type="adamw",
    optimizer_params={"lr": 1e-3},
    stopper_type="stop",
    stopper_params={"max_epoch": 3},
)

config1 = Config(
    output_dir="./experiments",
    save_model=True,
    save_config=True,

    data=data_config,
    model=ModelConfig(
        model_type="wae",
        model_params={"latent_dim": 256, "img_size": data_config.img_size,
            # "backbone": "resnet50",
            # "backbone": "vgg19",
        }
    ),
    train=train_config
)
config2 = Config(
    output_dir="./experiments",
    save_model=True,
    save_config=True,

    data=data_config,
    model=ModelConfig(
        model_type="vae",
        model_params={"latent_dim": 256, "img_size": data_config.img_size}
    ),
    train=train_config,
)
config3 = Config(
    output_dir="./experiments",
    save_model=True,
    save_config=True,

    data=data_config,
    model=ModelConfig(
        model_type="unet_ae",
        model_params={"latent_dim": 256, "img_size": data_config.img_size}
    ),
    train=train_config,
)
# config_list = [config3, config1, config2]
config_list = [config1]


def run(config):
    print_config(config.model)
    set_seed(config.seed)
    config.device = get_device()
    logger = get_logger(config.output_dir)
    logger.info(f"*** MODEL: {config.model.model_type}")

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

    # ###############################################################
    # 4. Evaluate Model on Test Dataset
    # ###############################################################

    evaluator = Evaluator(modeler)
    predictions = evaluator.predict(test_loader)
    scores, labels = predictions["score"], predictions["label"]

    threshold = compute_threshold(scores, labels, method="f1")
    evaluate_anomaly_detection(scores, labels, threshold)

    # ###############################################################
    # 5. Save Model
    # ###############################################################

    if config.save_model:
        filename = f"{config.model.model_type}_weights.pth"
        save_weights(model, config.output_dir, filename)

    if config.save_config:
        filename = f"{config.model.model_type}_config.json"
        save_config(config, config.output_dir, filename)


if __name__ == "__main__":

    for config in config_list:
        run(config)
