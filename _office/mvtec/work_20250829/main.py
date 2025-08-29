from dataset_factory import get_transform, get_dataloader
from model_factory import get_model, get_loss, get_metric
from trainer_factory import get_optimizer, get_scheduler, get_stopper, get_logger, get_device

from modeler import AEModeler
from trainer import AETrainer, set_seed
from evaluator import Evaluator, evaluate_anomaly_detection, compute_threshold

from types import SimpleNamespace


def get_config():
    config = SimpleNamespace()

    # global options
    config.seed = 42
    config.output_dir = "./experiments"

    # dataset options
    config.dataset = "mvtec"
    config.data_dir = '/home/namu/myspace/NAMU/datasets/mvtec'
    config.categories = ['tile', 'carpet', 'grid']
    config.img_size = 512

    config.train_transform = get_transform("train", img_size=config.img_size)
    config.test_transform = get_transform("test", img_size=config.img_size)
    config.train_batch_size = 32
    config.test_batch_size = 32

    # dataloader options
    config.dataloader = get_dataloader(config.dataset,
        data_dir=config.data_dir,
        categories=config.categories,
        train_transform=config.train_transform,
        test_transform=config.test_transform,
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # modeler options
    config.model_name= "unet_ae"
    config.model_params = dict(latent_dim=256, img_size=config.img_size)
    config.model = get_model(config.model_name, **config.model_params)

    config.loss_name = "combined"
    config.loss_params = dict()
    config.loss_fn = get_loss(config.loss_name, **config.loss_params)

    config.metrics = {"psnr": get_metric("psnr"), "ssim": get_metric("ssim")}
    config.device = get_device()

    # trainer options
    config.num_epochs = 50
    config.optimizer = get_optimizer(config.model, "adamw", lr=1e-4, weight_decay=1e-5)
    # config.scheduler = get_scheduler(config.optimizer, "step", step_size=5)
    # config.stopper = get_stopper("stop", max_epoch=config.num_epochs)
    config.stopper = get_stopper("early_stop")
    config.scheduler = None
    config.logger = get_logger(config.output_dir)

    return config


def run(config):
    set_seed(config.seed)

    train_loader = config.dataloader.train_loader()
    valid_loader = config.dataloader.valid_loader()
    test_loader = config.dataloader.test_loader()

    modeler = AEModeler(config.model, config.loss_fn, config.metrics, config.device)
    trainer = AETrainer(modeler, config.optimizer, config.scheduler, config.stopper, config.logger)
    history = trainer.fit(train_loader, config.num_epochs, valid_loader=valid_loader)

    evaluator = Evaluator(modeler)
    results = evaluator.predict(test_loader)

    scores, labels = results['score'], results['label']
    threshold = compute_threshold(scores, labels, method='percentile')
    evaluate_anomaly_detection(scores, labels, threshold)


if __name__ == "__main__":

    config = get_config()
    run(config)


