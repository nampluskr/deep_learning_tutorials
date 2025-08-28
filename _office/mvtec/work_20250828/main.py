from dataset import MVTecDataloader, get_transforms
from model import get_model, get_loss, get_metric
from modeler import AEModeler
from trainer import set_seed, get_device, get_logger
from trainer import AETrainer, get_optimizer, get_scheduler, get_stopper
from evaluator import Evaluator

from types import SimpleNamespace


def get_config():
    config = SimpleNamespace()

    # global options
    config.seed = 42
    config.output_dir = "./experiments"

    # modeler options
    config.model = get_model("vanilla_ae")
    config.loss_fn = get_loss("combined")
    config.metrics = {"psnr": get_metric("psnr"), "ssim": get_metric("ssim")}
    config.device = get_device()

    # trainer options
    config.optimizer = get_optimizer(config.model, "adamw")
    config.scheduler = get_scheduler(config.optimizer, "step", step_size=5)
    # config.stopper = get_stopper("stop", max_epoch=3)
    config.stopper = None

    config.logger = get_logger(config.output_dir)
    return config


def run(config):
    set_seed(config.seed)
    modeler = AEModeler(config.model, config.loss_fn, config.metrics, config.device)
    trainer = AETrainer(modeler, config.optimizer, config.scheduler, config.stopper, config.logger)
    evaluator = Evaluator(modeler)

    train_transform, test_transform = get_transforms(img_size=256)
    mvtec = MVTecDataloader(
        data_dir='/home/namu/myspace/NAMU/datasets/mvtec',
        categories=['bottle'],
        train_transform=train_transform,
        test_transform=test_transform,
        train_batch_size=32,
        test_batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    train_loader = mvtec.train_loader()
    valid_loader = mvtec.valid_loader()
    test_loader = mvtec.test_loader()

    results = trainer.fit(train_loader, num_epochs=10)


if __name__ == "__main__":

    config = get_config()
    run(config)
