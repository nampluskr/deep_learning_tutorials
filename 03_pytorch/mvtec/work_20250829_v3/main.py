###########################################################
# Data loaders
###########################################################

from dataset_factory import get_transform
from dataset_mvtec import MVTecDataloader

mvtec = MVTecDataloader(
    data_dir='/mnt/d/datasets/mvtec',
    categories=["bottle"],
    train_batch_size=32,
    test_batch_size=16,
    train_transform = get_transform("train"),
    test_transform = get_transform("test"),
    valid_ratio=0.3,
    seed=42,
    num_workers= 8,
    pin_memory=True,
    persistent_workers=True,
)
train_loader = mvtec.train_loader()
valid_loader = mvtec.valid_loader()
test_loader = mvtec.test_loader()

print(f" > Train dataset: {len(train_loader.dataset)}")
print(f" > Valid dataset: {len(valid_loader.dataset)}")
print(f" > Test dataset:  {len(test_loader.dataset)}")


###########################################################
# Modeler
###########################################################

from model_ae import VanillaAE, AELoss
from modeler_ae import AEModeler
from metrics import get_metric

model = VanillaAE()
loss_fn = AELoss()
metrics = {"psnr": get_metric("psnr"), "ssim": get_metric("ssim")}
modeler = AEModeler(model, loss_fn, metrics={})

###########################################################
# Trainer
###########################################################

from trainer import Trainer

trainer = Trainer(modeler)
history = trainer.fit(train_loader, num_epochs=10, valid_loader=valid_loader)
print(history)

if __name__ == "__main__":
    pass