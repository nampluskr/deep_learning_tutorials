import os
from dataloader import get_dataloaders
from registry import get_trainer, get_config

#####################################################################
# Global VARIABLES
#####################################################################

DATASET_DIR = "/mnt/d/datasets"
BACKBONE_DIR = "/mnt/d/backbones"
OUTPUT_DIR = "/mnt/d/outputs"
SEED = 42
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True



def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(trainer):
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    print()
    print(f" > Total params.:         {total_params:,}")
    print(f" > Trainable params.:     {trainable_params:,}")

    if trainer.optimizer is not None:
        optim_params = sum(p.numel() for group in trainer.optimizer.param_groups for p in group['params'])
        print(f" > Optimizer params.:     {optim_params:,}")


def train(dataset_type, category, model_type, num_epochs=None, batch_size=None, img_size=None, normalize=None):
    config = get_config(model_type)
    num_epochs = num_epochs or config["num_epochs"]
    img_size = img_size or config["img_size"]
    batch_size = batch_size or config["batch_size"]
    normalize = normalize or config["normalize"]

    result_dir = os.path.join(OUTPUT_DIR, dataset_type, category, model_type)
    weight_name = f"model_{dataset_type}_{category}_{model_type}_epochs-{num_epochs}.pth"
    image_prefix = f"image_{dataset_type}_{category}_{model_type}_epochs-{num_epochs}"

    set_seed(seed=SEED)
    train_loader, test_loader = get_dataloaders(dataset_type, category,
        root_dir=os.path.join(DATASET_DIR, dataset_type),
        img_size=img_size, batch_size=batch_size, normalize=normalize,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    trainer = get_trainer(model_type,
        backbone_dir=BACKBONE_DIR, dataset_dir=DATASET_DIR, img_size=img_size)
    count_parameters(trainer)

    trainer.fit(train_loader, num_epochs, valid_loader=test_loader, 
                weight_path=os.path.join(result_dir, weight_name))

    trainer.test(test_loader, result_dir=result_dir, image_prefix=image_prefix, normalize=normalize,
                 skip_normal=True, num_max=10)
    trainer.test(test_loader, result_dir=result_dir, image_prefix=image_prefix, normalize=normalize,
                 skip_anomaly=True, num_max=10)


# def train_stfpm(dataset_type, category, model_type, num_epochs=10, batch_size=16, img_size=256, normalize=True):
#     """ train_stfpm(dataset_type, category, "stfpm", num_epochs=10) """

#     from models.model_stfpm import STFPMTrainer
#     model_type = "stfpm"
#     result_dir = os.path.join(OUTPUT_DIR, dataset_type, category, model_type)
#     weight_name = f"model_{dataset_type}_{category}_{model_type}_epochs-{num_epochs}.pth"
#     image_prefix = f"image_{dataset_type}_{category}_{model_type}_epochs-{num_epochs}"

#     set_seed(seed=SEED)
#     train_loader, test_loader = get_dataloaders(dataset_type, category,
#         root_dir=os.path.join(DATASET_DIR, dataset_type),
#         img_size=img_size, batch_size=batch_size, normalize=normalize,
#         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

#     trainer = STFPMTrainer(backbone_dir=BACKBONE_DIR, 
#         backbone="resnet50", layers=["layer1", "layer2", "layer3"])
#     count_parameters(trainer)

#     trainer.fit(train_loader, num_epochs, valid_loader=test_loader, 
#                 weight_path=os.path.join(result_dir, weight_name))

#     trainer.test(test_loader, result_dir=result_dir, image_prefix=image_prefix, normalize=normalize,
#                  skip_normal=True, num_max=10)
#     trainer.test(test_loader, result_dir=result_dir, image_prefix=image_prefix, normalize=normalize,
#                  skip_anomaly=True, num_max=10)



if __name__ == "__main__":
    dataset_type, category = "mvtec", "tile"

    #############################################################
    ## 1. Memory-based: PaDim(2020), PatchCore(2022), DFKDE(2022)
    #############################################################

    # train(dataset_type, category, "padim", num_epochs=1)
    # train(dataset_type, category, "patchcore", num_epochs=1)

    #############################################################
    ## 2. Nomalizing Flow-based: CFlow(2021), FastFlow(2021), CSFlow(2021), UFlow(2022)
    #############################################################

    # train(dataset_type, category, "cflow-resnet18", num_epochs=3)
    # train(dataset_type, category, "cflow-resnet50", num_epochs=3)
    # train(dataset_type, category, "fastflow-resnet50", num_epochs=10)
    # train(dataset_type, category, "fastflow-cait", num_epochs=5)
    # train(dataset_type, category, "fastflow-deit", num_epochs=10)
    # train(dataset_type, category, "csflow", num_epochs=10)
    # train(dataset_type, category, "uflow-resnet50", num_epochs=10)
    # train(dataset_type, category, "uflow-mcait", num_epochs=10)

    #############################################################
    # 3. Knowledge Distillation: STFPM(2021), FRE(2023), Reverse Distillation(2022), EfficientAD(2024)
    #############################################################

    # train(dataset_type, category, "stfpm", num_epochs=50)
    # train(dataset_type, category, "fre", num_epochs=50)
    # train(dataset_type, category, "efficientad-small", num_epochs=10)
    # train(dataset_type, category, "efficientad-medium", num_epochs=10)
    # train(dataset_type, category, "reverse-distillation", num_epochs=50)

    #############################################################
    # 4. Reconstruction-based: GANomaly(2018), DRAEM(2021), DSR(2022)
    #############################################################

    # train(dataset_type, category, "autoencoder", num_epochs=50)
    # train(dataset_type, category, "draem", num_epochs=10)