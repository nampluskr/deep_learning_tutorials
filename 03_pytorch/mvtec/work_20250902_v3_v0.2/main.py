from utils import show_dataloader_info
from config import build_config, show_config


BACKBONE_DIR = '/mnt/d/github/deep_learning_tutorials/03_pytorch/backbones'
DATA_DIR = '/mnt/d/datasets/mvtec'
# DATALODAER_PARAMS = dict(num_workers=0, pin_memory=False) # Windows
#DATALODAER_PARAMS = dict(num_workers=8, pin_memory=True)  # NAMU
DATALODAER_PARAMS = dict(num_workers=8, pin_memory=True, persistent_workers=True)  # WSL2
OUTPUT_DIR = "./experiments"
DATALOADER_TYPE = "mvtec"
CATEGORIES = ["bottle", "tile"]
VERBOSE = True


def run_experiment(dataloder_type, model_type, config, verbose=True):
    from registry import build_dataloader
    
    if verbose:
        show_config(config)

    data = build_dataloader(dataloder_type,
        data_dir=DATA_DIR,
        categories=CATEGORIES,
        test_ratio=0.2,
        valid_ratio=0.2,
        **DATALODAER_PARAMS,
    )
    # if verbose:
    #     show_dataloader_info(data)

    train_loader = data.train_loader()
    valid_loader = data.valid_loader()
    test_loader = data.test_loader()





if __name__ == "__main__":



    dataloader_type, model_type = "mvtec", "unet_ae"
    overrides = dict(img_size=1024, train_batch_size=8)
    config = build_config(dataloader_type, model_type, overrides=overrides)
    run_experiment(dataloader_type, model_type, config)