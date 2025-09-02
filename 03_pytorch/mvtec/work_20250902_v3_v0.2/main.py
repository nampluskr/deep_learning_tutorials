from utils import show_data_info


BACKBONE_DIR = '/mnt/d/github/deep_learning_tutorials/03_pytorch/backbones'
DATA_DIR = '/mnt/d/datasets/mvtec'
# DATALODAER_PARAMS = dict(num_workers=0, pin_memory=False) # Windows
#DATALODAER_PARAMS = dict(num_workers=8, pin_memory=True)  # NAMU
DATALODAER_PARAMS = dict(num_workers=8, pin_memory=True, persistent_workers=True)  # WSL2
OUTPUT_DIR = "./experiments"
DATASET_TYPE = "mvtec"
CATEGORIES = ["bottle", "tile"]
VERBOSE = True

def run():
    from dataloaders.dataloader_mvtec import MVTecDataloader

    data = MVTecDataloader(
        data_dir=DATA_DIR,
        categories=CATEGORIES,
        test_ratio=0.2,
        valid_ratio=0.2,
        **DATALODAER_PARAMS,
    )

    train_loader = data.train_loader()
    valid_loader = data.valid_loader()
    test_loader = data.test_loader()

    if VERBOSE:
        show_data_info(data)




if __name__ == "__main__":

    run()