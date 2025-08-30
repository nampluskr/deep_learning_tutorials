import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, ImageReadMode

class OLEDDataset(Dataset):
    """Custom OLED dataset"""

    def __init__(self, data_dir, categories, split="train", transform=None, normal_only=False):
        self.data_dir = data_dir
        self.categories = categories if isinstance(categories, (list, tuple)) else [categories]
        self.split = split
        self.transform = transform
        self.normal_only = normal_only

        self.img_paths = []
        base_path = os.path.join(data_dir, "oled")
        for cat in self.categories:
            self.img_paths += glob.glob(os.path.join(base_path, cat, "*.png"))
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
        if self.transform:
            img = self.transform(img)

        label = 0 if "normal" in img_path else 1
        return {"image": img, "label": torch.tensor(label), "mask": None}


def get_dataloaders(data_dir, categories=None, img_size=256, batch_size=8, valid_ratio=0.2, transform=None):
    from torchvision.transforms import v2 as T
    if transform is None:
        transform = T.Compose([T.Resize((img_size, img_size))])

    dataset = OLEDDataset(data_dir, categories=categories or ["normal", "defect"], split="train", transform=transform)

    n_train = int(len(dataset) * (1 - valid_ratio))
    n_valid = len(dataset) - n_train
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False),
        DataLoader(dataset, batch_size=batch_size, shuffle=False),  # test_loader == 전체
    )
