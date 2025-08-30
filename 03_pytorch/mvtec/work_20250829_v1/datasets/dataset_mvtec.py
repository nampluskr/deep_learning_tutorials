import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, ImageReadMode

class MVTecDataset(Dataset):
    def __init__(self, data_dir, categories, split="train", transform=None, normal_only=False):
        self.data_dir = data_dir
        self.categories = categories if isinstance(categories, (list, tuple)) else [categories]
        self.split = split
        self.transform = transform
        self.normal_only = normal_only

        self.img_paths = []
        for cat in self.categories:
            base_path = os.path.join(data_dir, "mvtec", cat, split)
            if normal_only and split == "train":
                self.img_paths += glob.glob(os.path.join(base_path, "good", "*.png"))
            else:
                self.img_paths += glob.glob(os.path.join(base_path, "*", "*.png"))
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
        if self.transform:
            img = self.transform(img)

        label = 0 if "good" in img_path else 1
        mask = None
        if label == 1 and self.split == "test":
            mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
            if os.path.exists(mask_path):
                mask = read_image(mask_path, mode=ImageReadMode.GRAY).float() / 255.0
                if self.transform:
                    mask = self.transform(mask)

        return {"image": img, "label": torch.tensor(label), "mask": mask}


def get_dataloaders(data_dir, category="bottle", img_size=256, batch_size=8, valid_ratio=0.2, transform=None):
    from torchvision.transforms import v2 as T
    if transform is None:
        transform = T.Compose([T.Resize((img_size, img_size))])

    dataset = MVTecDataset(data_dir, categories=[category], split="train", transform=transform, normal_only=True)
    test_dataset = MVTecDataset(data_dir, categories=[category], split="test", transform=transform)

    n_train = int(len(dataset) * (1 - valid_ratio))
    n_valid = len(dataset) - n_train
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
