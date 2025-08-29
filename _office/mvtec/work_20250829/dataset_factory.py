import torch
from torchvision.transforms import v2
from dataset import MVTecDataloader


def get_dataloader(name, **params):
    available_list = {
        'mvtec': MVTecDataloader,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {'num_workers': 8, 'pin_memory': True}
    default_params.update(params)
    return selected(**default_params)


def get_transform(name, img_size=256):
    available_list = {
        'train': v2.Compose([
                    v2.Resize((img_size, img_size), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomRotation(degrees=15),
                    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    v2.ToDtype(torch.float32, scale=True),
                ]),
        'test': v2.Compose([
                    v2.Resize((img_size, img_size), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                ]),
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    return available_list[name]



