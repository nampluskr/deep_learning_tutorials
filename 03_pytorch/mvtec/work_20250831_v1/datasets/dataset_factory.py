import torch
from torchvision.transforms import v2

from .dataset_mvtec import MVTecDataloader
# from .dataset_btad import BTADDataloader
# from .dataset_visa import VisADataloader
# from .dataset_oled import OLEDDataloader



###########################################################
# Factory functions for datasets
###########################################################

def get_dataloader(name, **params):
    available_list = {
        'mvtec': MVTecDataloader,
        # 'btad': BTADDataloader,
        # 'visa': VisADataloader,
        # 'oled': OLEDDataloader,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    default_params = {
        'train_batch_size': 32,
        'test_batch_size': 16,
        'valid_ratio': 0.2,
        'seed': 42,
        'num_workers': 8,
        'pin_memory': True
    }
    default_params.update(params)
    return available_list[name](**default_params)


###########################################################
# transform with data augmentation
###########################################################

class TrainTransform:
    def __init__(self, img_size=256, **params):
        flip_prob = params.get('flip_prob', 0.5)
        rotation_degrees = params.get('rotation_degrees', 15)
        brightness = params.get('brightness', 0.1)
        contrast = params.get('contrast', 0.1)
        saturation = params.get('saturation', 0.1)
        hue = params.get('hue', 0.05)

        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.RandomHorizontalFlip(p=flip_prob),
            v2.RandomVerticalFlip(p=flip_prob),
            v2.RandomRotation(degrees=rotation_degrees),
            v2.ColorJitter(brightness=brightness, contrast=contrast,
                          saturation=saturation, hue=hue),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestTransform:
    def __init__(self, img_size=256, **params):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, image):
        return self.transform(image)


def get_transform(name, img_size=256, **params):
    available_list = {
        'train': TrainTransform,
        'test': TestTransform,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    default_params = {'img_size': img_size}
    default_params.update(params)
    return available_list[name](**default_params)