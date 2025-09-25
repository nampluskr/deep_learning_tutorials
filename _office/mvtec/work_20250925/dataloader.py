import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", transform=None, mask_transform=None):
        self.root = os.path.join(root, category, split)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.labels = []
        self.defect_types = []

        for defect_type in sorted(os.listdir(self.root)):
            label_dir = os.path.join(self.root, defect_type)
            if not os.path.isdir(label_dir):
                continue
            for img_name in sorted(os.listdir(label_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                image_path = os.path.join(label_dir, img_name)
                label = 0 if defect_type == "good" else 1
                self.image_paths.append(image_path)
                self.labels.append(label)
                self.defect_types.append(defect_type)

        print(f" > {split.capitalize()} set: {len(self.image_paths)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        height, width = image.shape[-2:]
        label = self.labels[idx]
        if label == 0:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_path = image_path.replace("test", "ground_truth").replace(".png", "_mask.png")
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = (np.array(mask) > 0).astype(np.uint8)

        label = torch.tensor(label).long()
        mask = torch.tensor(mask).long()
        name = os.path.basename(image_path)
        defect_type = self.defect_types[idx]
        return dict(image=image, label=label, mask=mask, name=name, defect_type=defect_type)


def get_dataloaders(config):
    train_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation(15),
        T.ToTensor(),
        # T.ToImage(),
        # T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])

    test_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        # T.ToImage(),
        # T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])
    mask_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
    ])

    train_set = MVTecDataset(root=config.data_root, category=config.category, split="train",
                             transform=train_transform, mask_transform=mask_transform)
    test_set  = MVTecDataset(root=config.data_root, category=config.category, split="test",
                             transform=test_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers)
    test_loader  = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers)

    return train_loader, test_loader
