```python
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
```

```python
import xml.etree.ElementTree as ET

class OxfordPets(Dataset):
    """ Dataset for Object Detection """
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if split == 'train':
            split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
        elif split == 'test':
            split_file = os.path.join(data_dir, 'annotations', 'test.txt')

        # https://github.com/tensorflow/models/issues/3134
        images_png = [
            "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
            "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
            "Abyssinian_5", "Abyssinian_34",
        ]
        images_corrupt = ["chihuahua_121", "beagle_116"]

        self.samples = []
        with open(split_file) as file:
            for line in file:
                filename, label, *_ = line.strip().split()
                if filename not in images_corrupt + images_png:
                    image_path = os.path.join(self.data_dir, 'images', filename + ".jpg")
                    bbox_path = os.path.join(self.data_dir, 'annotations', 'xmls', filename + ".xml")
                    self.samples.append((image_path, bbox_path, int(label) - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bbox_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)   # (H, W, 3)

        # annotation parsing
        tree = ET.parse(bbox_path)
        root = tree.getroot()
        bboxes = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            bboxes.append([xmin, ymin, xmax, ymax])

        labels = [label]*len(bboxes)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        target = {'bboxes': torch.tensor(bboxes).float(), 
                  'labels': torch.tensor(labels).long()}
        return image, target

train_transform = A.Compose([
    A.Resize(320, 320),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

valid_transform = A.Compose([
    A.Resize(320, 320),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
train_dataset = OxfordPets(data_dir, split="train", transform=train_transform)
valid_dataset = OxfordPets(data_dir, split="test", transform=valid_transform)

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

kwargs = {"num_workers": 4, "pin_memory": True, 
          "drop_last": True, "persistent_workers": True}
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, **kwargs)
```

```python
train_transform = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
valid_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
train_dataset = OxfordPets(data_dir, split="train", transform=train_transform)
valid_dataset = OxfordPets(data_dir, split="test", transform=valid_transform)

kwargs = {"num_workers": 4, "pin_memory": True, 
          "drop_last": True, "persistent_workers": True}
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, **kwargs)
```

```python
...
```

