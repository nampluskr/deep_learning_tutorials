```python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

## RuntimeError: CUDA error: device-side asseret triggered
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
```

```python
import xml.etree.ElementTree as ET

class OxfordPetsDetection(Dataset):
    """Object Detection Dataset for Oxford Pets (VOC format)"""
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Split txt
        if split == 'train':
            split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
        elif split == 'test':
            split_file = os.path.join(data_dir, 'annotations', 'test.txt')

        self.samples = []
        with open(split_file) as file:
            for line in file:
                filename, *_ = line.strip().split()
                image_path = os.path.join(self.data_dir, 'images', filename + ".jpg")
                xml_path = os.path.join(self.data_dir, 'annotations', 'xmls', filename + ".xml")
                self.samples.append((image_path, xml_path))

        # 라벨 이름과 int 매핑 (optional)
        self.class_to_idx = {}  # {class_name: class_idx}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, xml_path = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            # VOC bbox: xmin, ymin, xmax, ymax (1-based index)
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text)) - 1
            ymin = int(float(bndbox.find("ymin").text)) - 1
            xmax = int(float(bndbox.find("xmax").text)) - 1
            ymax = int(float(bndbox.find("ymax").text)) - 1
            boxes.append([xmin, ymin, xmax, ymax])

            # 클래스 이름을 인덱스로 변환
            if name not in self.class_to_idx:
                self.class_to_idx[name] = len(self.class_to_idx) + 1  # 0은 background
            labels.append(self.class_to_idx[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) # (N, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64) # (N,)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transform:
            # Albumentations 사용시, bbox 및 label 변환 필요
            transformed = self.transform(image=image, bboxes=boxes.numpy(), labels=labels.numpy())
            image = transformed["image"] / 255.0
            boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)
            target["boxes"] = boxes
            target["labels"] = labels
        else:
            # ToTensor 변환 (C,H,W)
            image = torch.from_numpy(image).permute(2,0,1).float() / 255.0

        return image, target
```

```python
train_transform = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

valid_transform = A.Compose([
    A.Resize(224,224),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
train_dataset = OxfordPetsDetection(data_dir, split="train", transform=train_transform)
valid_dataset = OxfordPetsDetection(data_dir, split="test", transform=valid_transform)

def collate_fn(batch):
    return tuple(zip(*batch))

# kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True, "persistent_workers": True}
kwargs = {"num_workers": 4}
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, **kwargs)
```

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.class_to_idx) + 1  # background 포함

model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4)
```

```python
num_epochs = 10
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        # print(masks.min(), masks.max())
        # assert mask.min() >= 0 and masks.max() < 37 or masks.max() == 255

        images = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)   # 반환: dict
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}/{num_epochs}] loss={avg_loss:.4f}")
```
