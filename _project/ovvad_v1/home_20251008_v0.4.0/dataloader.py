import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class BaseDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None, mask_transform=None):
        super().__init__()
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []

        self.load_data(root_dir, category, split)
        print(f" > {category} - {split} set: {len(self)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

    def load_data(self, root_dir, category, split):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        height, width = image.shape[-2:]

        if label == 0:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = (np.array(mask) > 0).astype(np.uint8)

        label = torch.tensor(label).long()
        mask = torch.tensor(mask).long()
        name = os.path.basename(image_path)
        defect_type = self.defect_types[idx]
        return dict(image=image, label=label, mask=mask, name=name, defect_type=defect_type)


class MVTecDataset(BaseDataset):
    def load_data(self, root_dir, category, split):
        category_dir = os.path.join(root_dir, category)

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "good")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")

        elif split == "test":
            normal_image_dir = os.path.join(category_dir, "test", "good")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")

            test_dir = os.path.join(category_dir, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                if defect_type == "good": continue

                anomaly_image_dir = os.path.join(test_dir, defect_type)
                anomaly_mask_dir = os.path.join(category_dir, "ground_truth", defect_type)
                for image_name in sorted(os.listdir(anomaly_image_dir)):
                    anomaly_image_path = os.path.join(anomaly_image_dir, image_name)
                    image_stem = os.path.splitext(image_name)[0]
                    anomaly_mask_path = os.path.join(anomaly_mask_dir, f"{image_stem}_mask.png")
                    self.image_paths.append(anomaly_image_path)
                    self.mask_paths.append(anomaly_mask_path)
                    self.labels.append(1)
                    self.defect_types.append(defect_type)


class VisADataset(BaseDataset):
    def load_data(self, root_dir, category, split="train", test_ratio=0.2):
        csv_path = os.path.join(root_dir, category, "image_anno.csv")
        df = pd.read_csv(csv_path)

        normal_df = df[df["label"] == "normal"].reset_index(drop=True)
        anomaly_df = df[df["label"] != "normal"].reset_index(drop=True)

        if split == "train":
            normal_train, _ = train_test_split(normal_df,
                test_size=test_ratio, random_state=42, shuffle=True)
            subset = normal_train
        elif split == "test":
            _, normal_test = train_test_split(normal_df,
                test_size=test_ratio, random_state=42, shuffle=True)
            subset = pd.concat([normal_test, anomaly_df], axis=0).reset_index(drop=True)

        for _, row in subset.iterrows():
            image_path = os.path.join(root_dir, row["image"])
            defect_type = row["label"]

            if defect_type == "normal":
                self.image_paths.append(image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
            else:
                self.image_paths.append(image_path)
                mask_path = os.path.join(root_dir, row["mask"])
                self.mask_paths.append(mask_path)
                self.labels.append(1)
                self.defect_types.append(str(defect_type))


class BTADDataset(BaseDataset):
    def load_data(self, root_dir, category, split):
        category_dir = os.path.join(root_dir, category)

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "ok")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")

        elif split == "test":
            normal_image_dir = os.path.join(category_dir, "test", "ok")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")

            anomaly_image_dir = os.path.join(category_dir, "test", "ko")
            anomaly_mask_dir = os.path.join(category_dir, "ground_truth", "ko")
            for image_name in sorted(os.listdir(anomaly_image_dir)):
                anomaly_image_path = os.path.join(anomaly_image_dir, image_name)
                image_stem = os.path.splitext(image_name)[0]
                anomaly_mask_path = None

                for ext in ['.png', '.bmp']:
                    candidate = os.path.join(anomaly_mask_dir, image_stem + ext)
                    if os.path.exists(candidate):
                        anomaly_mask_path = candidate
                        break

                self.image_paths.append(anomaly_image_path)
                self.mask_paths.append(anomaly_mask_path)
                self.labels.append(1)
                self.defect_types.append("anomaly")


def get_dataloaders(dataset_type, category, root_dir, img_size, batch_size, normalize=True,
                    num_workers=8, pin_memory=True, persistent_workers=True):
    train_transforms = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    test_transforms = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if normalize:
        normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms.append(normalize_transform)
        test_transforms.append(normalize_transform)

    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose(test_transforms)

    mask_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
    ])

    datasets = {"mvtec": MVTecDataset, "visa": VisADataset, "btad": BTADDataset}
    DatasetClass = datasets.get(dataset_type.lower())
    if DatasetClass is None:
        raise ValueError(f"Unknown dataset: {dataset_type}. Available: {list(datasets.keys())}")

    train_set = DatasetClass(root_dir=root_dir, category=category, split="train",
        transform=train_transform, mask_transform=mask_transform)
    test_set = DatasetClass(root_dir=root_dir, category=category, split="test",
        transform=test_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return train_loader, test_loader
