import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", transform=None, mask_transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []

        category_root = os.path.join(root, category)

        if split == "train":
            img_dir = os.path.join(category_root, "train", "good")
            if not os.path.exists(img_dir):
                raise ValueError(f"Train 이미지 경로 없음: {img_dir}")
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(img_dir, img_name)
                    self.image_paths.append(image_path)
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.defect_types.append("good")

        elif split == "test":
            # 1. 정상 (good)
            normal_dir = os.path.join(category_root, "test", "good")
            if os.path.exists(normal_dir):
                for img_name in sorted(os.listdir(normal_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(normal_dir, img_name)
                        self.image_paths.append(image_path)
                        self.mask_paths.append(None)
                        self.labels.append(0)
                        self.defect_types.append("good")

            # 2. 이상 (모든 다른 폴더: scratch, bent 등)
            test_root = os.path.join(category_root, "test")
            for defect_type in sorted(os.listdir(test_root)):
                if defect_type == "good":
                    continue
                defect_dir = os.path.join(test_root, defect_type)
                if not os.path.isdir(defect_dir):
                    continue

                mask_dir = os.path.join(category_root, "ground_truth", defect_type)
                if not os.path.exists(mask_dir):
                    raise ValueError(f"마스크 디렉토리 없음: {mask_dir}")

                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    image_path = os.path.join(defect_dir, img_name)
                    # 마스크 파일명: {이미지명}_mask.png
                    name_stem = os.path.splitext(img_name)[0]
                    mask_name = f"{name_stem}_mask.png"
                    mask_path = os.path.join(mask_dir, mask_name)
                    if not os.path.exists(mask_path):
                        raise FileNotFoundError(f"마스크 없음: {mask_path}")

                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(1)
                    self.defect_types.append(defect_type)

        print(f" > {self.category} - {split.capitalize()} set: {len(self)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

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


class VisADataset(Dataset):
    def __init__(self, root, category, split="train", transform=None, mask_transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []

        data_root = os.path.join(root, category, "Data")

        if split == "train":
            img_dir = os.path.join(data_root, "Images", "Normal")
            if not os.path.exists(img_dir):
                raise ValueError(f"Train 이미지 경로 없음: {img_dir}")
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(img_dir, img_name)
                    self.image_paths.append(image_path)
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.defect_types.append("good")

        elif split == "test":
            # 1. 정상 (Normal)
            normal_dir = os.path.join(data_root, "Images", "Normal")
            if os.path.exists(normal_dir):
                for img_name in sorted(os.listdir(normal_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(normal_dir, img_name)
                        self.image_paths.append(image_path)
                        self.mask_paths.append(None)
                        self.labels.append(0)
                        self.defect_types.append("good")

            # 2. 이상 (Anomaly)
            anomaly_img_dir = os.path.join(data_root, "Images", "Anomaly")
            anomaly_mask_dir = os.path.join(data_root, "Masks", "Anomaly")
            if not os.path.exists(anomaly_img_dir):
                raise ValueError(f"Anomaly 이미지 없음: {anomaly_img_dir}")
            if not os.path.exists(anomaly_mask_dir):
                raise ValueError(f"Anomaly 마스크 디렉토리 없음: {anomaly_mask_dir}")

            for img_name in sorted(os.listdir(anomaly_img_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(anomaly_img_dir, img_name)
                    image_stem = os.path.splitext(img_name)[0]

                    # 마스크 후보 확장자
                    mask_path = None
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        candidate = os.path.join(anomaly_mask_dir, image_stem + ext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break

                    if mask_path is None:
                        raise FileNotFoundError(f"마스크 파일을 찾을 수 없음: {image_stem} "
                                              f"(확장자: .png, .jpg, .jpeg, .bmp)")

                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(1)
                    self.defect_types.append("anomaly")

        print(f" > {self.category} - {split.capitalize()} set: {len(self)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

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
            mask = Image.fromarray(mask)
        else:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('RGB')
            mask_np = np.array(mask)

            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8)

            mask = (mask_np.sum(axis=-1) > 0).astype(np.uint8)
            mask = Image.fromarray(mask)

        # 이제 mask는 PIL.Image → transform 가능
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 최종: PIL → Tensor
        mask = torch.tensor(np.array(mask)).long()
        label = torch.tensor(label).long()
        name = os.path.basename(image_path)
        defect_type = self.defect_types[idx]

        return dict(image=image, label=label, mask=mask, name=name, defect_type=defect_type)


class BTADDataset(Dataset):
    def __init__(self, root, category, split="train", transform=None, mask_transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []

        category_root = os.path.join(root, category)

        if split == "train":
            img_dir = os.path.join(category_root, "train", "ok")
            if not os.path.exists(img_dir):
                raise ValueError(f"Train 이미지 경로 없음: {img_dir}")
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.lower().endswith(('.png', '.bmp')):
                    image_path = os.path.join(img_dir, img_name)
                    self.image_paths.append(image_path)
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.defect_types.append("good")

        elif split == "test":
            # 1. 정상 (ok)
            normal_dir = os.path.join(category_root, "test", "ok")
            if os.path.exists(normal_dir):
                for img_name in sorted(os.listdir(normal_dir)):
                    if img_name.lower().endswith(('.png', '.bmp')):
                        image_path = os.path.join(normal_dir, img_name)
                        self.image_paths.append(image_path)
                        self.mask_paths.append(None)
                        self.labels.append(0)
                        self.defect_types.append("good")

            # 2. 이상 (ko)
            anomaly_img_dir = os.path.join(category_root, "test", "ko")
            anomaly_mask_dir = os.path.join(category_root, "ground_truth", "ko")
            if not os.path.exists(anomaly_img_dir):
                raise ValueError(f"Anomaly 이미지 없음: {anomaly_img_dir}")
            if not os.path.exists(anomaly_mask_dir):
                raise ValueError(f"Anomaly 마스크 디렉토리 없음: {anomaly_mask_dir}")

            for img_name in sorted(os.listdir(anomaly_img_dir)):
                if img_name.lower().endswith(('.png', '.bmp')):
                    image_path = os.path.join(anomaly_img_dir, img_name)
                    image_stem = os.path.splitext(img_name)[0]

                    # 마스크 후보 확장자: .png, .bmp
                    mask_path = None
                    for ext in ['.png', '.bmp']:
                        candidate = os.path.join(anomaly_mask_dir, image_stem + ext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break

                    if mask_path is None:
                        raise FileNotFoundError(f"마스크 파일을 찾을 수 없음: {image_stem} (.png 또는 .bmp)")

                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(1)
                    self.defect_types.append("anomaly")

        print(f" > {self.category} - {split.capitalize()} set: {len(self)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

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
            mask = Image.open(mask_path).convert('L')  # 이진 마스크 (0 또는 255)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = (np.array(mask) > 0).astype(np.uint8)  # 이진화

        label = torch.tensor(label).long()
        mask = torch.tensor(mask).long()
        name = os.path.basename(image_path)
        defect_type = self.defect_types[idx]

        return dict(image=image, label=label, mask=mask, name=name, defect_type=defect_type)


def get_dataloaders(config):
    if config.imagenet_normalize:
        train_transform = T.Compose([
            T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = T.Compose([
            T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = T.Compose([
            T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(15),
            T.ToTensor(),
        ])
        test_transform = T.Compose([
            T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    mask_transform = T.Compose([
        T.Resize((config.img_size, config.img_size), interpolation=T.InterpolationMode.NEAREST),
    ])

    dataset_map = {
        "mvtec": MVTecDataset,
        "visa": VisADataset,
        "btad": BTADDataset,
    }
    DatasetClass = dataset_map.get(config.dataset.lower())
    if DatasetClass is None:
        raise ValueError(f"지원하지 않는 데이터셋: {config.dataset}")

    train_set = DatasetClass(root=config.data_root, category=config.category, split="train",
        transform=train_transform, mask_transform=mask_transform)
    test_set = DatasetClass(root=config.data_root, category=config.category, split="test",
        transform=test_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers)

    print(f"[INFO] 데이터로더 생성 완료 | 모델: {config.model_type}, 데이터셋: {config.dataset}/{config.category}")
    print(f"  Train: {len(train_set)} samples | Test: {len(test_set)} samples")
    return train_loader, test_loader
