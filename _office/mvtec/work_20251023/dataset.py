import os
import numpy as np
import gzip
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T




class BaseDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform or T.ToTensor()
        self.image_paths = []
        self.images = []
        self.load_data(root_dir, split, **kwargs)

    def load_data(self, root_dir, split, **kwargs):
        raise NotImplementedError

    def __len__(self):
        if self.images:
            return len(self.images)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.images:
            image = self.images[idx]
        else:
            image = Image.open(self.image_paths[idx])

        image = image.convert('L') if image.mode in ('L', 'LA') else image.convert('RGB')
        image = self.transform(image)
        image_path = self.image_paths[idx] if self.image_paths else None
        return dict(image=image, image_path=image_path)


class ClassificationDataset(BaseDataset):
    def __init__(self, root_dir, split, transform=None, **kwargs):
        super().__init__(root_dir, split, transform, **kwargs)
        self.labels = []
        self.class_names = []

    def load_data(self, root_dir, split, **kwargs):
        raise NotImplementedError

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        image = base["image"]
        label = torch.tensor(self.labels[idx]).long()
        class_name = self.class_names[idx]
        return dict(image=image, label=label, class_name=class_name)


class MNIST(ClassificationDataset):
    CLASS_NAMES = [str(i) for i in range(10)]

    def __init__(self, root_dir, split="train", transform=None, **kwargs):
        super().__init__(root_dir, split, transform, **kwargs)
        self.num_classes = len(self.CLASS_NAMES)

    def load_data(self, root_dir, split, **kwargs):
        if split == "train":
            image_path = os.path.join(root_dir, "train-images-idx3-ubyte.gz")
            label_path = os.path.join(root_dir, "train-labels-idx1-ubyte.gz")
        elif split == "test":
            image_path = os.path.join(root_dir, "t10k-images-idx3-ubyte.gz")
            label_path = os.path.join(root_dir, "t10k-labels-idx1-ubyte.gz")
        else:
            raise ValueError("split must be 'train' or 'test'")

        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            raise FileNotFoundError(f"MNIST files not found in {root_dir}")

        with gzip.open(label_path, "rb") as f:
            header = np.frombuffer(f.read(8), dtype='>u4')
            magic, num_items = header
            if magic != 2049:
                raise ValueError(f"Invalid label file magic number: {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8).tolist()

        with gzip.open(image_path, "rb") as f:
            header = np.frombuffer(f.read(16), dtype='>u4')
            magic, num_items, rows, cols = header
            if magic != 2051:
                raise ValueError(f"Invalid image file magic number: {magic}")
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape((num_items, rows, cols))

        self.images = [
            Image.frombytes('L', (cols, rows), images[i].tobytes())
            for i in range(num_items)
        ]
        self.labels = labels
        self.class_names = [str(label) for label in labels]


class FashionMNIST(ClassificationDataset):
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    def __init__(self, root_dir, split="train", transform=None, **kwargs):
        super().__init__(root_dir, split, transform, **kwargs)
        self.num_classes = len(self.CLASS_NAMES)

    def load_data(self, root_dir, split, **kwargs):
        if split == "train":
            image_path = os.path.join(root_dir, "train-images-idx3-ubyte.gz")
            label_path = os.path.join(root_dir, "train-labels-idx1-ubyte.gz")
        elif split == "test":
            image_path = os.path.join(root_dir, "t10k-images-idx3-ubyte.gz")
            label_path = os.path.join(root_dir, "t10k-labels-idx1-ubyte.gz")
        else:
            raise ValueError("split must be 'train' or 'test'")

        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            raise FileNotFoundError(f"FashionMNIST files not found in {root_dir}")

        with gzip.open(label_path, "rb") as f:
            header = np.frombuffer(f.read(8), dtype='>u4')
            magic, num_items = header
            if magic != 2049:
                raise ValueError(f"Invalid label file magic number: {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8).tolist()
            if len(labels) != num_items:
                raise ValueError("Label count mismatch")

        with gzip.open(image_path, "rb") as f:
            header = np.frombuffer(f.read(16), dtype='>u4')
            magic, num_items, rows, cols = header
            if magic != 2051:
                raise ValueError(f"Invalid image file magic number: {magic}")
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape((num_items, rows, cols))

        self.images = [
            Image.frombytes('L', (cols, rows), images[i].tobytes())
            for i in range(num_items)
        ]
        self.labels = labels
        self.class_names = [self.CLASS_NAMES[label] for label in labels]


class Cifar10(ClassificationDataset):
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, root_dir, split="train", transform=None, **kwargs):
        super().__init__(root_dir, split, transform, **kwargs)
        self.num_classes = len(self.CLASS_NAMES)

    def load_data(self, root_dir, split, **kwargs):
        data_dir = os.path.join(root_dir, "cifar-10-batches-py")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"CIFAR-10 data not found in {data_dir}")

        if split == "train":
            data_batches = [f"data_batch_{i}" for i in range(1, 6)]
        elif split == "test":
            data_batches = ["test_batch"]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.images = []
        self.labels = []
        self.class_names = []

        for batch_name in data_batches:
            batch_file = os.path.join(data_dir, batch_name)
            if not os.path.isfile(batch_file):
                raise FileNotFoundError(f"Batch file not found: {batch_file}")

            with open(batch_file, "rb") as f:
                import pickle
                data = pickle.load(f, encoding="bytes")
                images = data[b"data"]      # (N, 3072)
                labels = data[b"labels"]

                # (N, 3072) → (N, 3, 32, 32) -> PIL Image
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (H, W, C)
                for i in range(images.shape[0]):
                    image = Image.fromarray(images[i])
                    self.images.append(image)
                    self.labels.append(labels[i])
                    self.class_names.append(self.CLASS_NAMES[labels[i]])

class OxfordPets(ClassificationDataset):
    CLASS_NAMES = [
        'Abyssinian', 'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound',
        'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_Shorthair',
        'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter',
        'German_Shepherd', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin',
        'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher',
        'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll',
        'Russian_Blue', 'Saint_Bernard', 'Samoyed', 'Scottish_Terrier',
        'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
        'Wheaten_Terrier', 'Yorkshire_Terrier'
    ]

    def __init__(self, root_dir, split="train", transform=None, **kwargs):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.class_names = []
        super().__init__(root_dir, split, transform, **kwargs)

    def load_data(self, root_dir, split, **kwargs):
        if split == "train":
            split_file = os.path.join(root_dir, "annotations", "trainval.txt")
        elif split == "test":
            split_file = os.path.join(root_dir, "annotations", "test.txt")
        else:
            raise ValueError("split must be 'train' or 'test'")

        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # https://github.com/tensorflow/models/issues/3134
        images_png = [
            "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
            "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
            "Abyssinian_5", "Abyssinian_34",
        ]
        images_corrupt = ["chihuahua_121", "beagle_116"]

        with open(split_file, 'r') as file:
            for line in file:
                filename, label_str, *_ = line.strip().split()
                if filename in images_png + images_corrupt : continue

                image_path = os.path.join(root_dir, "images", f"{filename}.jpg")
                label = int(label_str) - 1  # 1-based -> 0-based
                class_name = self.CLASS_NAMES[label]

                if not os.path.isfile(image_path):
                    print(f"Warning: Image not found {image_path}")
                    continue

                self.image_paths.append(image_path)
                self.labels.append(label)
                self.class_names.append(class_name)


class AnomalyDataset(BaseDataset):
    def __init__(self, root_dir, split="train", transform=None, mask_transform=None, **kwargs):
        self.labels = []        # 0: normal, 1: anomaly
        self.mask_paths = []
        self.defect_types = []
        self.categories = []
        self.has_mask = None
        self.mask_transform = mask_transform or T.ToTensor()

        super().__init__(root_dir, split, transform, **kwargs)
        if self.has_mask is None:
            raise ValueError(f"Set self.has_mask to True or False")

    def load_data(self, root_dir, split, **kwargs):
        raise NotImplementedError

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()
        defect_type = self.defect_types[idx]
        category = self.categories[idx]

        has_mask = torch.tensor(self.has_mask).bool()
        mask = None

        if self.has_mask:
            mask_path = self.mask_paths[idx]
            height, width = image.shape[1], image.shape[2]

            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = (np.array(mask) > 0).astype(np.uint8) * 255
                mask = Image.fromarray(mask, mode='L')
            else:
                mask = Image.new('L', (width, height), 0)

            mask = self.mask_transform(mask).float()

        return dict(image=image, label=label, mask=mask, has_mask=has_mask,
                    defect_type=defect_type, category=category)


class MVTec(AnomalyDataset):
    def load_data(self, root_dir, split="train", dataset_type="mvtec", category="bottle", **kwargs):
        self.has_mask = True
        category_dir = os.path.join(root_dir, dataset_type, category)

        if not os.path.exists(category_dir):
            raise ValueError(f"Category directory not found: {category_dir}")

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "good")
            if not os.path.exists(normal_image_dir):
                raise ValueError(f"Train normal directory not found: {normal_image_dir}")

            for image_name in sorted(os.listdir(normal_image_dir)):
                ext = os.path.splitext(image_name)[1].lower()
                if ext not in ('.png', '.jpg', '.jpeg', '.bmp'): continue

                normal_image_path = os.path.join(normal_image_dir, image_name)
                if not os.path.isfile(normal_image_path): continue

                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

        elif split == "test":
            # 1. Normal (good)
            normal_image_dir = os.path.join(category_dir, "test", "good")
            if os.path.exists(normal_image_dir):
                for image_name in sorted(os.listdir(normal_image_dir)):
                    ext = os.path.splitext(image_name)[1].lower()
                    if ext not in ('.png', '.jpg', '.jpeg', '.bmp'): continue
                    normal_image_path = os.path.join(normal_image_dir, image_name)
                    if not os.path.isfile(normal_image_path): continue

                    self.image_paths.append(normal_image_path)
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.defect_types.append("good")
                    self.categories.append(category)

            # 2. Anomaly (defect)
            test_dir = os.path.join(category_dir, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                if defect_type == "good": continue

                anomaly_image_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(anomaly_image_dir): continue

                anomaly_mask_dir = os.path.join(category_dir, "ground_truth", defect_type)
                if not os.path.exists(anomaly_mask_dir):
                    print(f"Warning: Ground truth mask directory not found: {anomaly_mask_dir}")
                    continue

                for image_name in sorted(os.listdir(anomaly_image_dir)):
                    ext = os.path.splitext(image_name)[1].lower()
                    if ext not in ('.png', '.jpg', '.jpeg', '.bmp'): continue
                    anomaly_image_path = os.path.join(anomaly_image_dir, image_name)
                    if not os.path.isfile(anomaly_image_path): continue

                    image_stem = os.path.splitext(image_name)[0]
                    anomaly_mask_path = os.path.join(anomaly_mask_dir, f"{image_stem}_mask.png")
                    if not os.path.exists(anomaly_mask_path):
                        print(f"Warning: Mask file not found for {anomaly_image_path}")

                    self.image_paths.append(anomaly_image_path)
                    self.mask_paths.append(anomaly_mask_path if os.path.exists(anomaly_mask_path) else None)
                    self.labels.append(1)
                    self.defect_types.append(defect_type)
                    self.categories.append(category)

        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'test'")


class VisA(AnomalyDataset):
    def load_data(self, root_dir, split, dataset_type, category, test_ratio=0.2, **kwargs):
        self.has_mask = True
        category_dir = os.path.join(root_dir, dataset_type, category)
        csv_path = os.path.join(category_dir, "image_anno.csv")

        if not os.path.exists(csv_path):
            raise ValueError(f"Annotation CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        normal_df = df[df["label"] == "normal"].reset_index(drop=True)
        anomaly_df = df[df["label"] != "normal"].reset_index(drop=True)

        if split == "train":
            normal_train, _ = train_test_split(normal_df, test_size=test_ratio,
                random_state=42, shuffle=True)
            subset = normal_train
        elif split == "test":
            _, normal_test = train_test_split(normal_df, test_size=test_ratio,
                random_state=42, shuffle=True)
            subset = pd.concat([normal_test, anomaly_df], axis=0).reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'test'")

        for _, row in subset.iterrows():
            image_path = os.path.join(root_dir, dataset_type, row["image"])
            if not os.path.isfile(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            defect_type = row["label"]
            if defect_type == "normal":
                self.image_paths.append(image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("normal")
                self.categories.append(category)
            else:
                mask_path = os.path.join(root_dir, dataset_type, row["mask"])
                if not os.path.isfile(mask_path):
                    print(f"Warning: Mask not found: {mask_path}")

                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path if os.path.isfile(mask_path) else None)
                self.labels.append(1)
                self.defect_types.append(defect_type)
                self.categories.append(category)

class BTAD(AnomalyDataset):
    def load_data(self, root_dir, dataset_type, category, split="train", **kwargs):
        self.has_mask = True
        category_dir = os.path.join(root_dir, dataset_type, category)

        if not os.path.exists(category_dir):
            raise ValueError(f"Category directory not found: {category_dir}")

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "ok")
            if not os.path.exists(normal_image_dir):
                raise ValueError(f"Train normal directory not found: {normal_image_dir}")

            for image_name in sorted(os.listdir(normal_image_dir)):
                ext = os.path.splitext(image_name)[1].lower()
                if ext not in ('.png', '.jpg', '.jpeg', '.bmp'):
                    continue
                image_path = os.path.join(normal_image_dir, image_name)
                if not os.path.isfile(image_path):
                    continue

                self.image_paths.append(image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("normal")
                self.categories.append(category)

        elif split == "test":
            # 1. Normal (ok)
            normal_image_dir = os.path.join(category_dir, "test", "ok")
            if os.path.exists(normal_image_dir):
                for image_name in sorted(os.listdir(normal_image_dir)):
                    ext = os.path.splitext(image_name)[1].lower()
                    if ext not in ('.png', '.jpg', '.jpeg', '.bmp'):
                        continue
                    image_path = os.path.join(normal_image_dir, image_name)
                    if not os.path.isfile(image_path):
                        continue

                    self.image_paths.append(image_path)
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.defect_types.append("normal")
                    self.categories.append(category)

            # 2. Anomaly (ko)
            anomaly_image_dir = os.path.join(category_dir, "test", "ko")
            anomaly_mask_dir = os.path.join(category_dir, "ground_truth", "ko")

            if not os.path.exists(anomaly_image_dir):
                print(f"Warning: Anomaly image directory not found: {anomaly_image_dir}")
                return

            if not os.path.exists(anomaly_mask_dir):
                print(f"Warning: Ground truth mask directory not found: {anomaly_mask_dir}")
                return

            for image_name in sorted(os.listdir(anomaly_image_dir)):
                ext = os.path.splitext(image_name)[1].lower()
                if ext not in ('.png', '.jpg', '.jpeg', '.bmp'): continue
                image_path = os.path.join(anomaly_image_dir, image_name)
                if not os.path.isfile(image_path): continue

                image_stem = os.path.splitext(image_name)[0]
                mask_path = None
                for mask_ext in ['.png', '.bmp']:
                    candidate = os.path.join(anomaly_mask_dir, image_stem + mask_ext)
                    if os.path.isfile(candidate):
                        mask_path = candidate
                        break

                if not mask_path:
                    print(f"Warning: Mask not found for image: {image_path}")

                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)  # 없으면 None
                self.labels.append(1)
                self.defect_types.append("anomaly")
                self.categories.append(category)

        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'test'")


import os
from torch.utils.data import DataLoader
from dataset import Transform


def get_dataset_dir(alias):
    return os.path.join(os.getenv("DATA_DIR", "./data"), alias)


class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, dataset_type, dataset_class, task, split, dataset_config, dataloader_config):
        key = f"{dataset_type}/{split}"
        cls._registry[key] = {
            "dataset_path": get_dataset_dir(dataset_type),
            "task": task,
            "dataset_class": dataset_class,
            "dataset_config": dataset_config,
            "dataloader_config": dataloader_config,
            "split": split
        }

    @classmethod
    def get(cls, dataset_type, split="train"):
        key = f"{dataset_type}/{split}"
        config = cls._registry.get(key)
        if config is None:
            raise ValueError(f"Unknown dataset: '{key}'. Available: {list(cls._registry.keys())}")
        return config

    @classmethod
    def is_registered(cls, dataset_type: str, split: str = "train") -> bool:
        key = f"{dataset_type}/{split}"
        return key in cls._registry

    @classmethod
    def list_datasets(cls):
        datasets = set()
        for key in cls._registry.keys():
            dataset_type, _ = key.split("/", 1)
            datasets.add(dataset_type)
        return sorted(datasets)

    @classmethod
    def list_by_task(cls):
        tasks = {
            "binary_classification": [],
            "multiclass_classification": [],
            "segmentation": [],
            "object_detection": [],
            "anomaly_detection": [],
        }
        for key, config in cls._registry.items():
            dataset_type = key.split("/")[0]
            task = config["task"]
            if dataset_type not in sum(tasks.values(), []):
                if task in tasks:
                    tasks[task].append(dataset_type)
        return tasks


def get_dataloader(dataset_type, split="train", **kwargs):
    config = DatasetRegistry.get(dataset_type, split)

    # Transform 설정
    transform_config = {
        "split": split,
        "img_size": config["dataset_config"].get("img_size", 224),
        "normalize": config["dataset_config"].get("normalize", True),
        "task": config["task"]
    }
    transform = Transform(**transform_config)

    # Dataset 설정
    dataset_class_path = config["dataset_class"]
    module_path, class_name = dataset_class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    dataset_class = getattr(module, class_name)

    dataset_config = config["dataset_config"].copy()
    dataset_config.update({
        "split": split,
        "transform": transform
    })
    # anomaly 데이터셋의 경우 mask_transform 추가
    if config["task"] == "anomaly_detection":
        mask_transform = Transform(split=split, img_size=transform_config["img_size"], normalize=False)
        dataset_config["mask_transform"] = mask_transform

    dataset = dataset_class(**dataset_config)

    # Dataloader 설정
    dataloader_config = config["dataloader_config"]
    return DataLoader(dataset, **dataloader_config)


def register_all_datasets():
    data_dir = get_dataset_dir("")

    # Classification Datasets
    for split in ["train", "test"]:
        shuffle = True if split == "train" else False
        batch_size = 64 if split == "train" else 100
        workers = 4 if split == "train" else 2

        DatasetRegistry.register(
            dataset_type="mnist",
            dataset_class="dataset.MNIST",
            task="multiclass_classification",
            split=split,
            dataset_config=dict(root_dir=os.path.join(data_dir, "mnist")),
            dataloader_config=dict(shuffle=shuffle, batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=True)
        )

        DatasetRegistry.register(
            dataset_type="fashion_mnist",
            dataset_class="dataset.FashionMNIST",
            task="multiclass_classification",
            split=split,
            dataset_config=dict(root_dir=os.path.join(data_dir, "fashion_mnist")),
            dataloader_config=dict(shuffle=shuffle, batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=True)
        )

        DatasetRegistry.register(
            dataset_type="cifar10",
            dataset_class="dataset.Cifar10",
            task="multiclass_classification",
            split=split,
            dataset_config=dict(root_dir=os.path.join(data_dir, "cifar10")),
            dataloader_config=dict(shuffle=shuffle, batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=True)
        )

        DatasetRegistry.register(
            dataset_type="oxford_pets",
            dataset_class="dataset.OxfordPets",
            task="multiclass_classification",
            split=split,
            dataset_config=dict(root_dir=os.path.join(data_dir, "oxford_pets")),
            dataloader_config=dict(shuffle=shuffle, batch_size=16, num_workers=8 if split == "train" else 4, pin_memory=True, persistent_workers=True)
        )

    # Anomaly Detection Datasets
    for dataset_type in ["mvtec", "visa", "btad"]:
        for split in ["train", "test"]:
            shuffle = split == "train"
            batch_size = 16 if split == "train" else 1
            workers = 4 if split == "train" else 1

            DatasetRegistry.register(
                dataset_type=dataset_type,
                dataset_class=f"dataset.{dataset_type.capitalize()}",
                task="anomaly_detection",
                split=split,
                dataset_config=dict(
                    root_dir=data_dir,
                    dataset_type=dataset_type,
                    category="bottle",
                    img_size=224,
                    normalize=True
                ),
                dataloader_config=dict(shuffle=shuffle, batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=True)
            )


register_all_datasets()
