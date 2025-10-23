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
