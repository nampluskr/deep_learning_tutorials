import torch
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset


class BaseDataloader:
    def __init__(self, data_dir, categories,
                 train_transform=None, test_transform=None,
                 train_batch_size=32, test_batch_size=16,
                 test_ratio=0.2, valid_ratio=0.0, seed=42, **params):

        self.categories = categories
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.params = params

        train_normal_dataset = self.get_dataset(data_dir, categories, train_transform, load_normal=True, load_anomaly=False, **params)
        test_normal_dataset = self.get_dataset(data_dir, categories, test_transform, load_normal=True, load_anomaly=False, **params)

        total_size = len(train_normal_dataset)
        test_size = int(total_size * test_ratio)
        valid_size = int(total_size * valid_ratio)
        train_size = total_size - test_size - valid_size

        train_subset, valid_subset, test_subset = random_split(
            range(total_size), [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        self.train_normal_dataset = Subset(train_normal_dataset, train_subset.indices)
        self.valid_normal_dataset = Subset(test_normal_dataset, valid_subset.indices)
        self.test_normal_dataset = Subset(test_normal_dataset, test_subset.indices)
        self.test_anomaly_dataset = self.get_dataset(data_dir, categories, test_transform, load_normal=False, load_anomaly=True, **params)

    def get_dataset(self, data_dir, categories, transform, load_normal, load_anomaly, **params):
        raise NotImplementedError("get_dataset method must be implemented in subclass")

    def train_loader(self):
        return DataLoader(self.train_normal_dataset, self.train_batch_size,
                          shuffle=True, drop_last=True, **self.params)

    def valid_loader(self):
        if len(self.valid_normal_dataset) == 0:
            return None
        else:
            return DataLoader(self.valid_normal_dataset, self.test_batch_size,
                              shuffle=False, drop_last=False, **self.params)

    def test_loader(self):
        if len(self.test_normal_dataset) == 0 and len(self.test_anomaly_dataset) == 0:
            return None
        else:
            test_dataset = ConcatDataset([self.test_normal_dataset, self.test_anomaly_dataset])
            return DataLoader(test_dataset, self.test_batch_size,
                              shuffle=False, drop_last=False, **self.params)
