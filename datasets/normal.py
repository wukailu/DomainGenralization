from torch.utils.data import Dataset
import torch
from .utils import FullDatasetBase
from torchvision import transforms


class NormalDataset(Dataset):
    def __init__(self, total_size=10000):
        self.len = total_size
        self.data = torch.randn((total_size,))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor([self.data[index]]), torch.tensor([0])


class NormalFullDataset(FullDatasetBase):
    mean = (0,)
    std = (1,)
    img_shape = (1,)
    num_classes = 1
    name = "normal"

    def __init__(self, total_size=100000, **kwargs):
        super().__init__(**kwargs)
        self.train = NormalDataset(total_size=total_size)
        self.val = NormalDataset(total_size=total_size)
        self.test = NormalDataset(total_size=total_size)

    def gen_test_transforms(self):
        return transforms.Compose([]), None

    def gen_train_transforms(self):
        return transforms.Compose([]), None

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return self.train

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return self.val

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return self.test

    def sample_imgs(self) -> torch.Tensor:
        return torch.stack([torch.zeros(self.img_shape)] * 2)

    @staticmethod
    def is_dataset_name(name: str):
        return name == "normal"
