import torch
from torch.utils.data import Dataset
from torchvision.datasets import cifar
import datasets

from datasets.utils import FullDatasetBase

__all__ = ["PartialDataset", "OrderDataset", "ConcatFullDataset", "AdvDataset", "RandDataset"]


def _add_attr(target_obj, source_obj):
    new_keys = [at for at in dir(source_obj) if at not in dir(target_obj)]
    for key in new_keys:
        setattr(target_obj, key, getattr(source_obj, key))


class PartialDataset(Dataset):
    def __init__(self, dataset, total=1, selected=0, seed=-1):
        import numpy as np
        assert 0 <= selected < total
        self.dataset = dataset
        self.perms = np.arange(0, len(dataset))
        self.seed = seed
        if seed != -1:
            np.random.seed(seed)
            np.random.shuffle(self.perms)
        self.each_len = [len(dataset) // total + (1 if i < len(dataset) % total else 0) for i in range(total)]
        self.starts = [0]
        for i in range(1, total):
            self.starts.append(self.starts[-1] + self.each_len[i])
        self.total = total
        self.selected = selected

    def __getitem__(self, index):
        index = index % self.each_len[self.selected]
        return self.dataset[self.starts[self.selected] + index]

    def __len__(self):
        return self.each_len[self.selected] * self.total


class OrderDataset(Dataset):
    def __init__(self, data):
        self.dataset = data
        self.classes = set(data.targets)
        print(f"totally {len(self.classes)} classes found in dataset!")
        self.target_by_label = {i: [] for i in self.classes}
        for idx, lb in enumerate(data.targets):
            self.target_by_label[lb].append(idx)
        self.subclass_len = min([len(self.target_by_label[i]) for i in self.classes])
        self.maps = [self.target_by_label[j][i] for i in range(self.subclass_len) for j in self.classes]

    def __getitem__(self, item):
        return self.dataset[self.maps[item]]

    def __len__(self):
        return self.subclass_len * len(self.classes)


class AdvDataset(Dataset):
    def __init__(self, dataset: cifar.CIFAR10, mean, std, model_path: str, attack_type: str, device=None):
        from model.utils import load_models
        from datasets.utils import AdvAttack
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.dataset = dataset
        model = load_models({"pretrain_paths": [model_path]})[0]
        self.adv_attack: torch.nn.Module = AdvAttack(model=model, attack_name=attack_type, mean=mean, std=std,
                                                     device=self.device)
        self.adv_attack.to(self.device)

    #     Pre Generate the dataset to speed up
        self.adv_imgs = self.pre_gen(batch_size=64)

    def pre_gen(self, batch_size=1):
        ret_imgs = []
        start = 0
        while start < len(self.dataset):
            end = min(start+batch_size, int(len(self.dataset)))
            imgs, targets = list(zip(*[self.dataset[i] for i in range(start, end)]))
            images = torch.stack(imgs).to(self.device)
            labels = torch.tensor(targets).to(self.device)
            assert images.dim() == 4
            with torch.enable_grad():
                adv_images = self.adv_attack(images, labels)
            assert adv_images.grad_fn is None
            ret_imgs.append(adv_images)
            start = end
        return torch.cat(ret_imgs).cpu()

    def __getitem__(self, item):
        _, target = self.dataset[item]
        img = self.adv_imgs[item]
        return img, target

    def __len__(self):
        return len(self.dataset)


class ConcatFullDataset(FullDatasetBase):
    def gen_train_transforms(self):
        print("Warning!!! This is a Concat dataset, all data is after transforms, querying transforms for this"
              "dataset is meaningless!!!")
        return self.all_datasets[0].gen_train_transforms

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return self.train

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return self.val

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return self.test

    @staticmethod
    def is_dataset_name(name: str):
        return name == "concat" or name == "Concat"

    def __init__(self, dataset_params, **kwargs):
        super().__init__(**kwargs)
        sub_datasets = dataset_params['sub_datasets']
        self.dataset_params = dataset_params
        self.all_datasets, self.all_dataset_params, trains, vals, tests = list(
            zip(*[list(datasets.dataProvider.DatasetFactory.build_dataset(d)) for d in sub_datasets]))

        from torch.utils.data import ConcatDataset
        self.train = ConcatDataset(trains)
        self.val = ConcatDataset(vals)
        self.test = ConcatDataset(tests)

        self.num_classes = self.all_datasets[0].num_classes
        self.mean = self.all_datasets[0].mean
        self.std = self.all_datasets[0].std
        self.img_shape = self.all_datasets[0].img_shape
        self.name = "ConcatDataset"


class RandDataset(Dataset):
    def __init__(self, dataset: cifar.CIFAR10, alpha=0):
        self.alpha = alpha
        self.dataset = dataset

    def __getitem__(self, item):
        img, target = self.dataset[item]
        # img is mean = 0, std = 1
        # print(img.min(), img.max(), img.mean())
        img = torch.randn_like(img) * self.alpha + img * (1 - self.alpha)
        # print(img.min(), img.max(), img.mean())
        return img, target

    def __len__(self):
        return len(self.dataset)