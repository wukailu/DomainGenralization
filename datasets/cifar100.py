from datasets.decorators import OrderDataset
from datasets.utils import FullDatasetBase, CIFAR100PATH
from torchvision.datasets import cifar
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset


class CIFAR100(FullDatasetBase):
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    img_shape = (3, 32, 32)
    num_classes = 100
    name = "cifar100"

    def gen_train_transforms(self):
        test_transforms, _ = self.gen_test_transforms()
        return transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                   transforms.RandomHorizontalFlip(),
                                   test_transforms]), _

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        return cifar.CIFAR100(root=CIFAR100PATH, train=True, download=True,
                              transform=transform, target_transform=target_transform)

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        return cifar.CIFAR100(root=CIFAR100PATH, train=False, download=True,
                              transform=transform, target_transform=target_transform)

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        return cifar.CIFAR100(root=CIFAR100PATH, train=False, download=True,
                              transform=transform, target_transform=target_transform)

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(Cifar|cifar|CIFAR)([-_])*100$", name)


class OrderedCIFAR100VAL(CIFAR100):
    name = "cifar100_withval"

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        ds = cifar.CIFAR100(root=CIFAR100PATH, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        return Subset(OrderDataset(ds), list(range(40000)))

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        ds = cifar.CIFAR100(root=CIFAR100PATH, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        return Subset(OrderDataset(ds), list(range(40000, 50000)))

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(Order|order|o|O)(Cifar|cifar|CIFAR)([-_])*100(val|VAL)$", name)



